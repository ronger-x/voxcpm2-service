import { PCMQueue, PCM16Decoder } from './pcm-ring.mjs';

const $ = (id) => document.getElementById(id);
let active = null;
let downloadURL = null;

class StreamingPlayer {
  constructor(context, signal, report) {
    this.context = context;
    this.signal = signal;
    this.report = report;
    this.capacity = context.sampleRate * 30;
    this.prefill = Math.ceil(context.sampleRate * 0.6);
    this.pending = 0;
    this.waiters = [];
    this.drained = false;
    signal.addEventListener('abort', () => this.wake(), { once: true });
  }
  wake() { this.waiters.splice(0).forEach((resolve) => resolve()); }
  update({ consumed, underruns, drained }) {
    this.pending -= consumed;
    this.drained = drained;
    this.report(this.pending / this.context.sampleRate, underruns);
    this.wake();
  }
  async init() {
    const options = { capacity: this.capacity, prefill: this.prefill };
    if (this.context.audioWorklet) {
      await this.context.audioWorklet.addModule('/player-assets/pcm-worklet.js');
      this.node = new AudioWorkletNode(this.context, 'voxcpm-pcm', {
        numberOfInputs: 0, numberOfOutputs: 1, outputChannelCount: [1], processorOptions: options,
      });
      this.node.port.onmessage = ({ data }) => this.update(data);
      this.node.onprocessorerror = () => {
        this.error = new Error('播放器运行失败，请重新生成');
        this.wake();
      };
    } else {
      // AudioWorklet requires HTTPS or localhost. LAN HTTP uses the same ring.
      this.queue = new PCMQueue(this.capacity, this.prefill);
      this.node = this.context.createScriptProcessor(2048, 0, 1);
      this.node.onaudioprocess = ({ outputBuffer }) => {
        const consumed = this.queue.render(outputBuffer.getChannelData(0));
        this.update({ consumed, underruns: this.queue.underruns, drained: this.queue.ended && this.queue.size === 0 });
      };
    }
    this.node.connect(this.context.destination);
  }
  check() {
    this.signal.throwIfAborted();
    if (this.error) throw this.error;
  }
  async write(samples) {
    // Limit individual messages so a large HTTP chunk never overflows the ring.
    for (let offset = 0; offset < samples.length; offset += 16384) {
      const part = samples.slice(offset, offset + 16384);
      while (this.pending + part.length > this.capacity) {
        this.check();
        await new Promise((resolve) => this.waiters.push(resolve));
      }
      this.check();
      this.pending += part.length;
      if (this.queue) this.queue.push(part);
      else this.node.port.postMessage({ type: 'samples', samples: part }, [part.buffer]);
    }
  }
  async finish() {
    if (this.queue) this.queue.ended = true;
    else this.node.port.postMessage({ type: 'end' });
    while (!this.drained) {
      this.check();
      await new Promise((resolve) => this.waiters.push(resolve));
    }
    // The last rendered block still has to reach the audio device before close.
    const latency = (this.context.baseLatency || 0) + (this.context.outputLatency || 0) + 2048 / this.context.sampleRate;
    await new Promise((resolve) => setTimeout(resolve, Math.ceil(latency * 1000)));
    this.check();
  }
  async close() {
    this.node?.disconnect();
    if (this.node?.port) this.node.port.close();
    if (this.queue && this.node) this.node.onaudioprocess = null;
    await this.context.close();
  }
}

function waveBlob(chunks, sampleRate, size) {
  const header = new ArrayBuffer(44), view = new DataView(header);
  const text = (offset, value) => [...value].forEach((char, i) => view.setUint8(offset + i, char.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, 36 + size, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, 1, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * 2, true);
  view.setUint16(32, 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, size, true);
  return new Blob([header, ...chunks], { type: 'audio/wav' });
}

async function referenceData(file) {
  if (!file) return undefined;
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result);
    reader.onerror = () => reject(reader.error);
    reader.readAsDataURL(file);
  });
}

$('stop').addEventListener('click', () => active?.abort());
$('speech-form').addEventListener('submit', async (event) => {
  event.preventDefault();
  if (active) return;
  const controller = new AbortController();
  active = controller;
  $('play').disabled = true;
  $('stop').disabled = false;
  $('download').hidden = true;
  if (downloadURL) URL.revokeObjectURL(downloadURL);
  $('status').textContent = '正在生成，首次使用需要加载模型…';
  let player, context, reader;
  let received = 0, first = null, networkDone = false;
  const started = performance.now();
  try {
    context = new AudioContext({ sampleRate: 48000 });
    // Resume synchronously in the click gesture before network/file awaits.
    await context.resume();
    player = new StreamingPlayer(context, controller.signal, (buffer, underruns) => {
      if (first === null) return;
      const phase = networkDone ? '生成完成，正在播放' : '正在生成并播放';
      $('status').textContent = `${phase} · 缓冲 ${buffer.toFixed(1)} 秒 · 首段 ${first.toFixed(2)} 秒 · 重新缓冲 ${underruns} 次`;
    });
    await player.init();
    const body = {
      input: $('text').value, instructions: $('instructions').value,
      inference_timesteps: Number($('steps').value), split_sentences: $('split').checked,
      response_format: 'pcm', sample_rate: context.sampleRate, stream: true,
      reference_audio: await referenceData($('reference').files[0]),
    };
    const response = await fetch('/v1/audio/speech', {
      method: 'POST', headers: { 'Content-Type': 'application/json', Authorization: `Bearer ${$('api-key').value.trim()}` },
      body: JSON.stringify(body), signal: controller.signal,
    });
    if (!response.ok) {
      const error = await response.json().catch(() => null);
      throw new Error(typeof error?.detail === 'string' ? error.detail : `请求失败（${response.status}）`);
    }
    if (Number(response.headers.get('X-Sample-Rate')) !== context.sampleRate) throw new Error('音频采样率与播放器不一致');
    reader = response.body.getReader();
    const chunks = [], decoder = new PCM16Decoder();
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      if (first === null) first = (performance.now() - started) / 1000;
      chunks.push(value);
      received += value.length;
      await player.write(decoder.decode(value));
    }
    decoder.finish();
    if (!received) throw new Error('服务未返回音频');
    networkDone = true;
    downloadURL = URL.createObjectURL(waveBlob(chunks, context.sampleRate, received));
    $('download').href = downloadURL;
    $('download').hidden = false;
    await player.finish();
    $('status').textContent = `播放完成 · 音频 ${(received / 2 / context.sampleRate).toFixed(1)} 秒 · 首段 ${first.toFixed(2)} 秒`;
  } catch (error) {
    $('status').textContent = controller.signal.aborted ? '已停止' : `生成失败：${error.message}`;
  } finally {
    controller.abort();
    await reader?.cancel().catch(() => {});
    if (player) await player.close();
    else if (context) await context.close();
    active = null;
    $('play').disabled = false;
    $('stop').disabled = true;
  }
});
