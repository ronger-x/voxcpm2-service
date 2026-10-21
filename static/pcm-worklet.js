import { PCMQueue } from './pcm-ring.mjs';

class PCMProcessor extends AudioWorkletProcessor {
  constructor({ processorOptions }) {
    super();
    this.queue = new PCMQueue(processorOptions.capacity, processorOptions.prefill);
    this.consumed = 0;
    this.ticks = 0;
    this.port.onmessage = ({ data }) => {
      if (data.type === 'samples') this.queue.push(data.samples);
      if (data.type === 'end') this.queue.ended = true;
    };
  }
  process(_inputs, outputs) {
    this.consumed += this.queue.render(outputs[0][0]);
    const drained = this.queue.ended && this.queue.size === 0;
    if (++this.ticks >= 32 || drained) {
      this.port.postMessage({ consumed: this.consumed, underruns: this.queue.underruns, drained });
      this.consumed = 0;
      this.ticks = 0;
    }
    return !drained;
  }
}

registerProcessor('voxcpm-pcm', PCMProcessor);
