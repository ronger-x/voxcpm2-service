// Shared by AudioWorklet and the HTTP-compatible fallback renderer.
export class PCMQueue {
  constructor(capacity, prefill) {
    this.samples = new Float32Array(capacity);
    this.prefill = Math.min(prefill, capacity);
    this.read = 0;
    this.write = 0;
    this.size = 0;
    this.playing = false;
    this.ended = false;
    this.underruns = 0;
  }

  push(samples) {
    if (samples.length > this.samples.length - this.size) throw new Error('音频缓冲区已满');
    const first = Math.min(samples.length, this.samples.length - this.write);
    this.samples.set(samples.subarray(0, first), this.write);
    this.samples.set(samples.subarray(first), 0);
    this.write = (this.write + samples.length) % this.samples.length;
    this.size += samples.length;
  }

  render(output) {
    output.fill(0);
    if (!this.playing && (this.size >= this.prefill || this.ended)) this.playing = true;
    if (!this.playing) return 0;
    const count = Math.min(output.length, this.size);
    const first = Math.min(count, this.samples.length - this.read);
    output.set(this.samples.subarray(this.read, this.read + first));
    output.set(this.samples.subarray(0, count - first), first);
    this.read = (this.read + count) % this.samples.length;
    this.size -= count;
    if (count < output.length && !this.ended) {
      this.playing = false;
      this.underruns++;
    }
    return count;
  }
}

// Fetch boundaries can fall between the two bytes of a PCM16 sample.
export class PCM16Decoder {
  constructor() { this.tail = null; }
  decode(bytes) {
    const result = new Float32Array(Math.floor((bytes.length + (this.tail !== null ? 1 : 0)) / 2));
    let offset = 0, index = 0;
    if (this.tail !== null && bytes.length) {
      const value = this.tail | (bytes[offset++] << 8);
      result[index++] = (value >= 32768 ? value - 65536 : value) / 32768;
      this.tail = null;
    }
    for (; offset + 1 < bytes.length; offset += 2) {
      const value = bytes[offset] | (bytes[offset + 1] << 8);
      result[index++] = (value >= 32768 ? value - 65536 : value) / 32768;
    }
    if (offset < bytes.length) this.tail = bytes[offset];
    return result;
  }
  finish() {
    if (this.tail !== null) throw new Error('音频流不完整，请重新生成');
  }
}
