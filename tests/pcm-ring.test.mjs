import assert from 'node:assert/strict';
import { test } from 'node:test';
import { PCMQueue, PCM16Decoder } from '../static/pcm-ring.mjs';

test('wraparound preserves sample order and overflow cannot overwrite audio', () => {
  const ring = new PCMQueue(8, 2), out = new Float32Array(4);
  ring.push(Float32Array.of(1, 2, 3, 4, 5, 6));
  assert.equal(ring.render(out), 4);
  assert.deepEqual([...out], [1, 2, 3, 4]);
  ring.push(Float32Array.of(7, 8, 9, 10, 11, 12));
  assert.throws(() => ring.push(Float32Array.of(13)), /已满/);
  ring.render(out);
  assert.deepEqual([...out], [5, 6, 7, 8]);
  ring.render(out);
  assert.deepEqual([...out], [9, 10, 11, 12]);
});

test('prefill, underrun, refill and final short tail', () => {
  const ring = new PCMQueue(8, 4), out = new Float32Array(3);
  ring.push(Float32Array.of(1, 2));
  assert.equal(ring.render(out), 0);
  assert.equal(ring.underruns, 0);
  ring.push(Float32Array.of(3, 4));
  ring.render(out);
  assert.deepEqual([...out], [1, 2, 3]);
  ring.render(out);
  assert.deepEqual([...out], [4, 0, 0]);
  assert.equal(ring.underruns, 1);
  ring.push(Float32Array.of(5, 6));
  assert.equal(ring.render(out), 0);
  ring.ended = true;
  ring.render(out);
  assert.deepEqual([...out], [5, 6, 0]);
  assert.equal(ring.size, 0);
  assert.equal(ring.underruns, 1);
});

test('PCM16 decoding survives every possible network split and empty reads', () => {
  const bytes = Uint8Array.of(0, 128, 255, 127, 0, 0, 255, 255, 1, 0);
  const expected = [-1, 32767 / 32768, 0, -1 / 32768, 1 / 32768];
  for (let split = 0; split <= bytes.length; split++) {
    const decoder = new PCM16Decoder();
    const actual = [...decoder.decode(bytes.slice(0, split)), ...decoder.decode(new Uint8Array()),
                    ...decoder.decode(bytes.slice(split))];
    decoder.finish();
    assert.deepEqual(actual, expected);
  }
  const broken = new PCM16Decoder();
  broken.decode(Uint8Array.of(1));
  assert.throws(() => broken.finish(), /不完整/);
});
