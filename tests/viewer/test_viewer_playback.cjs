const {test} = require('node:test');
const assert = require('node:assert/strict');
const Playback = require('../../viewer/playback.js');
const item = seq => ({meta: {stream: {seq}}, bitmap: {closed: false, close() {this.closed = true;}}});

test('prebuffers contiguous frames and presents at evenly spaced times', () => {
  const buffer = new Playback(60, 10, 24);
  for (let i = 0; i < 9; i++) buffer.push(item(i));
  assert.equal(buffer.take(100), null);
  buffer.push(item(9));
  assert.equal(buffer.take(200).meta.stream.seq, 0);
  assert.equal(buffer.take(208), null);
  for (let i = 1; i < 10; i++) assert.equal(buffer.take(200 + i * 1000 / 60).meta.stream.seq, i);
  assert.equal(buffer.underruns, 0);
});
test('out of order decoding does not skip a missing startup frame', () => {
  const buffer = new Playback(60, 3);
  buffer.push(item(1)); buffer.push(item(2)); buffer.push(item(3));
  assert.equal(buffer.take(0), null);
  buffer.push(item(0)); assert.equal(buffer.take(20).meta.stream.seq, 0);
});
test('browser scheduling lag drops only outdated frames and releases bitmaps', () => {
  const buffer = new Playback(60, 2), frames = Array.from({length: 10}, (_, i) => item(i));
  frames.forEach(frame => buffer.push(frame));
  buffer.take(0).bitmap.close();
  assert.equal(buffer.take(50).meta.stream.seq, 3);
  assert.equal(buffer.dropped, 2); assert.equal(frames[1].bitmap.closed, true);
  buffer.close(); assert.equal(buffer.frames.size, 0); assert.equal(frames[9].bitmap.closed, true);
});
test('short sequences finish without waiting for ten frames', () => {
  const buffer = new Playback(60, 10);
  buffer.push(item(0)); buffer.push(item(1)); buffer.finish(1);
  assert.equal(buffer.take(0).meta.stream.seq, 0);
  assert.equal(buffer.take(1000 / 60).meta.stream.seq, 1);
  assert.deepEqual(buffer.take(34), {ended: true});
});
test('buffer underrun waits, then resumes without accelerating the animation', () => {
  const buffer = new Playback(60, 2);
  buffer.push(item(0)); buffer.push(item(1));
  buffer.take(0); buffer.take(17);
  assert.equal(buffer.take(34), null); assert.equal(buffer.underruns, 1);
  buffer.push(item(2)); buffer.push(item(3));
  assert.equal(buffer.take(200).meta.stream.seq, 2);
  assert.equal(buffer.take(210), null);
  assert.equal(buffer.take(217).meta.stream.seq, 3);
});
test('bounded storage rejects duplicate and excess frames', () => {
  const buffer = new Playback(60, 1, 2), duplicate = item(0), extra = item(2);
  buffer.push(item(0)); buffer.push(item(1));
  assert.equal(buffer.push(duplicate), false); assert.equal(duplicate.bitmap.closed, true);
  assert.equal(buffer.push(extra), false); assert.equal(extra.bitmap.closed, true);
  buffer.close();
});
test('sub-millisecond rAF jitter does not cause alternating long and short frame holds', () => {
  const buffer = new Playback(30, 2);
  for (let i = 0; i < 10; i++) buffer.push(item(i));
  assert.equal(buffer.take(0).meta.stream.seq, 0);
  for (let i = 1; i < 10; i++) {
    assert.equal(buffer.take(i * 1000 / 30 - .4).meta.stream.seq, i);
    assert.equal(buffer.take(i * 1000 / 30 + 16), null);
  }
  assert.equal(buffer.dropped, 0); assert.equal(buffer.underruns, 0);
});
test('a 400 ms buffer absorbs a 200 ms network burst gap without a stop', () => {
  const buffer = new Playback(30, 12);
  for (let i = 0; i < 12; i++) buffer.push(item(i));
  for (let i = 0; i < 30; i++) {
    if (i === 6) for (let j = 12; j < 24; j++) buffer.push(item(j));
    if (i === 18) for (let j = 24; j < 36; j++) buffer.push(item(j));
    assert.equal(buffer.take(i * 1000 / 30).meta.stream.seq, i);
  }
  assert.equal(buffer.underruns, 0); assert.equal(buffer.dropped, 0);
});
