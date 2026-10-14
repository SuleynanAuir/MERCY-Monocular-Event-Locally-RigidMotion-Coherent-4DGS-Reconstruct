/* Small bounded jitter buffer. Pure scheduling logic also runs in Node tests. */
(function (root) {
  "use strict";
  class GaussianPlayback {
    constructor(rate = 60, warmup = 10, limit = 24) {
      this.rate = rate; this.warmup = warmup; this.limit = limit;
      this.frames = new Map(); this.last = -1; this.end = null;
      this.start = null; this.buffering = true; this.underruns = 0; this.dropped = 0;
    }
    push(item) {
      const seq = item.meta.stream.seq;
      if (seq <= this.last || this.frames.has(seq) || this.frames.size >= this.limit) {
        item.bitmap.close(); return false;
      }
      this.frames.set(seq, item); return true;
    }
    finish(seq) { this.end = seq; }
    take(now) {
      if (this.end !== null && this.last >= this.end) return {ended: true};
      const first = this.last + 1;
      if (this.buffering) {
        const available = this.end === null ? this.warmup : Math.min(this.warmup, this.end - this.last);
        // Wait for contiguous decoded frames, not just network packet count.
        for (let i = first; i < first + available; i++) if (!this.frames.has(i)) return null;
        this.start = now - first * 1000 / this.rate; this.buffering = false;
      }
      // rAF timestamps jitter slightly around a display refresh. A strict floor
      // otherwise turns an early 33.2 ms tick into a 50 ms / 16 ms cadence at
      // 30 FPS. Allow <= 2 ms of early presentation, without changing the clock.
      const tolerance = Math.min(2, 100 / this.rate);
      const desired = Math.floor((now - this.start + tolerance) * this.rate / 1000 + 1e-5);
      if (desired <= this.last) return null;
      let selected = null;
      for (let seq = first; seq <= desired; seq++) {
        const item = this.frames.get(seq);
        if (!item) break;
        if (selected) {selected.bitmap.close(); this.dropped++;}
        selected = item; this.frames.delete(seq); this.last = seq;
      }
      if (!selected && (this.end === null || this.last < this.end)) {
        this.buffering = true; this.underruns++;
      }
      return selected;
    }
    close() {
      for (const item of this.frames.values()) item.bitmap.close();
      this.frames.clear();
    }
  }
  if (typeof module !== "undefined" && module.exports) module.exports = GaussianPlayback;
  else root.GaussianPlayback = GaussianPlayback;
})(globalThis);
