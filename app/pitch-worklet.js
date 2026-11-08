// Live F0 estimate for the microphone. Normalised autocorrelation over a 40 ms window every
// 10 ms. Good enough to draw a contour while you speak; the authoritative comparison is done
// server-side with the same tracker that produced the reference.
class PitchProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.win = Math.round(sampleRate * 0.04);
    this.hop = Math.round(sampleRate * 0.01);
    this.buf = new Float32Array(this.win * 2);
    this.filled = 0;
    this.sinceHop = 0;
    this.minLag = Math.floor(sampleRate / 500);
    this.maxLag = Math.ceil(sampleRate / 60);
  }

  estimate(frame) {
    let energy = 0;
    for (let i = 0; i < frame.length; i++) energy += frame[i] * frame[i];
    const rms = Math.sqrt(energy / frame.length);
    if (rms < 0.01) return 0;

    let bestLag = 0, best = 0;
    for (let lag = this.minLag; lag <= this.maxLag; lag++) {
      let num = 0, d1 = 0, d2 = 0;
      for (let i = 0; i + lag < frame.length; i++) {
        num += frame[i] * frame[i + lag];
        d1 += frame[i] * frame[i];
        d2 += frame[i + lag] * frame[i + lag];
      }
      const r = num / (Math.sqrt(d1 * d2) + 1e-9);
      if (r > best) { best = r; bestLag = lag; }
    }
    if (best < 0.6 || bestLag === 0) return 0;
    return sampleRate / bestLag;
  }

  process(inputs) {
    const ch = inputs[0] && inputs[0][0];
    if (!ch) return true;
    for (let i = 0; i < ch.length; i++) {
      if (this.filled < this.buf.length) {
        this.buf[this.filled++] = ch[i];
      } else {
        this.buf.copyWithin(0, 1);
        this.buf[this.buf.length - 1] = ch[i];
      }
      if (++this.sinceHop >= this.hop && this.filled >= this.win) {
        this.sinceHop = 0;
        const frame = this.buf.subarray(this.filled - this.win, this.filled);
        this.port.postMessage({ t: currentTime, f0: this.estimate(frame) });
      }
    }
    return true;
  }
}
registerProcessor('pitch-processor', PitchProcessor);
