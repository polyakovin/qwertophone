/* global AudioWorkletProcessor, registerProcessor, sampleRate */
// Audio-rate controls for one continuously running sample loop. Output 0 is
// playbackRate; output 1 is the input RMS envelope. Neither output is audible.
class TimbreLoopFollower extends AudioWorkletProcessor {
  constructor(options) {
    super();
    this.reference = options.processorOptions.referenceFrequency;
    this.rate = 16000;
    this.size = 768; // 48 ms at the analysis rate, independent of device rate.
    this.ring = new Float32Array(this.size);
    this.frame = new Float32Array(this.size);
    this.difference = new Float32Array(294);
    this.samples = 0;
    this.generated = 0;
    this.nextSample = 0;
    this.filtered = 0;
    this.filtered2 = 0;
    this.previous = 0;
    this.filter = 1 - Math.exp((-2 * Math.PI * 4500) / sampleRate);
    this.energy = 0;
    this.levelDecay = Math.exp(-1 / (sampleRate * 0.015));
    this.pitchDecay = Math.exp(-1 / (sampleRate * 0.004));
    this.gain = 0;
    this.attack = Math.exp(-Math.log(1000) / (sampleRate * 0.003));
    this.release = Math.exp(-Math.log(1000) / (sampleRate * 0.03));
    this.quietSamples = sampleRate;
    this.silent = true;
    this.frequency = 220;
    this.target = 220;
    this.candidate = 0;
    this.confirmations = 0;
    this.missing = 3;
    this.disposed = false;
    this.port.onmessage = ({ data }) => {
      if (data.dispose) this.disposed = true;
    };
  }
  detect() {
    let mean = 0,
      energy = 0;
    for (let i = 0; i < this.size; i++) {
      const value = this.ring[(this.generated + i) % this.size];
      this.frame[i] = value;
      mean += value / this.size;
    }
    for (let i = 0; i < this.size; i++) {
      this.frame[i] -= mean;
      energy += this.frame[i] ** 2;
    }
    const rms = Math.sqrt(energy / this.size);
    if (rms < 0.0015) return null;
    let cumulative = 0;
    for (let lag = 1; lag <= 292; lag++) {
      let sum = 0;
      for (let i = 0; i < 384; i++)
        sum += (this.frame[i] - this.frame[i + lag]) ** 2;
      cumulative += sum;
      this.difference[lag] = cumulative ? (sum * lag) / cumulative : 1;
    }
    for (let lag = 8; lag < 291; lag++) {
      if (this.difference[lag] >= 0.12) continue;
      while (lag + 1 < 292 && this.difference[lag + 1] < this.difference[lag])
        lag++;
      const denominator =
        this.difference[lag - 1] -
        2 * this.difference[lag] +
        this.difference[lag + 1];
      const shift = denominator
        ? (this.difference[lag - 1] - this.difference[lag + 1]) /
          (2 * denominator)
        : 0;
      const frequency = this.rate / (lag + Math.max(-1, Math.min(1, shift)));
      if (frequency < 55 || frequency > 2000) return null;
      let real = 0,
        imaginary = 0;
      for (let i = 0; i < this.size; i++) {
        const w = 0.5 - 0.5 * Math.cos((2 * Math.PI * i) / this.size);
        real +=
          this.frame[i] *
          w *
          Math.cos((2 * Math.PI * frequency * i) / this.rate);
        imaginary +=
          this.frame[i] *
          w *
          Math.sin((2 * Math.PI * frequency * i) / this.rate);
      }
      // A common chord period must not invent an absent fundamental.
      return (Math.hypot(real, imaginary) * 4) / this.size >= rms * 0.12
        ? frequency
        : null;
    }
    return null;
  }
  analyze() {
    if (this.silent) {
      this.missing = 3;
      return;
    }
    const frequency = this.detect();
    if (frequency) {
      if (
        this.candidate &&
        Math.abs(1200 * Math.log2(frequency / this.candidate)) < 80
      )
        this.confirmations++;
      else this.confirmations = 1;
      this.candidate = frequency;
      if (
        this.confirmations >= 2 ||
        (this.missing === 0 &&
          Math.abs(1200 * Math.log2(frequency / this.target)) < 700)
      ) {
        this.target = frequency;
        this.missing = 0;
      }
    } else {
      this.missing++;
      this.confirmations = 0;
      this.candidate = 0;
    }
  }
  process(inputs, outputs) {
    const pitch = outputs[0]?.[0],
      envelope = outputs[1]?.[0];
    if (!pitch || !envelope) return !this.disposed;
    if (this.disposed) {
      pitch.fill(0);
      envelope.fill(0);
      return false;
    }
    const input = inputs[0]?.[0];
    for (let i = 0; i < pitch.length; i++) {
      const value = Number.isFinite(input?.[i]) ? input[i] : 0;
      if (Math.abs(value) > 0.0005) {
        if (this.silent) {
          this.ring.fill(0);
          this.generated = 0;
          this.missing = 3;
          this.confirmations = 0;
          this.candidate = 0;
          this.silent = false;
        }
        this.quietSamples = 0;
      } else if (++this.quietSamples > sampleRate * 0.02) this.silent = true;
      this.energy =
        this.levelDecay * this.energy + (1 - this.levelDecay) * value * value;
      this.filtered += this.filter * (value - this.filtered);
      this.filtered2 += this.filter * (this.filtered - this.filtered2);
      while (this.nextSample <= this.samples) {
        const fraction = Math.max(
          0,
          Math.min(1, this.nextSample - this.samples + 1),
        );
        this.ring[this.generated % this.size] =
          this.previous + (this.filtered2 - this.previous) * fraction;
        this.generated++;
        this.nextSample += sampleRate / this.rate;
        if (this.generated >= this.size && this.generated % 160 === 0)
          this.analyze();
      }
      this.previous = this.filtered2;
      this.samples++;
      this.frequency =
        this.pitchDecay * this.frequency + (1 - this.pitchDecay) * this.target;
      const rms = Math.sqrt(this.energy);
      const target =
        !this.silent &&
        this.quietSamples < sampleRate * 0.005 &&
        this.missing < 2 &&
        rms > 0.0015
          ? rms
          : 0;
      const decay = target > this.gain ? this.attack : this.release;
      this.gain = decay * this.gain + (1 - decay) * target;
      pitch[i] = this.frequency / this.reference;
      envelope[i] = this.gain < 1e-7 ? 0 : this.gain;
    }
    return true;
  }
}
registerProcessor('qwertophone-timbre-loop', TimbreLoopFollower);
