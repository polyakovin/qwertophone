/* global AudioWorkletProcessor, registerProcessor, sampleRate */
// Phase-preserving spectral-envelope cross-synthesis. The carrier is the real
// input, never a pitched sample or oscillator. No pitch/note detector is used.
// Algorithm reference: ccrma.stanford.edu/~jos/sasp/Cross_Synthesis.html
class TransferFFT {
  constructor(size) {
    this.size = size;
    this.reversed = new Uint32Array(size);
    this.cos = new Float64Array(size / 2);
    this.sin = new Float64Array(size / 2);
    for (let i = 1, j = 0; i < size; i++) {
      let bit = size >> 1;
      while (j & bit) {
        j ^= bit;
        bit >>= 1;
      }
      j ^= bit;
      this.reversed[i] = j;
    }
    for (let i = 0; i < size / 2; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / size);
      this.sin[i] = Math.sin((2 * Math.PI * i) / size);
    }
  }
  transform(real, imaginary, inverse = false) {
    const size = this.size;
    for (let i = 0; i < size; i++) {
      const j = this.reversed[i];
      if (i < j) {
        let value = real[i];
        real[i] = real[j];
        real[j] = value;
        value = imaginary[i];
        imaginary[i] = imaginary[j];
        imaginary[j] = value;
      }
    }
    for (let span = 2; span <= size; span *= 2) {
      const half = span / 2,
        stride = size / span;
      for (let block = 0; block < size; block += span) {
        for (let i = 0; i < half; i++) {
          const a = block + i,
            b = a + half;
          const wr = this.cos[i * stride];
          const wi = this.sin[i * stride] * (inverse ? 1 : -1);
          const tr = wr * real[b] - wi * imaginary[b];
          const ti = wr * imaginary[b] + wi * real[b];
          real[b] = real[a] - tr;
          imaginary[b] = imaginary[a] - ti;
          real[a] += tr;
          imaginary[a] += ti;
        }
      }
    }
    if (inverse)
      for (let i = 0; i < size; i++) {
        real[i] /= size;
        imaginary[i] /= size;
      }
  }
}

class TimbreTransferProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'amount',
        defaultValue: 1,
        minValue: 0,
        maxValue: 1,
        automationRate: 'k-rate',
      },
    ];
  }
  constructor(options) {
    super();
    const frames = options.processorOptions.frames;
    const size = frames[0]?.length;
    if (
      ![2048, 4096].includes(size) ||
      !frames.length ||
      frames.some(frame => frame.length !== size)
    )
      throw new Error('Invalid timbre reference');
    this.size = size;
    this.hop = size / 4;
    this.fft = new TransferFFT(size);
    this.input = new Float32Array(size);
    this.output = new Float32Array(size * 2);
    this.real = new Float64Array(size);
    this.imaginary = new Float64Array(size);
    this.envelope = new Float64Array(size);
    this.scratch = new Float64Array(size);
    this.target = new Float64Array(size / 2 + 1);
    this.window = Float64Array.from({ length: size }, (_, i) =>
      Math.sqrt(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / size)),
    );
    this.received = 0;
    this.dryEnergy = 0;
    this.wetEnergy = 0;
    this.envelopeDecay = Math.exp(-1 / (sampleRate * 0.005));
    this.disposed = false;
    // Average reference power, then remove its harmonic comb with a smoothly
    // tapered 1.5 ms cepstral lifter. Only the spectral colour is retained.
    for (const frame of frames) {
      for (let i = 0; i < size; i++) {
        this.real[i] = frame[i] * this.window[i];
        this.imaginary[i] = 0;
      }
      this.fft.transform(this.real, this.imaginary);
      for (let k = 0; k <= size / 2; k++)
        this.target[k] +=
          (this.real[k] ** 2 + this.imaginary[k] ** 2) / frames.length;
    }
    for (let k = 0; k <= size / 2; k++)
      this.envelope[k] = Math.sqrt(this.target[k]);
    this.smoothEnvelope();
    for (let k = 0; k <= size / 2; k++) this.target[k] = this.envelope[k];
    this.port.onmessage = ({ data }) => {
      if (data.dispose) this.disposed = true;
    };
  }
  smoothEnvelope() {
    const n = this.size;
    let maximum = 0;
    for (let k = 0; k <= n / 2; k++)
      maximum = Math.max(maximum, this.envelope[k]);
    const floor = Math.max(1e-12, maximum * 1e-4);
    for (let k = 0; k <= n / 2; k++)
      this.envelope[k] = Math.log(Math.max(floor, this.envelope[k]));
    for (let k = 1; k < n / 2; k++) this.envelope[n - k] = this.envelope[k];
    this.scratch.fill(0);
    this.fft.transform(this.envelope, this.scratch, true);
    const cutoff = Math.round(sampleRate * 0.0015);
    for (let i = 1; i < n; i++) {
      const q = Math.min(i, n - i);
      this.envelope[i] *=
        q < cutoff ? 0.5 + 0.5 * Math.cos((Math.PI * q) / cutoff) : 0;
    }
    // Removing c[0] normalizes overall level, leaving only spectral shape.
    this.envelope[0] = 0;
    this.scratch.fill(0);
    this.fft.transform(this.envelope, this.scratch);
  }
  renderFrame(amount) {
    const n = this.size,
      half = n / 2;
    let energy = 0;
    for (let i = 0; i < n; i++) {
      const value = this.input[(this.received + i) % n] * this.window[i];
      this.real[i] = value;
      this.imaginary[i] = 0;
      energy += value * value;
    }
    if (energy < 1e-16) return;
    this.fft.transform(this.real, this.imaginary);
    if (amount > 0) {
      for (let k = 0; k <= half; k++)
        this.envelope[k] = Math.hypot(this.real[k], this.imaginary[k]);
      this.smoothEnvelope();
      let before = 0,
        after = 0;
      for (let k = 0; k <= half; k++) {
        const gain = Math.exp(
          Math.max(
            -1.8,
            Math.min(1.8, (this.target[k] - this.envelope[k]) * amount),
          ),
        );
        const power = this.real[k] ** 2 + this.imaginary[k] ** 2;
        const weight = k === 0 || k === half ? 1 : 2;
        before += power * weight;
        after += power * gain * gain * weight;
        this.real[k] *= gain;
        this.imaginary[k] *= gain;
        if (k > 0 && k < half) {
          this.real[n - k] *= gain;
          this.imaginary[n - k] *= gain;
        }
      }
      const normalization = Math.sqrt(before / Math.max(1e-20, after));
      for (let i = 0; i < n; i++) {
        this.real[i] *= normalization;
        this.imaginary[i] *= normalization;
      }
    }
    this.fft.transform(this.real, this.imaginary, true);
    // sqrt-Hann analysis/synthesis and 75% overlap sum to 2. A fixed N-sample
    // delay makes all contributions causal, independent of render block size.
    for (let i = 0; i < n; i++)
      this.output[(this.received + i) % this.output.length] +=
        (this.real[i] * this.window[i]) / 2;
  }
  process(inputs, outputs, parameters) {
    const output = outputs[0]?.[0];
    if (!output) return !this.disposed;
    if (this.disposed) {
      output.fill(0);
      return false;
    }
    const input = inputs[0]?.[0];
    const amount = parameters.amount[0];
    for (let i = 0; i < output.length; i++) {
      const position = this.received;
      const dry = this.input[position % this.size];
      const value = input?.[i] ?? 0;
      this.input[position % this.size] = Number.isFinite(value) ? value : 0;
      const slot = position % this.output.length;
      let wet = this.output[slot];
      this.output[slot] = 0;
      // Restore the incoming loudness envelope after spectral shaping, using
      // the delayed input, so quiet playing stays quiet and bends stay smooth.
      this.dryEnergy =
        this.envelopeDecay * this.dryEnergy +
        (1 - this.envelopeDecay) * dry * dry;
      this.wetEnergy =
        this.envelopeDecay * this.wetEnergy +
        (1 - this.envelopeDecay) * wet * wet;
      const gain =
        this.wetEnergy > 1e-14
          ? Math.max(
              0.25,
              Math.min(4, Math.sqrt(this.dryEnergy / this.wetEnergy)),
            )
          : 1;
      wet *= gain;
      output[i] = Number.isFinite(wet) ? wet : 0;
      this.received++;
      if (this.received % this.hop === 0) this.renderFrame(amount);
    }
    return true;
  }
}
registerProcessor('qwertophone-timbre-transfer', TimbreTransferProcessor);
