/* global currentFrame, sampleRate */
// Capture one selected input channel. The output stays silent.
class GuitarCaptureProcessor extends AudioWorkletProcessor {
  constructor() {
    super();
    this.recording = false;
    this.samples = new Float32Array(4096);
    this.length = 0;
    this.started = false;
    this.port.onmessage = ({ data }) => {
      if (data.recording) {
        this.recording = true;
        this.length = 0;
        this.started = false;
      } else {
        this.recording = false;
        this.flush();
        this.port.postMessage({ type: 'stopped' });
      }
    };
  }

  flush() {
    if (!this.length) return;
    const samples = this.samples.slice(0, this.length);
    this.port.postMessage({ type: 'samples', samples }, [samples.buffer]);
    this.length = 0;
  }

  process(inputs) {
    const input = inputs[0]?.[0];
    if (!this.recording || !input) return true;
    if (!this.started) {
      this.started = true;
      this.port.postMessage({ type: 'start', frame: currentFrame });
    }
    for (let i = 0; i < input.length; i++) {
      this.samples[this.length++] = input[i];
      if (this.length === this.samples.length) this.flush();
    }
    return true;
  }
}

registerProcessor('qwertophone-guitar-capture', GuitarCaptureProcessor);

// Live-only mono gate, before distortion. The recording branch stays raw.
class NoiseGateProcessor extends AudioWorkletProcessor {
  static get parameterDescriptors() {
    return [
      {
        name: 'enabled',
        defaultValue: 0,
        minValue: 0,
        maxValue: 1,
        automationRate: 'k-rate',
      },
      {
        name: 'threshold',
        defaultValue: -50,
        minValue: -80,
        maxValue: -10,
        automationRate: 'k-rate',
      },
      {
        name: 'release',
        defaultValue: 150,
        minValue: 20,
        maxValue: 1000,
        automationRate: 'k-rate',
      },
    ];
  }

  constructor() {
    super();
    this.envelope = 0;
    this.gain = 0;
    this.open = false;
    this.disposed = false;
    this.detectorDecay = Math.exp(-1 / (0.02 * sampleRate));
    this.attack = Math.exp(-Math.log(1000) / (0.002 * sampleRate));
    this.port.onmessage = ({ data }) => {
      if (data.dispose) this.disposed = true;
    };
  }

  process(inputs, outputs, parameters) {
    if (this.disposed) return false;
    const input = inputs[0]?.[0];
    const output = outputs[0]?.[0];
    if (!output) return true;
    if (!input) {
      output.fill(0);
      this.envelope = 0;
      this.open = false;
      this.gain = 0;
      return true;
    }
    const enabled = parameters.enabled[0] >= 0.5;
    const threshold = Math.pow(10, parameters.threshold[0] / 20);
    // 6 dB hysteresis avoids flutter near the threshold and on zero crossings.
    const closeThreshold = threshold * 0.5;
    const release = Math.exp(
      -Math.log(1000) / ((parameters.release[0] * sampleRate) / 1000),
    );
    for (let i = 0; i < output.length; i++) {
      const sample = input[i] ?? 0;
      this.envelope = Math.max(
        Math.abs(sample),
        this.envelope * this.detectorDecay,
      );
      if (this.envelope >= threshold) this.open = true;
      else if (this.envelope < closeThreshold) this.open = false;
      if (!enabled) {
        this.gain = 1;
        output[i] = sample;
        continue;
      }
      const target = this.open ? 1 : 0;
      const coefficient = target > this.gain ? this.attack : release;
      this.gain = target + (this.gain - target) * coefficient;
      output[i] = sample * this.gain;
    }
    return true;
  }
}

registerProcessor('qwertophone-noise-gate', NoiseGateProcessor);
