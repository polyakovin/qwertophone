/* global currentFrame */
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
