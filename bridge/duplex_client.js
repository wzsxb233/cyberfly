/* Native MiniCPM duplex client. Load as an ES module from the workbench. */
export class MiniCPMDuplexClient {
  constructor({url = 'ws://127.0.0.1:18645/v1/realtime?mode=audio', onEvent = () => {}} = {}) {
    this.url = url; this.onEvent = onEvent; this.sequence = 0;
    this.inflight = 0; this.pending = []; this.samples = [];
    this.sources = new Set(); this.playhead = 0; this.epoch = 0;
    this.cancelling = false; this.ready = false;
  }
  async open() {
    this.playback = new AudioContext({sampleRate: 24000});
    await this.playback.resume();
    this.socket = new WebSocket(this.url);
    return new Promise((resolve, reject) => {
      let resolved = false;
      this.socket.onerror = () => reject(new Error('无法连接本地 MiniCPM 流式语音'));
      this.socket.onclose = () => { this.ready = false; this.stopMicrophone(); this.stopPlayback(); };
      this.socket.onmessage = ({data}) => {
        let event;
        try { event = JSON.parse(data); } catch { reject(new Error('无效模型事件')); return; }
        if (event.type === 'session.queue_done') this.send({type: 'session.init', payload: {}});
        if (event.type === 'session.created') {
          this.epoch = event.epoch; this.ready = true;
          if (!resolved) { resolved = true; resolve(event); }
        }
        if (event.type === 'input.processed') { this.inflight = Math.max(0, this.inflight - 1); this.flush(); }
        if (event.type === 'response.cancelled') {
          this.cancelling = false; this.inflight = 0; this.samples = []; this.pending = []; this.stopPlayback();
        }
        if (event.type === 'response.output.delta' && event.kind === 'audio' && !this.cancelling && event.epoch === this.epoch) {
          this.play(event.audio);
        }
        if (event.type === 'error') { this.ready = false; this.stopMicrophone(); this.stopPlayback(); reject(new Error(event.error?.message || '模型流式语音失败')); }
        this.onEvent(event);
      };
    });
  }
  send(event) {
    if (this.socket?.readyState !== WebSocket.OPEN) throw new Error('语音会话尚未连接');
    this.socket.send(JSON.stringify(event));
  }
  appendPcm(frame) {
    if (!(frame instanceof Float32Array) || frame.length < 1600 || frame.length > 16000) throw new Error('每块必须是0.1至1秒、16k单声道Float32音频');
    if (!frame.every(value => Number.isFinite(value) && Math.abs(value) <= 1)) throw new Error('无效音频幅度');
    if (this.cancelling) return;
    if (this.pending.length >= 3) {
      this.onEvent({type: 'client.backpressure', message: '模型处理较慢，已丢弃最旧的待发音频块'});
      this.pending.shift();
    }
    const bytes = new Uint8Array(frame.buffer, frame.byteOffset, frame.byteLength);
    let binary = '';
    for (let offset = 0; offset < bytes.length; offset += 8192) binary += String.fromCharCode(...bytes.subarray(offset, offset + 8192));
    this.pending.push(btoa(binary)); this.flush();
  }
  flush() {
    while (this.ready && !this.cancelling && this.inflight < 3 && this.pending.length) {
      this.send({type: 'input.append', sequence: this.sequence++, input: {audio: this.pending.shift()}});
      this.inflight++;
    }
  }
  async startMicrophone() {
    if (!this.ready) throw new Error('先等待模型语音会话就绪');
    this.capture = new AudioContext({sampleRate: 16000});
    if (this.capture.sampleRate !== 16000) { await this.capture.close(); throw new Error('浏览器未提供16k采样率，请使用外部PCM输入'); }
    this.media = await navigator.mediaDevices.getUserMedia({audio: {channelCount: 1, echoCancellation: true, noiseSuppression: true}});
    await this.capture.resume();
    this.microphone = this.capture.createMediaStreamSource(this.media);
    // Kept dependency-free for the local toy. A worklet may replace capture
    // later; transport and sample boundaries remain identical.
    this.processor = this.capture.createScriptProcessor(4096, 1, 1);
    this.processor.onaudioprocess = event => {
      if (this.cancelling) { this.samples = []; return; }
      this.samples.push(...event.inputBuffer.getChannelData(0));
      while (this.samples.length >= 16000) this.appendPcm(new Float32Array(this.samples.splice(0, 16000)));
    };
    this.microphone.connect(this.processor); this.processor.connect(this.capture.destination);
  }
  play(encoded) {
    const raw = atob(encoded); const bytes = Uint8Array.from(raw, ch => ch.charCodeAt(0));
    const data = new Float32Array(bytes.buffer);
    const buffer = this.playback.createBuffer(1, data.length, 24000); buffer.copyToChannel(data, 0);
    const source = this.playback.createBufferSource(); source.buffer = buffer; source.connect(this.playback.destination);
    source.onended = () => this.sources.delete(source); this.sources.add(source);
    this.playhead = Math.max(this.playhead, this.playback.currentTime + 0.02);
    source.start(this.playhead); this.playhead += buffer.duration;
  }
  stopPlayback() {
    for (const source of this.sources) { try { source.stop(); } catch {} }
    this.sources.clear(); this.playhead = 0;
  }
  interrupt() {
    this.cancelling = true; this.pending = []; this.samples = []; this.stopPlayback();
    this.send({type: 'input.cancel'});
  }
  stopMicrophone() {
    this.processor?.disconnect(); this.microphone?.disconnect();
    for (const track of this.media?.getTracks() || []) track.stop();
    if (this.capture && this.capture.state !== 'closed') this.capture.close();
    this.samples = []; this.pending = [];
  }
  close() {
    this.stopMicrophone(); this.stopPlayback();
    if (this.socket?.readyState === WebSocket.OPEN) this.send({type: 'session.close'});
    this.ready = false;
    if (this.playback && this.playback.state !== 'closed') this.playback.close();
  }
}
