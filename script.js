const reference = {
  jitter: 0.00494,
  shimmer: 0.02297,
  ppe: 0.19405,
};

const elements = {
  manualTab: document.querySelector("#manual-tab"),
  voiceTab: document.querySelector("#voice-tab"),
  manualMode: document.querySelector("#manual-mode"),
  voiceMode: document.querySelector("#voice-mode"),
  date: document.querySelector("#screening-date"),
  jitter: document.querySelector("#jitter"),
  shimmer: document.querySelector("#shimmer"),
  ppe: document.querySelector("#ppe"),
  duration: document.querySelector("#duration"),
  recordingStatus: document.querySelector("#recording-status"),
  resultCard: document.querySelector("#result-card"),
  riskCard: document.querySelector("#risk-card"),
  resultText: document.querySelector("#result-text"),
  riskText: document.querySelector("#risk-text"),
  confidenceText: document.querySelector("#confidence-text"),
  ndiText: document.querySelector("#ndi-text"),
  notesText: document.querySelector("#notes-text"),
  featureChart: document.querySelector("#feature-chart"),
  waveChart: document.querySelector("#wave-chart"),
  printReport: document.querySelector("#print-report"),
};

function setToday() {
  const today = new Date();
  const value = today.toISOString().slice(0, 10);
  elements.date.value = value;
}

function switchMode(mode) {
  const manual = mode === "manual";
  elements.manualTab.classList.toggle("active", manual);
  elements.voiceTab.classList.toggle("active", !manual);
  elements.manualMode.classList.toggle("active", manual);
  elements.voiceMode.classList.toggle("active", !manual);
}

function classifyNdi(ndi) {
  if (ndi < 0.12) return "Normal";
  if (ndi < 0.16) return "Mild";
  return "Severe";
}

function evaluateSample(jitter, shimmer, ppe) {
  const ndi = (jitter + shimmer + ppe) / 3;
  const weightedScore =
    jitter / reference.jitter * 0.26 +
    shimmer / reference.shimmer * 0.28 +
    ppe / reference.ppe * 0.46;

  let risk = "Low";
  let summary = "No Parkinson pattern detected";

  if (weightedScore >= 1.75 || ndi >= 0.16) {
    risk = "High";
    summary = "Parkinson-like pattern detected";
  } else if (weightedScore >= 1.25 || ndi >= 0.12) {
    risk = "Moderate";
    summary = "Review recommended";
  }

  const confidence = Math.max(42, Math.min(96, 48 + Math.abs(weightedScore - 1) * 28));

  return {
    confidence,
    ndi,
    ndiCondition: classifyNdi(ndi),
    risk,
    summary,
    weightedScore,
  };
}

function drawFeatureChart(values) {
  const canvas = elements.featureChart;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = { top: 34, right: 28, bottom: 58, left: 58 };
  const labels = ["Jitter", "Shimmer", "PPE"];
  const refs = [reference.jitter, reference.shimmer, reference.ppe];
  const max = Math.max(...values, ...refs) * 1.25 || 1;
  const plotWidth = width - padding.left - padding.right;
  const plotHeight = height - padding.top - padding.bottom;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);

  ctx.strokeStyle = "#e6eaf0";
  ctx.lineWidth = 1;
  ctx.font = "24px Segoe UI";
  ctx.fillStyle = "#172033";
  ctx.fillText("Current sample vs dataset median", padding.left, 28);

  for (let i = 0; i <= 4; i += 1) {
    const y = padding.top + (plotHeight / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(width - padding.right, y);
    ctx.stroke();
  }

  labels.forEach((label, index) => {
    const groupX = padding.left + (plotWidth / labels.length) * index + plotWidth / labels.length / 2;
    const barWidth = 42;
    const refHeight = (refs[index] / max) * plotHeight;
    const valueHeight = (values[index] / max) * plotHeight;
    const baseY = padding.top + plotHeight;

    ctx.fillStyle = "#d8dee8";
    ctx.fillRect(groupX - barWidth - 5, baseY - refHeight, barWidth, refHeight);

    ctx.fillStyle = "#147d73";
    ctx.fillRect(groupX + 5, baseY - valueHeight, barWidth, valueHeight);

    ctx.fillStyle = "#475467";
    ctx.font = "20px Segoe UI";
    ctx.textAlign = "center";
    ctx.fillText(label, groupX, height - 20);
  });

  ctx.textAlign = "left";
  ctx.font = "18px Segoe UI";
  ctx.fillStyle = "#d8dee8";
  ctx.fillRect(width - 255, 20, 18, 18);
  ctx.fillStyle = "#475467";
  ctx.fillText("Dataset median", width - 228, 35);
  ctx.fillStyle = "#147d73";
  ctx.fillRect(width - 255, 48, 18, 18);
  ctx.fillStyle = "#475467";
  ctx.fillText("Current sample", width - 228, 63);
}

function drawWaveChart(samples = []) {
  const canvas = elements.waveChart;
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  const padding = 38;
  const mid = height / 2;

  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, width, height);
  ctx.strokeStyle = "#e6eaf0";
  ctx.lineWidth = 1;

  for (let i = 0; i <= 4; i += 1) {
    const y = padding + ((height - padding * 2) / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding, y);
    ctx.lineTo(width - padding, y);
    ctx.stroke();
  }

  ctx.font = "24px Segoe UI";
  ctx.fillStyle = "#172033";
  ctx.fillText("Recorded voice waveform", padding, 28);

  if (!samples.length) {
    ctx.font = "20px Segoe UI";
    ctx.fillStyle = "#98a2b3";
    ctx.fillText("Voice waveform appears after recording.", padding, mid);
    return;
  }

  const step = Math.max(1, Math.floor(samples.length / (width - padding * 2)));
  ctx.beginPath();
  ctx.strokeStyle = "#2952cc";
  ctx.lineWidth = 2;

  for (let x = padding; x < width - padding; x += 1) {
    const sample = samples[(x - padding) * step] || 0;
    const y = mid - sample * (height * 0.34);
    if (x === padding) {
      ctx.moveTo(x, y);
    } else {
      ctx.lineTo(x, y);
    }
  }

  ctx.stroke();
}

function renderResult(values, result) {
  const riskClass = `risk-${result.risk.toLowerCase()}`;
  elements.resultCard.className = `metric-card ${riskClass}`;
  elements.riskCard.className = `metric-card ${riskClass}`;
  elements.resultText.textContent = result.summary;
  elements.riskText.textContent = result.risk;
  elements.confidenceText.textContent = `${result.confidence.toFixed(1)}%`;
  elements.ndiText.textContent = result.ndi.toFixed(4);
  elements.notesText.textContent =
    `NDI category: ${result.ndiCondition}. Browser-side screening score: ` +
    `${result.weightedScore.toFixed(2)}. Exact AI model inference requires the Python backend.`;
  drawFeatureChart(values);
  window.location.hash = "results";
}

function runManual(event) {
  event.preventDefault();
  const values = [
    Number(elements.jitter.value),
    Number(elements.shimmer.value),
    Number(elements.ppe.value),
  ];
  renderResult(values, evaluateSample(...values));
}

async function recordAudio(event) {
  event.preventDefault();
  const duration = Number(elements.duration.value);

  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    elements.recordingStatus.textContent = "Microphone capture is not supported by this browser.";
    return;
  }

  try {
    elements.recordingStatus.textContent = "Requesting microphone access...";
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    const audioContext = new AudioContext();
    const source = audioContext.createMediaStreamSource(stream);
    const processor = audioContext.createScriptProcessor(4096, 1, 1);
    const chunks = [];

    processor.onaudioprocess = (audioEvent) => {
      chunks.push(new Float32Array(audioEvent.inputBuffer.getChannelData(0)));
    };

    source.connect(processor);
    processor.connect(audioContext.destination);
    elements.recordingStatus.textContent = `Recording ${duration} seconds...`;

    await new Promise((resolve) => setTimeout(resolve, duration * 1000));

    processor.disconnect();
    source.disconnect();
    stream.getTracks().forEach((track) => track.stop());
    await audioContext.close();

    const sampleCount = chunks.reduce((sum, chunk) => sum + chunk.length, 0);
    const samples = new Float32Array(sampleCount);
    let offset = 0;
    chunks.forEach((chunk) => {
      samples.set(chunk, offset);
      offset += chunk.length;
    });

    const values = extractVoiceFeatures(samples);
    elements.jitter.value = values[0].toFixed(5);
    elements.shimmer.value = values[1].toFixed(5);
    elements.ppe.value = values[2].toFixed(5);
    elements.recordingStatus.textContent = "Recording analyzed.";

    renderResult(values, evaluateSample(...values));
    drawWaveChart(samples);
  } catch (error) {
    elements.recordingStatus.textContent = "Microphone permission was denied or recording failed.";
  }
}

function extractVoiceFeatures(samples) {
  if (!samples.length) return [0, 0, 0];

  let diffSum = 0;
  let energySum = 0;
  let flatnessSum = 0;
  const frameSize = 512;
  let frames = 0;

  for (let i = 1; i < samples.length; i += 1) {
    diffSum += Math.abs(samples[i] - samples[i - 1]);
    energySum += samples[i] * samples[i];
  }

  for (let i = 0; i + frameSize < samples.length; i += frameSize) {
    let arithmetic = 0;
    let geometric = 0;
    for (let j = 0; j < frameSize; j += 1) {
      const magnitude = Math.abs(samples[i + j]) + 0.000001;
      arithmetic += magnitude;
      geometric += Math.log(magnitude);
    }
    arithmetic /= frameSize;
    geometric = Math.exp(geometric / frameSize);
    flatnessSum += geometric / arithmetic;
    frames += 1;
  }

  const jitter = diffSum / Math.max(1, samples.length - 1);
  const shimmer = Math.sqrt(energySum / samples.length);
  const ppe = flatnessSum / Math.max(1, frames);

  return [
    Math.min(0.1, jitter),
    Math.min(0.2, shimmer),
    Math.min(0.6, ppe),
  ];
}

elements.manualTab.addEventListener("click", () => switchMode("manual"));
elements.voiceTab.addEventListener("click", () => switchMode("voice"));
elements.manualMode.addEventListener("submit", runManual);
elements.voiceMode.addEventListener("submit", recordAudio);
elements.printReport.addEventListener("click", () => window.print());

setToday();
drawFeatureChart([Number(elements.jitter.value), Number(elements.shimmer.value), Number(elements.ppe.value)]);
drawWaveChart();
