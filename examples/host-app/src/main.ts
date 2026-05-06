import type { LoadModelResult, RunResult } from "@cantoo/capacitor-onnx";
import { CapacitorOnnx } from "@cantoo/capacitor-onnx";
import { Capacitor } from "@capacitor/core";

import "./style.css";

type AssertionResult = {
  name: string;
  ok: boolean;
  details?: unknown;
};

type InferenceSuccessConfig = {
  modelId: string;
  version: string;
  url: string;
  sha256?: string;
};

type RunInferenceConfig = {
  modelId: string;
  version: string;
  normalizedData: number[];
};

type InferencePreset = {
  id: string;
  label: string;
  modelId: string;
  version: string;
  tensorData: number[];
};

type NormalizedPluginError = {
  message: string;
  code?: string;
  retryable?: boolean;
  correlationId?: string;
  details?: unknown;
  raw: unknown;
};

const loadModelInFlight = new Map<string, Promise<LoadModelResult>>();
let isModelLoading = false;

function makeRampData(length: number): number[] {
  return Array.from({ length }, (_, index) =>
    Number((index / Math.max(1, length - 1)).toFixed(6)),
  );
}

const inferencePresets: InferencePreset[] = [
  {
    id: "audio-silence-short",
    label: "Audio short silence (zeros)",
    modelId: "demo-model",
    version: "1.0.0",
    tensorData: [0, 0, 0, 0],
  },
  {
    id: "audio-short-ramp",
    label: "Audio short ramp",
    modelId: "demo-model",
    version: "1.0.0",
    tensorData: [0.1, 0.2, 0.3, 0.4],
  },
  {
    id: "audio-short-pulse",
    label: "Audio short pulse",
    modelId: "demo-model",
    version: "1.0.0",
    tensorData: [1, 0, 1, 0],
  },
  {
    id: "audio-long-ramp",
    label: "Audio long ramp (192 samples)",
    modelId: "demo-model",
    version: "1.0.0",
    tensorData: makeRampData(192),
  },
];

const app = document.querySelector<HTMLDivElement>("#app");
if (!app) {
  throw new Error("Failed to find app root element");
}

app.innerHTML = `
  <main class="container">
    <h1>Capacitor ONNX Host</h1>
    <p class="subtitle">E2E smoke runner for inference contract (success + structured error).</p>
    <section class="config">
      <h2>Inference Config</h2>
      <div class="grid">
        <div class="preset-row wide">
          <label>
            Quick preset
            <select id="input-preset">
              ${inferencePresets
                .map((preset) => `<option value="${preset.id}">${preset.label}</option>`)
                .join("")}
            </select>
          </label>
          <button id="btn-apply-preset" type="button">Apply preset</button>
        </div>
        <label>
          Model ID
          <input id="input-model-id" value="demo-model" />
        </label>
        <label>
          Version
          <input id="input-version" value="1.0.0" />
        </label>
        <label class="wide">
          Model URL
          <input id="input-url" placeholder="https://.../model.onnx" />
        </label>
        <label class="wide">
          Model SHA-256
          <input id="input-sha256" placeholder="64 hex chars" />
        </label>
        <label class="wide">
          Normalized input (CSV)
          <textarea id="input-normalized-data" rows="2">0,0,0,0</textarea>
        </label>
        <div class="mock-row wide">
          <label>
            Sample rate
            <input id="input-mock-sample-rate" type="number" min="8000" step="1000" value="16000" />
          </label>
          <label>
            Duration (ms)
            <input id="input-mock-duration-ms" type="number" min="50" step="50" value="1000" />
          </label>
          <label>
            Frequency (Hz)
            <input id="input-mock-frequency-hz" type="number" min="40" step="10" value="440" />
          </label>
          <button id="btn-generate-mock-audio" type="button">Generate mock audio</button>
        </div>
      </div>
    </section>
    <section class="actions">
      <button id="btn-load-model">Load model</button>
      <button id="btn-success-e2e">Run inference</button>
      <button id="btn-error-e2e">Run error E2E</button>
      <button id="btn-clear-model-cache">Clear model cache</button>
      <button id="btn-clear-all-cache">Clear all cache</button>
    </section>
    <section>
      <h2>Output</h2>
      <pre id="output">Ready.</pre>
    </section>
  </main>
`;

const output = document.querySelector<HTMLPreElement>("#output");
if (!output) {
  throw new Error("Failed to find output element");
}
const outputElement: HTMLPreElement = output;

function writeOutput(data: unknown) {
  outputElement.textContent = JSON.stringify(data, null, 2);
}

function parseCsvNumbers(raw: string): number[] {
  return raw
    .split(",")
    .map((item) => item.trim())
    .filter((item) => item.length > 0)
    .map((item) => Number(item));
}

function parseTensorData(raw: string): number[] {
  const values = parseCsvNumbers(raw);
  if (values.length === 0) {
    throw new Error("inputTensor.data is required");
  }
  if (values.some((value) => Number.isNaN(value) || !Number.isFinite(value))) {
    throw new Error("inputTensor.data must contain only finite numeric values");
  }
  return values;
}

function parsePositiveNumber(value: string, fieldName: string): number {
  const parsed = Number(value);
  if (!Number.isFinite(parsed) || parsed <= 0) {
    throw new Error(`${fieldName} must be a positive number`);
  }
  return parsed;
}

function generateMockAudioCsv(
  sampleRate: number,
  durationMs: number,
  frequencyHz: number,
): string {
  const totalSamples = Math.max(1, Math.floor((sampleRate * durationMs) / 1000));
  const twoPiF = 2 * Math.PI * frequencyHz;
  const values = new Array<string>(totalSamples);

  for (let i = 0; i < totalSamples; i += 1) {
    const t = i / sampleRate;
    const base = Math.sin(twoPiF * t) * 0.45;
    const harmonic = Math.sin(twoPiF * 2 * t) * 0.15;
    const envelope = 0.6 + 0.4 * Math.sin((Math.PI * i) / Math.max(1, totalSamples - 1));
    const sample = Math.max(-1, Math.min(1, (base + harmonic) * envelope));
    values[i] = sample.toFixed(6);
  }

  return values.join(",");
}

function validateNormalizedAudioLength(values: number[]) {
  // Guardrail for manual smoke tests: very short vectors usually indicate placeholder input.
  if (values.length < 64) {
    throw new Error(
      "normalizedData is too short for audio inference. Provide real normalized audio samples (e.g., hundreds/thousands of values), not a tiny placeholder like 0,0,0,0.",
    );
  }
}

function product(values: readonly number[]): number {
  return values.reduce((acc, current) => acc * current, 1);
}

function normalizePluginError(error: unknown): NormalizedPluginError {
  const asRecord = (value: unknown): Record<string, unknown> | undefined => {
    if (typeof value === "object" && value !== null) {
      return value as Record<string, unknown>;
    }
    return undefined;
  };

  const root = asRecord(error);
  const nestedData = asRecord(root?.data);
  const message =
    (typeof root?.message === "string" && root.message) ||
    (typeof nestedData?.message === "string" && nestedData.message) ||
    "Unknown plugin error";

  const code =
    (typeof root?.code === "string" && root.code) ||
    (typeof nestedData?.code === "string" && nestedData.code) ||
    undefined;

  const retryable =
    (typeof root?.retryable === "boolean" && root.retryable) ||
    (typeof nestedData?.retryable === "boolean" && nestedData.retryable) ||
    undefined;

  const correlationId =
    (typeof root?.correlationId === "string" && root.correlationId) ||
    (typeof nestedData?.correlationId === "string" && nestedData.correlationId) ||
    undefined;

  const details = nestedData?.details ?? root?.details;

  return {
    message,
    code,
    retryable,
    correlationId,
    details,
    raw: error,
  };
}

function assertion(name: string, ok: boolean, details?: unknown): AssertionResult {
  return { name, ok, details };
}

function getFormFields() {
  const modelIdInput = document.querySelector<HTMLInputElement>("#input-model-id");
  const versionInput = document.querySelector<HTMLInputElement>("#input-version");
  const urlInput = document.querySelector<HTMLInputElement>("#input-url");
  const shaInput = document.querySelector<HTMLInputElement>("#input-sha256");
  const normalizedInput = document.querySelector<HTMLTextAreaElement>(
    "#input-normalized-data",
  );
  const mockSampleRateInput = document.querySelector<HTMLInputElement>(
    "#input-mock-sample-rate",
  );
  const mockDurationInput = document.querySelector<HTMLInputElement>(
    "#input-mock-duration-ms",
  );
  const mockFrequencyInput = document.querySelector<HTMLInputElement>(
    "#input-mock-frequency-hz",
  );
  const presetSelect = document.querySelector<HTMLSelectElement>("#input-preset");
  const applyPresetButton = document.querySelector<HTMLButtonElement>("#btn-apply-preset");
  const generateMockAudioButton = document.querySelector<HTMLButtonElement>(
    "#btn-generate-mock-audio",
  );

  if (
    !modelIdInput ||
    !versionInput ||
    !urlInput ||
    !shaInput ||
    !normalizedInput ||
    !mockSampleRateInput ||
    !mockDurationInput ||
    !mockFrequencyInput ||
    !presetSelect ||
    !applyPresetButton ||
    !generateMockAudioButton
  ) {
    throw new Error("Failed to resolve one or more config fields");
  }

  return {
    modelIdInput,
    versionInput,
    urlInput,
    shaInput,
    normalizedInput,
    mockSampleRateInput,
    mockDurationInput,
    mockFrequencyInput,
    presetSelect,
    applyPresetButton,
    generateMockAudioButton,
  };
}

function applyPreset(presetId: string) {
  const preset = inferencePresets.find((item) => item.id === presetId);
  if (!preset) {
    throw new Error(`Unknown preset: ${presetId}`);
  }

  const fields = getFormFields();
  fields.modelIdInput.value = preset.modelId;
  fields.versionInput.value = preset.version;
  fields.normalizedInput.value = preset.tensorData.join(",");

  writeOutput({
    operation: "apply-preset",
    preset: preset.label,
    note: "Preset applied. Fill model URL and (optionally) SHA-256, then click Load model and Run inference.",
  });
}

function getLoadConfigFromForm(): InferenceSuccessConfig {
  const { modelIdInput, versionInput, urlInput, shaInput } = getFormFields();

  const modelId = modelIdInput.value.trim();
  const version = versionInput.value.trim();
  const url = urlInput.value.trim();
  const sha256Raw = shaInput.value.trim();
  const sha256 = sha256Raw.length > 0 ? sha256Raw : undefined;

  if (!modelId || !version || !url) {
    throw new Error("modelId, version and url are required to load model");
  }

  return {
    modelId,
    version,
    url,
    sha256,
  };
}

function getRunConfigFromForm(): RunInferenceConfig {
  const { modelIdInput, versionInput, normalizedInput } = getFormFields();

  const modelId = modelIdInput.value.trim();
  const version = versionInput.value.trim();
  const normalizedData = parseTensorData(normalizedInput.value);
  validateNormalizedAudioLength(normalizedData);

  if (!modelId || !version) {
    throw new Error("modelId and version are required to run inference");
  }

  return {
    modelId,
    version,
    normalizedData,
  };
}

function getModelLoadKey(config: InferenceSuccessConfig): string {
  return `${config.modelId}::${config.version}::${config.url}::${config.sha256 ?? ""}`;
}

const loadModelButton = document.querySelector<HTMLButtonElement>("#btn-load-model");
const successE2EButton = document.querySelector<HTMLButtonElement>("#btn-success-e2e");
const errorE2EButton = document.querySelector<HTMLButtonElement>("#btn-error-e2e");
const clearModelCacheButton = document.querySelector<HTMLButtonElement>(
  "#btn-clear-model-cache",
);
const clearAllCacheButton =
  document.querySelector<HTMLButtonElement>("#btn-clear-all-cache");
const presetSelect = document.querySelector<HTMLSelectElement>("#input-preset");
const applyPresetButton = document.querySelector<HTMLButtonElement>("#btn-apply-preset");
const generateMockAudioButton = document.querySelector<HTMLButtonElement>(
  "#btn-generate-mock-audio",
);

if (
  !loadModelButton ||
  !successE2EButton ||
  !errorE2EButton ||
  !clearModelCacheButton ||
  !clearAllCacheButton ||
  !presetSelect ||
  !applyPresetButton ||
  !generateMockAudioButton
) {
  throw new Error("Failed to find one or more action buttons");
}

const actionButtons = [
  loadModelButton,
  successE2EButton,
  errorE2EButton,
  clearModelCacheButton,
  clearAllCacheButton,
  applyPresetButton,
  generateMockAudioButton,
];

function setActionButtonsDisabled(disabled: boolean) {
  actionButtons.forEach((button) => {
    button.disabled = disabled;
  });
}

function setModelLoading(loading: boolean) {
  isModelLoading = loading;
  setActionButtonsDisabled(loading);
}

async function ensureModelPrepared(
  config: InferenceSuccessConfig,
): Promise<LoadModelResult> {
  const key = getModelLoadKey(config);
  const existing = loadModelInFlight.get(key);
  if (existing) {
    return existing;
  }

  setModelLoading(true);
  const preparePromise = CapacitorOnnx.loadModel({
    modelId: config.modelId,
    version: config.version,
    url: config.url,
    forceRedownload: true,
    ...(config.sha256 ? { sha256: config.sha256 } : {}),
  });

  loadModelInFlight.set(key, preparePromise);
  try {
    return await preparePromise;
  } finally {
    loadModelInFlight.delete(key);
    setModelLoading(false);
  }
}

applyPresetButton.addEventListener("click", () => {
  try {
    applyPreset(presetSelect.value);
  } catch (error) {
    writeOutput({
      operation: "apply-preset",
      error: normalizePluginError(error),
    });
  }
});

generateMockAudioButton.addEventListener("click", () => {
  try {
    const { normalizedInput, mockSampleRateInput, mockDurationInput, mockFrequencyInput } =
      getFormFields();
    const sampleRate = parsePositiveNumber(mockSampleRateInput.value, "sample rate");
    const durationMs = parsePositiveNumber(mockDurationInput.value, "duration");
    const frequencyHz = parsePositiveNumber(mockFrequencyInput.value, "frequency");
    const csv = generateMockAudioCsv(sampleRate, durationMs, frequencyHz);
    normalizedInput.value = csv;

    writeOutput({
      operation: "generate-mock-audio",
      sampleRate,
      durationMs,
      frequencyHz,
      sampleCount: Math.floor((sampleRate * durationMs) / 1000),
      note: "Mock audio generated and applied to normalized input field.",
    });
  } catch (error) {
    writeOutput({
      operation: "generate-mock-audio",
      error: normalizePluginError(error),
    });
  }
});

loadModelButton.addEventListener("click", async () => {
  const startedAt = Date.now();
  try {
    if (isModelLoading) {
      return;
    }

    const config = getLoadConfigFromForm();
    const loadModel = await ensureModelPrepared(config);

    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "load-model",
      passed: loadModel.sessionReady === true,
      durationMs: Date.now() - startedAt,
      loadModel,
    });
  } catch (error) {
    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "load-model",
      passed: false,
      durationMs: Date.now() - startedAt,
      error: normalizePluginError(error),
    });
  }
});

successE2EButton.addEventListener("click", async () => {
  const startedAt = Date.now();
  try {
    if (isModelLoading) {
      return;
    }

    const config = getRunConfigFromForm();

    const inference: RunResult = await CapacitorOnnx.run({
      modelId: config.modelId,
      version: config.version,
      inputTensor: {
        data: config.normalizedData,
        dims: [1, config.normalizedData.length],
        type: "float32",
      },
    });

    const assertions: AssertionResult[] = [
      assertion(
        "runInference.logits.type",
        inference.logits.type === "float32",
        inference.logits.type,
      ),
      assertion(
        "runInference.latencyMs is numeric",
        Number.isFinite(inference.latencyMs),
        inference.latencyMs,
      ),
    ];

    const allPassed = assertions.every((item) => item.ok);

    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "success-e2e",
      passed: allPassed,
      durationMs: Date.now() - startedAt,
      assertions,
      inferenceSummary: {
        logitsType: inference.logits.type,
        logitsDims: inference.logits.dims,
        logitsLength: inference.logits.data.length,
      },
    });
  } catch (error) {
    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "success-e2e",
      passed: false,
      durationMs: Date.now() - startedAt,
      error: normalizePluginError(error),
    });
  }
});

clearModelCacheButton.addEventListener("click", async () => {
  if (isModelLoading) {
    return;
  }

  try {
    const { modelIdInput, versionInput } = getFormFields();
    const modelId = modelIdInput.value.trim();
    const version = versionInput.value.trim();

    if (!modelId || !version) {
      throw new Error("modelId and version are required to clear model cache");
    }

    const result = await CapacitorOnnx.clearModel({
      modelId,
      version,
    });

    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "clear-model-cache",
      result,
    });
  } catch (error) {
    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "clear-model-cache",
      error: normalizePluginError(error),
    });
  }
});

clearAllCacheButton.addEventListener("click", async () => {
  if (isModelLoading) {
    return;
  }

  try {
    const result = await CapacitorOnnx.clearAllCache();
    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "clear-all-cache",
      result,
    });
  } catch (error) {
    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "clear-all-cache",
      error: normalizePluginError(error),
    });
  }
});

errorE2EButton.addEventListener("click", async () => {
  const startedAt = Date.now();
  try {
    // Intentionally invalid model/version to force plugin-side structured error contract.
    await CapacitorOnnx.run({
      modelId: "missing-model",
      version: "0.0.0",
      inputTensor: {
        data: [0, 0, 0, 0],
        dims: [1, 4],
        type: "float32",
      },
    });

    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "error-e2e",
      passed: false,
      durationMs: Date.now() - startedAt,
      reason: "Expected runInference to fail but it succeeded",
    });
  } catch (error) {
    const normalized = normalizePluginError(error);
    const assertions: AssertionResult[] = [
      assertion(
        "error has code",
        typeof normalized.code === "string" && normalized.code.length > 0,
        normalized.code,
      ),
      assertion("error has message", normalized.message.length > 0, normalized.message),
      assertion(
        "error has retryable boolean",
        typeof normalized.retryable === "boolean",
        normalized.retryable,
      ),
      assertion(
        "error has correlationId",
        typeof normalized.correlationId === "string" && normalized.correlationId.length > 0,
        normalized.correlationId,
      ),
    ];

    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "error-e2e",
      passed: assertions.every((item) => item.ok),
      durationMs: Date.now() - startedAt,
      assertions,
      error: normalized,
    });
  }
});
