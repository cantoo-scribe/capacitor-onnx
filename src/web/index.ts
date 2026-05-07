import * as ort from "onnxruntime-web";
import type {
  CacheStorage,
  CapacitorOnnxPlugin,
  ClearAllCacheResult,
  ClearModelInput,
  ClearModelResult,
  LoadModelInput,
  LoadModelResult,
  RawTensor,
  RunInput,
  RunResult,
  SessionOptionsInput,
  WebConfig,
} from "../definitions";
import { CapacitorOnnxError } from "../errors";
import { elapsedMs, nowMs } from "../helpers/time";
import { createSessionWithFallback } from "./provider-resolver";
import { applyRuntimeThreads, applyWebRuntimeConfig } from "./runtime-config";

const DEFAULT_CACHE_STORAGE: CacheStorage = {
  read: async () => null,
  write: async () => {},
  delete: async () => {},
};

type SharedWebState = {
  cacheStorage: CacheStorage;
};

const SHARED_WEB_STATE_KEY = "__cantooCapacitorOnnxWebState__";

function getSharedWebState(): SharedWebState {
  const globalScope = globalThis as typeof globalThis & {
    [SHARED_WEB_STATE_KEY]?: SharedWebState;
  };

  if (!globalScope[SHARED_WEB_STATE_KEY]) {
    globalScope[SHARED_WEB_STATE_KEY] = {
      cacheStorage: DEFAULT_CACHE_STORAGE,
    };
  }

  return globalScope[SHARED_WEB_STATE_KEY];
}

export class CapacitorOnnxWeb implements CapacitorOnnxPlugin {
  private sessions = new Map<string, ort.InferenceSession>();
  private knownModelKeys = new Set<string>();

  private static get cacheStorage(): CacheStorage {
    return getSharedWebState().cacheStorage;
  }

  private static set cacheStorage(value: CacheStorage) {
    getSharedWebState().cacheStorage = value;
  }

  private static modelKey(modelId: string, version: string): string {
    return `model-${modelId}-${version}`;
  }

  static setWebConfig(config?: WebConfig): void {
    applyWebRuntimeConfig(config?.wasmPath);
    CapacitorOnnxWeb.cacheStorage = config?.cacheStorage ?? DEFAULT_CACHE_STORAGE;
  }

  async loadModel(_input: LoadModelInput): Promise<LoadModelResult> {
    applyRuntimeThreads(_input.sessionOptions?.intraOpNumThreads);

    const startTime = nowMs();
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    this.knownModelKeys.add(modelKey);

    let cacheHit = false;
    let loadedModel: ArrayBuffer | null = null;

    if (!_input.forceRedownload) {
      loadedModel = (await CapacitorOnnxWeb.cacheStorage.read(modelKey)) ?? null;
    }

    const executionProvider = _input.sessionOptions?.executionProvider ?? "auto";

    if (loadedModel && !_input.forceRedownload) {
      cacheHit = true;
      try {
        await CapacitorOnnxWeb.checkModelIntegrity(loadedModel, executionProvider);
      } catch {
        await CapacitorOnnxWeb.cacheStorage.delete(modelKey).catch(() => {});
        loadedModel = null;
        cacheHit = false;
      }
    }

    if (!loadedModel) {
      loadedModel = await CapacitorOnnxWeb.fetchModel(_input.url);
      await CapacitorOnnxWeb.checkModelIntegrity(loadedModel, executionProvider);
      await CapacitorOnnxWeb.cacheStorage.write(modelKey, loadedModel).catch(() => {});
    }

    await this.sessions
      .get(modelKey)
      ?.release()
      .catch(() => {});

    const created = await createSessionWithFallback(loadedModel, executionProvider);
    const session = created.session;

    this.sessions.set(modelKey, session);

    let warmed = false;
    let warmupLatencyMs: number | undefined;

    if (_input.warmupInput) {
      const warmupStartTime = nowMs();
      warmed = await CapacitorOnnxWeb.warmupSession(session, _input.warmupInput);
      warmupLatencyMs = elapsedMs(warmupStartTime);
    }

    return {
      status: cacheHit ? "cache_hit" : "downloaded",
      sessionReady: true,
      latencyMs: elapsedMs(startTime),
      warmed,
      warmupLatencyMs,
      executionProviderUsed: created.providerUsed,
    };
  }

  async run(_input: RunInput): Promise<RunResult> {
    const startTime = nowMs();
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const session = this.sessions.get(modelKey);

    const inputTensor = new ort.Tensor(
      _input.inputTensor.type,
      _input.inputTensor.data,
      _input.inputTensor.dims,
    );

    if (!session) {
      throw new CapacitorOnnxError(
        "SESSION_INIT_ERROR",
        "ONNX Runtime session is not initialized. Please load a model before running inference.",
      );
    }

    const inputName = session.inputNames[0];
    if (!inputName) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model has no inputs.");
    }

    const inferenceResult = await session.run({ [inputName]: inputTensor });
    const outputName = session.outputNames[0] ?? Object.keys(inferenceResult)[0];
    const outputTensor = outputName ? inferenceResult[outputName] : undefined;

    if (!outputTensor) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model has no outputs.");
    }

    return {
      logits: {
        type: outputTensor.type as RawTensor["type"],
        data: Array.from(outputTensor.data as ArrayLike<number | bigint>, Number),
        dims: outputTensor.dims,
      },
      latencyMs: elapsedMs(startTime),
    };
  }

  async release(_input: ClearModelInput): Promise<void> {
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const session = this.sessions.get(modelKey);
    if (session) {
      await session.release().catch(() => {});
      this.sessions.delete(modelKey);
    }
  }

  async clearModel(_input: ClearModelInput): Promise<ClearModelResult> {
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    await this.release(_input);

    this.knownModelKeys.delete(modelKey);
    await CapacitorOnnxWeb.cacheStorage.delete(modelKey).catch(() => {});

    return { removed: true };
  }

  async clearAllCache(): Promise<ClearAllCacheResult> {
    const removedModels = this.knownModelKeys.size;

    for (const session of this.sessions.values()) {
      await session.release().catch(() => {});
    }

    this.sessions.clear();

    for (const modelKey of this.knownModelKeys) {
      await CapacitorOnnxWeb.cacheStorage.delete(modelKey).catch(() => {});
    }

    this.knownModelKeys.clear();

    return {
      removedModels,
    };
  }

  private static async fetchModel(url: string): Promise<ArrayBuffer> {
    const response = await fetch(url);

    if (!response.ok) {
      throw new CapacitorOnnxError(
        "NETWORK_ERROR",
        `Failed to fetch model from URL: ${url}. Status: ${response.status} ${response.statusText}`,
      );
    }

    return await response.arrayBuffer();
  }

  private static async checkModelIntegrity(
    model: ArrayBuffer,
    executionProvider: SessionOptionsInput["executionProvider"],
  ): Promise<void> {
    const session = await createSessionWithFallback(
      model,
      executionProvider ?? "auto",
    ).then((result) => result.session);

    await session.release();
  }

  private static async warmupSession(
    session: ort.InferenceSession,
    warmupInput: RawTensor,
  ): Promise<boolean> {
    const inputName = session.inputNames[0];
    if (!inputName) {
      return false;
    }

    try {
      const tensor = new ort.Tensor(
        warmupInput.type,
        warmupInput.data,
        warmupInput.dims,
      );
      await session.run({ [inputName]: tensor });
      return true;
    } catch (err) {
      console.warn(
        "[CapacitorOnnxWeb] Warmup inference failed; continuing without warmup.",
        err,
      );
      return false;
    }
  }
}
