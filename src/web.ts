import { WebPlugin } from "@capacitor/core";
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
} from "./definitions";
import { CapacitorOnnxError } from "./errors";
import { elapsedMs, nowMs } from "./helpers/time";
import { createSessionWithFallback } from "./web-provider-resolver";
import { applyRuntimeThreads, applyWebRuntimeConfig } from "./web-runtime-config";

export class CapacitorOnnxWeb extends WebPlugin implements CapacitorOnnxPlugin {
  private sessions = new Map<string, ort.InferenceSession>();
  private knownModelKeys = new Set<string>();
  private static cacheStorage: CacheStorage;

  private static modelKey(modelId: string, version: string): string {
    return `model-${modelId}-${version}`;
  }

  static setWebConfig(config?: WebConfig): void {
    applyWebRuntimeConfig(config?.wasmPath);

    CapacitorOnnxWeb.cacheStorage = config?.cacheStorage ?? {
      read: async () => null,
      write: async () => {},
      delete: async () => {},
    };
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

    if (_input.warmup) {
      const warmupStartTime = nowMs();
      await CapacitorOnnxWeb.warmupSession(session);
      warmupLatencyMs = elapsedMs(warmupStartTime);
      warmed = true;
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

  async clearModel(_input: ClearModelInput): Promise<ClearModelResult> {
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const session = this.sessions.get(modelKey);

    if (session) {
      await session.release().catch(() => {});
      this.sessions.delete(modelKey);
    }

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

  private static async warmupSession(session: ort.InferenceSession): Promise<void> {
    if (!session.inputNames.length) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model has no inputs to warm up.");
    }

    const inputMetadata = (
      session as unknown as {
        inputMetadata?: Record<
          string,
          { dimensions?: readonly (number | string | null)[]; type?: string }
        >;
      }
    ).inputMetadata;

    const feeds: Record<string, ort.Tensor> = {};

    for (const inputName of session.inputNames) {
      const metadata = inputMetadata?.[inputName];
      const dims = metadata?.dimensions?.length
        ? metadata.dimensions.map((dim) =>
            typeof dim === "number" && Number.isFinite(dim) && dim > 0 ? dim : 1,
          )
        : [1];
      const type = metadata?.type ?? "float32";
      const elementCount = dims.reduce((acc, dim) => acc * dim, 1);

      switch (type) {
        case "float64":
          feeds[inputName] = new ort.Tensor(
            "float64",
            new Float64Array(elementCount),
            dims,
          );
          break;
        case "int8":
          feeds[inputName] = new ort.Tensor("int8", new Int8Array(elementCount), dims);
          break;
        case "uint8":
        case "bool":
          feeds[inputName] = new ort.Tensor("uint8", new Uint8Array(elementCount), dims);
          break;
        case "int16":
          feeds[inputName] = new ort.Tensor("int16", new Int16Array(elementCount), dims);
          break;
        case "uint16":
          feeds[inputName] = new ort.Tensor("uint16", new Uint16Array(elementCount), dims);
          break;
        case "int32":
          feeds[inputName] = new ort.Tensor("int32", new Int32Array(elementCount), dims);
          break;
        case "uint32":
          feeds[inputName] = new ort.Tensor("uint32", new Uint32Array(elementCount), dims);
          break;
        case "int64":
          feeds[inputName] = new ort.Tensor(
            "int64",
            Array.from({ length: elementCount }, () => 0n),
            dims,
          );
          break;
        case "uint64":
          feeds[inputName] = new ort.Tensor(
            "uint64",
            Array.from({ length: elementCount }, () => 0n),
            dims,
          );
          break;
        default:
          feeds[inputName] = new ort.Tensor(
            "float32",
            new Float32Array(elementCount),
            dims,
          );
          break;
      }
    }

    await session.run(feeds);
  }
}
