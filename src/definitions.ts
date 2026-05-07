export type PluginErrorCode =
  | "NETWORK_ERROR"
  | "INTEGRITY_ERROR"
  | "MODEL_INTEGRITY_ERROR"
  | "MODEL_INVALID"
  | "SESSION_INIT_ERROR"
  | "INFERENCE_ERROR"
  | "TIMEOUT"
  | "CANCELED"
  | "INTERNAL_ERROR";

export interface PluginError {
  code: PluginErrorCode;
  message: string;
  retryable: boolean;
  correlationId?: string;
  details?: Record<string, unknown>;
}

export type CacheStorage = {
  read: (path: string) => Promise<ArrayBuffer | null>;
  write: (path: string, data: ArrayBuffer) => Promise<void>;
  delete: (path: string) => Promise<void>;
};

export interface WebConfig {
  cacheStorage?: CacheStorage | null;
  wasmPath?: string;
}

export interface LoadModelInput {
  modelId: string;
  version: string;
  url: string;
  sha256?: string;
  warmupInput?: RawTensor;
  timeoutMs?: number;
  forceRedownload?: boolean;
  sessionOptions?: SessionOptionsInput;
}

export interface LoadModelResult {
  status: "downloaded" | "cache_hit";
  sessionReady: boolean;
  warmed?: boolean;
  warmupLatencyMs?: number;
  latencyMs: number;
  executionProviderUsed?: "cpu" | "nnapi" | "coreml" | "wasm" | "webgpu" | "webnn";
}

export interface SessionOptionsInput {
  executionProvider?: "cpu" | "nnapi" | "coreml" | "auto" | "wasm" | "webgpu" | "webnn";
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
}

export interface RunInput {
  modelId: string;
  version: string;
  inputTensor: RawTensor;
}

export interface RawTensor {
  data: number[];
  dims?: readonly number[];
  type: "float32" | "float16" | "int32" | "int64" | "uint32" | "uint8" | "bool";
}

export interface RunResult {
  logits: RawTensor;
  latencyMs: number;
}

export interface ClearModelInput {
  modelId: string;
  version: string;
}

export interface ClearModelResult {
  removed: boolean;
}

export interface ClearAllCacheResult {
  removedModels: number;
}

export interface CapacitorOnnxPlugin {
  loadModel(input: LoadModelInput): Promise<LoadModelResult>;
  run(input: RunInput): Promise<RunResult>;
  release(input: ClearModelInput): Promise<void>;
  clearModel(input: ClearModelInput): Promise<ClearModelResult>;
  clearAllCache(): Promise<ClearAllCacheResult>;
}
