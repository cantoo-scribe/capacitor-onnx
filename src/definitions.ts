export type PluginErrorCode =
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

export interface WebConfig {
  wasmPath?: string;
}

export interface LoadModelInput {
  modelId: string;
  version: string;
  filePath?: string;
  modelBuffer?: Uint8Array;
  warmupInputs?: Record<string, RawTensor>;
  sessionOptions?: SessionOptionsInput;
}

export interface LoadModelResult {
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
  inputs: Record<string, RawTensor>;
}

export interface RawTensor {
  data: number[];
  dims?: readonly number[];
  type: "float32" | "float16" | "int32" | "int64" | "uint32" | "uint8" | "bool";
}

export interface RunResult {
  outputs: Record<string, RawTensor>;
  latencyMs: number;
}

export interface ReleaseModelInput {
  modelId: string;
  version: string;
}

export interface CapacitorOnnxPlugin {
  loadModel(input: LoadModelInput): Promise<LoadModelResult>;
  run(input: RunInput): Promise<RunResult>;
  release(input: ReleaseModelInput): Promise<void>;
}
