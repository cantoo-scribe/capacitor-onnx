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
  /** Web-only runtime tuning. Ignored on Android/iOS. */
  web?: WebSessionOptions;
}

export interface WebSessionOptions {
  /**
   * When `true` (default), the WASM backend uses multiple threads
   * (current auto-threading behavior). When `false`, runs single-threaded
   * with SIMD enabled (`ort.env.wasm.numThreads = 1`, `ort.env.wasm.simd = true`).
   */
  multithread?: boolean;
  /**
   * When `true`, runs the WASM backend inside a proxy Web Worker
   * (`ort.env.wasm.proxy = true`) so inference does not block the main
   * thread. Input/output tensors are copied to/from the worker on every
   * call, so prefer `false` (default) for small models with frequent calls.
   * Requires the ORT worker artifacts to be loadable — set `wasmPath` if
   * they are not served alongside the page.
   */
  proxy?: boolean;
  /**
   * Base path/URL where the ONNX Runtime `.wasm` artifacts are served from.
   * Applied to `ort.env.wasm.wasmPaths` before the session is created.
   * Defaults to "" (resolved relative to the page).
   */
  wasmPath?: string;
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
