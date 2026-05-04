export type PluginErrorCode =
  | 'NETWORK_ERROR'
  | 'INTEGRITY_ERROR'
  | 'MODEL_INVALID'
  | 'SESSION_INIT_ERROR'
  | 'INFERENCE_ERROR'
  | 'TIMEOUT'
  | 'CANCELED'
  | 'INTERNAL_ERROR';

export interface PluginError {
  code: PluginErrorCode;
  message: string;
  retryable: boolean;
  correlationId?: string;
  details?: Record<string, unknown>;
}

export interface PrepareModelInput {
  modelId: string;
  version: string;
  url: string;
  sha256?: string;
  timeoutMs?: number;
  forceRedownload?: boolean;
  sessionOptions?: SessionOptionsInput;
}

export interface PrepareModelResult {
  status: 'downloaded' | 'cache_hit';
  sessionReady: boolean;
  latencyMs: number;
  executionProviderUsed?: 'cpu' | 'nnapi';
}

export interface SessionOptionsInput {
  executionProvider?: 'cpu' | 'nnapi' | 'auto';
  intraOpNumThreads?: number;
  interOpNumThreads?: number;
}

export interface WarmupModelInput {
  modelId: string;
  version: string;
}

export interface WarmupModelResult {
  warmed: boolean;
  latencyMs: number;
}

export interface RunInferenceInput {
  modelId: string;
  version: string;
  inputTensor: RawTensor;
}

export interface RawTensor {
  data: number[];
  shape: number[];
  type: 'float32';
}

export interface RunInferenceResult {
  logits: RawTensor;
  latencyMs: number;
}

export interface InferFromAudioInput {
  modelId: string;
  version: string;
  normalizedData: ArrayLike<number>;
  shape?: number[];
}

export interface GetModelStatusInput {
  modelId: string;
  version: string;
}

export interface GetModelStatusResult {
  exists: boolean;
  integrityOk: boolean;
  sessionLoaded: boolean;
  sizeBytes?: number;
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

export interface DiagnosticsResult {
  activeSessions: number;
  cacheEntries: number;
}

export interface IsActiveResult {
  value: boolean;
}

export interface CapacitorOnnxPlugin {
  isActive(): Promise<IsActiveResult>;
  prepareModel(input: PrepareModelInput): Promise<PrepareModelResult>;
  warmupModel(input: WarmupModelInput): Promise<WarmupModelResult>;
  runInference(input: RunInferenceInput): Promise<RunInferenceResult>;
  getModelStatus(input: GetModelStatusInput): Promise<GetModelStatusResult>;
  clearModel(input: ClearModelInput): Promise<ClearModelResult>;
  clearAllCache(): Promise<ClearAllCacheResult>;
  getDiagnostics(): Promise<DiagnosticsResult>;
}
