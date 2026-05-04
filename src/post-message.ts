import type {
  ClearAllCacheResult,
  ClearModelInput,
  ClearModelResult,
  DiagnosticsResult,
  GetModelStatusInput,
  GetModelStatusResult,
  PrepareModelInput,
  PrepareModelResult,
  RawTensor,
  RunInferenceInput,
  RunInferenceResult,
  WarmupModelInput,
  WarmupModelResult,
} from './definitions';
import { getInputTensor, GetInputTensorOptions } from './runtime';

export const DEFAULT_BRIDGE_CHANNEL = '@cantoo/capacitor-onnx';
export const ONNX_BRIDGE_EVENT_TYPE = 'cantoo:onnx:bridge:event';

export interface BridgeMessage<TPayload = unknown> {
  channel: string;
  type: string;
  payload?: TPayload;
}

export type BridgeMessageHandler<TPayload = unknown> = (
  message: BridgeMessage<TPayload>,
  event: MessageEvent,
) => void;

export interface HostBridgeOptions {
  iframeWindow: Window;
  targetOrigin: string;
  allowedOrigins?: string[];
  channel?: string;
}

export interface IFrameBridgeOptions {
  parentWindow?: Window;
  targetOrigin: string;
  allowedOrigins?: string[];
  channel?: string;
}

export interface PostMessageBridge {
  emit<TPayload = unknown>(type: string, payload?: TPayload): void;
  onMessage<TPayload = unknown>(handler: BridgeMessageHandler<TPayload>): () => void;
}

export interface OnnxBridgeEvent<TData = unknown> {
  key: string;
  requestId?: string;
  data?: TData;
  err?: OnnxBridgeSerializedError;
}

export interface OnnxBridgeSerializedError {
  message: string;
  name?: string;
  stack?: string;
  details?: unknown;
}

export const ONNX_BRIDGE_KEYS = {
  request: {
    isActive: 'plugin:active',
    prepareModel: 'INITIALIZE_MODEL',
    warmupModel: 'WARMUP_MODEL',
    runInference: 'RUN_INFERENCE',
    getModelStatus: 'GET_MODEL_STATUS',
    clearModel: 'CLEAR_MODEL',
    clearAllCache: 'CLEAR_ALL_CACHE',
    getDiagnostics: 'GET_DIAGNOSTICS',
  },
  status: {
    isActiveRequested: 'is-active:requested',
    isActiveResult: 'is-active:result',
    isActiveError: 'is-active:error',
    prepareModelRequested: 'initialize-model:requested',
    prepareModelResult: 'initialize-model:result',
    prepareModelError: 'initialize-model:error',
    warmupModelRequested: 'warmup-model:requested',
    warmupModelResult: 'warmup-model:result',
    warmupModelError: 'warmup-model:error',
    runInferenceRequested: 'run-inference:requested',
    runInferenceResult: 'run-inference:result',
    runInferenceError: 'run-inference:error',
    getModelStatusRequested: 'get-model-status:requested',
    getModelStatusResult: 'get-model-status:result',
    getModelStatusError: 'get-model-status:error',
    clearModelRequested: 'clear-model:requested',
    clearModelResult: 'clear-model:result',
    clearModelError: 'clear-model:error',
    clearAllCacheRequested: 'clear-all-cache:requested',
    clearAllCacheResult: 'clear-all-cache:result',
    clearAllCacheError: 'clear-all-cache:error',
    getDiagnosticsRequested: 'get-diagnostics:requested',
    getDiagnosticsResult: 'get-diagnostics:result',
    getDiagnosticsError: 'get-diagnostics:error',
  },
} as const;

export interface OnnxIFrameClientOptions {
  requestTimeoutMs?: number;
}

export interface OnnxHostHandler {
  isActive(): Promise<boolean>;
  prepareModel(input: PrepareModelInput): Promise<PrepareModelResult>;
  warmupModel(input: WarmupModelInput): Promise<WarmupModelResult>;
  runInference(input: RunInferenceInput): Promise<RunInferenceResult>;
  getModelStatus(input: GetModelStatusInput): Promise<GetModelStatusResult>;
  clearModel(input: ClearModelInput): Promise<ClearModelResult>;
  clearAllCache(): Promise<ClearAllCacheResult>;
  getDiagnostics(): Promise<DiagnosticsResult>;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null;
}

function isOnnxBridgeEvent(value: unknown): value is OnnxBridgeEvent {
  return isRecord(value) && typeof value.key === 'string';
}

function escapeRegex(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
}

function matchesOriginPattern(origin: string, pattern: string): boolean {
  if (pattern === '*') {
    return true;
  }

  if (!pattern.includes('*')) {
    return origin === pattern;
  }

  const regex = new RegExp(`^${pattern.split('*').map(escapeRegex).join('.*')}$`);
  return regex.test(origin);
}

function resolvePostMessageTargetOrigin(targetOrigin: string): string {
  return targetOrigin.includes('*') ? '*' : targetOrigin;
}

function serializeError(error: unknown): OnnxBridgeSerializedError {
  if (error instanceof Error) {
    return {
      message: error.message,
      name: error.name,
      stack: error.stack,
    };
  }

  if (typeof error === 'string') {
    return { message: error };
  }

  return {
    message: 'Unknown bridge error',
    details: error,
  };
}

function isBridgeMessage(value: unknown, channel: string): value is BridgeMessage {
  if (!isRecord(value)) {
    return false;
  }

  return (
    value.channel === channel &&
    typeof value.type === 'string' &&
    (value.payload === undefined || isRecord(value.payload) || Array.isArray(value.payload) || typeof value.payload !== 'function')
  );
}

function isOriginAllowed(origin: string, targetOrigin: string, allowedOrigins?: string[]): boolean {
  if (allowedOrigins && allowedOrigins.length > 0) {
    return allowedOrigins.some((allowedOrigin) => matchesOriginPattern(origin, allowedOrigin));
  }

  return matchesOriginPattern(origin, targetOrigin);
}

function createBridge(targetWindow: Window, targetOrigin: string, options?: { channel?: string; allowedOrigins?: string[] }): PostMessageBridge {
  const channel = options?.channel ?? DEFAULT_BRIDGE_CHANNEL;

  return {
    emit(type, payload) {
      const message: BridgeMessage = {
        channel,
        type,
        payload,
      };

      targetWindow.postMessage(message, resolvePostMessageTargetOrigin(targetOrigin));
    },
    onMessage<TPayload = unknown>(handler: BridgeMessageHandler<TPayload>) {
      const listener = (event: MessageEvent) => {
        if (event.source !== targetWindow) {
          return;
        }

        if (!isOriginAllowed(event.origin, targetOrigin, options?.allowedOrigins)) {
          return;
        }

        if (!isBridgeMessage(event.data, channel)) {
          return;
        }

        handler(event.data as BridgeMessage<TPayload>, event);
      };

      window.addEventListener('message', listener);
      return () => {
        window.removeEventListener('message', listener);
      };
    },
  };
}

export function createHostBridge(options: HostBridgeOptions): PostMessageBridge {
  return createBridge(options.iframeWindow, options.targetOrigin, {
    channel: options.channel,
    allowedOrigins: options.allowedOrigins,
  });
}

export function createIFrameBridge(options: IFrameBridgeOptions): PostMessageBridge {
  const parentWindow = options.parentWindow ?? window.parent;

  if (!parentWindow) {
    throw new Error('Failed to resolve parent window for iFrame bridge');
  }

  return createBridge(parentWindow, options.targetOrigin, {
    channel: options.channel,
    allowedOrigins: options.allowedOrigins,
  });
}

interface PendingRequest {
  resolve: (value: unknown) => void;
  reject: (reason?: unknown) => void;
  resultKey: string;
  errorKey: string;
  timeoutId: ReturnType<typeof setTimeout>;
}

export class OnnxIFrameClient {
  private readonly pending = new Map<string, PendingRequest>();

  private readonly removeListener: () => void;

  private nextRequestNumber = 0;

  constructor(
    private readonly bridge: PostMessageBridge,
    private readonly options?: OnnxIFrameClientOptions,
  ) {
    this.removeListener = this.bridge.onMessage<OnnxBridgeEvent>((message) => {
      if (message.type !== ONNX_BRIDGE_EVENT_TYPE || !isOnnxBridgeEvent(message.payload)) {
        return;
      }

      const event = message.payload;
      if (!event.requestId) {
        return;
      }

      const pending = this.pending.get(event.requestId);
      if (!pending) {
        return;
      }

      if (event.key === pending.resultKey) {
        clearTimeout(pending.timeoutId);
        this.pending.delete(event.requestId);
        pending.resolve(event.data);
      }

      if (event.key === pending.errorKey) {
        clearTimeout(pending.timeoutId);
        this.pending.delete(event.requestId);
        pending.reject(event.err ?? { message: 'Unknown bridge error' });
      }
    });
  }

  destroy(): void {
    this.removeListener();

    this.pending.forEach((pending) => {
      clearTimeout(pending.timeoutId);
      pending.reject({ message: 'Bridge client destroyed' });
    });

    this.pending.clear();
  }

  async isActive(): Promise<boolean> {
    const data = await this.call<{ isActive: boolean }>({
      requestKey: ONNX_BRIDGE_KEYS.request.isActive,
      resultKey: ONNX_BRIDGE_KEYS.status.isActiveResult,
      errorKey: ONNX_BRIDGE_KEYS.status.isActiveError,
    });

    return data.isActive;
  }

  prepareModel(input: PrepareModelInput): Promise<PrepareModelResult> {
    return this.call<PrepareModelResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.prepareModel,
      resultKey: ONNX_BRIDGE_KEYS.status.prepareModelResult,
      errorKey: ONNX_BRIDGE_KEYS.status.prepareModelError,
      data: input,
    });
  }

  warmupModel(input: WarmupModelInput): Promise<WarmupModelResult> {
    return this.call<WarmupModelResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.warmupModel,
      resultKey: ONNX_BRIDGE_KEYS.status.warmupModelResult,
      errorKey: ONNX_BRIDGE_KEYS.status.warmupModelError,
      data: input,
    });
  }

  runInference(input: RunInferenceInput): Promise<RunInferenceResult> {
    return this.call<RunInferenceResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.runInference,
      resultKey: ONNX_BRIDGE_KEYS.status.runInferenceResult,
      errorKey: ONNX_BRIDGE_KEYS.status.runInferenceError,
      data: input,
    });
  }

  getModelStatus(input: GetModelStatusInput): Promise<GetModelStatusResult> {
    return this.call<GetModelStatusResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.getModelStatus,
      resultKey: ONNX_BRIDGE_KEYS.status.getModelStatusResult,
      errorKey: ONNX_BRIDGE_KEYS.status.getModelStatusError,
      data: input,
    });
  }

  clearModel(input: ClearModelInput): Promise<ClearModelResult> {
    return this.call<ClearModelResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.clearModel,
      resultKey: ONNX_BRIDGE_KEYS.status.clearModelResult,
      errorKey: ONNX_BRIDGE_KEYS.status.clearModelError,
      data: input,
    });
  }

  clearAllCache(): Promise<ClearAllCacheResult> {
    return this.call<ClearAllCacheResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.clearAllCache,
      resultKey: ONNX_BRIDGE_KEYS.status.clearAllCacheResult,
      errorKey: ONNX_BRIDGE_KEYS.status.clearAllCacheError,
    });
  }

  getDiagnostics(): Promise<DiagnosticsResult> {
    return this.call<DiagnosticsResult>({
      requestKey: ONNX_BRIDGE_KEYS.request.getDiagnostics,
      resultKey: ONNX_BRIDGE_KEYS.status.getDiagnosticsResult,
      errorKey: ONNX_BRIDGE_KEYS.status.getDiagnosticsError,
    });
  }

  getInputTensor(input: ArrayLike<number>, options?: GetInputTensorOptions): RawTensor {
    return getInputTensor(input, options);
  }

  private call<TData>(params: {
    requestKey: string;
    resultKey: string;
    errorKey: string;
    data?: unknown;
  }): Promise<TData> {
    const requestId = this.nextRequestId();
    const timeoutMs = this.options?.requestTimeoutMs ?? 15000;

    return new Promise<TData>((resolve, reject) => {
      const timeoutId = setTimeout(() => {
        this.pending.delete(requestId);
        reject({ message: `Bridge request timed out: ${params.requestKey}`, requestId });
      }, timeoutMs);

      this.pending.set(requestId, {
        resolve: resolve as (value: unknown) => void,
        reject,
        resultKey: params.resultKey,
        errorKey: params.errorKey,
        timeoutId,
      });

      const event: OnnxBridgeEvent = {
        key: params.requestKey,
        requestId,
        ...(params.data !== undefined ? { data: params.data } : {}),
      };

      this.bridge.emit(ONNX_BRIDGE_EVENT_TYPE, event);
    });
  }

  private nextRequestId(): string {
    this.nextRequestNumber += 1;
    return `req_${Date.now()}_${this.nextRequestNumber}`;
  }
}

export class OnnxHostDispatcher {
  private removeListener?: () => void;

  constructor(
    private readonly bridge: PostMessageBridge,
    private readonly handler: OnnxHostHandler,
  ) {}

  start(): () => void {
    if (this.removeListener) {
      return this.removeListener;
    }

    this.removeListener = this.bridge.onMessage<OnnxBridgeEvent>((message) => {
      if (message.type !== ONNX_BRIDGE_EVENT_TYPE || !isOnnxBridgeEvent(message.payload)) {
        return;
      }

      void this.handleEvent(message.payload);
    });

    return this.removeListener;
  }

  stop(): void {
    if (this.removeListener) {
      this.removeListener();
      this.removeListener = undefined;
    }
  }

  getInputTensor(input: ArrayLike<number>, options?: GetInputTensorOptions): RawTensor {
    return getInputTensor(input, options);
  }

  private async handleEvent(event: OnnxBridgeEvent): Promise<void> {
    switch (event.key) {
      case ONNX_BRIDGE_KEYS.request.isActive:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.isActiveRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.isActiveResult,
          errorKey: ONNX_BRIDGE_KEYS.status.isActiveError,
          run: async () => ({ isActive: await this.handler.isActive() }),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.prepareModel:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.prepareModelRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.prepareModelResult,
          errorKey: ONNX_BRIDGE_KEYS.status.prepareModelError,
          run: async () => this.handler.prepareModel(event.data as PrepareModelInput),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.warmupModel:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.warmupModelRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.warmupModelResult,
          errorKey: ONNX_BRIDGE_KEYS.status.warmupModelError,
          run: async () => this.handler.warmupModel(event.data as WarmupModelInput),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.runInference:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.runInferenceRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.runInferenceResult,
          errorKey: ONNX_BRIDGE_KEYS.status.runInferenceError,
          run: async () => this.handler.runInference(event.data as RunInferenceInput),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.getModelStatus:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.getModelStatusRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.getModelStatusResult,
          errorKey: ONNX_BRIDGE_KEYS.status.getModelStatusError,
          run: async () => this.handler.getModelStatus(event.data as GetModelStatusInput),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.clearModel:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.clearModelRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.clearModelResult,
          errorKey: ONNX_BRIDGE_KEYS.status.clearModelError,
          run: async () => this.handler.clearModel(event.data as ClearModelInput),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.clearAllCache:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.clearAllCacheRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.clearAllCacheResult,
          errorKey: ONNX_BRIDGE_KEYS.status.clearAllCacheError,
          run: async () => this.handler.clearAllCache(),
        });
        break;
      case ONNX_BRIDGE_KEYS.request.getDiagnostics:
        await this.runRequest({
          event,
          requestedKey: ONNX_BRIDGE_KEYS.status.getDiagnosticsRequested,
          resultKey: ONNX_BRIDGE_KEYS.status.getDiagnosticsResult,
          errorKey: ONNX_BRIDGE_KEYS.status.getDiagnosticsError,
          run: async () => this.handler.getDiagnostics(),
        });
        break;
      default:
        break;
    }
  }

  private async runRequest<TData>(params: {
    event: OnnxBridgeEvent;
    requestedKey: string;
    resultKey: string;
    errorKey: string;
    run: () => Promise<TData>;
  }): Promise<void> {
    this.emitEvent({
      key: params.requestedKey,
      requestId: params.event.requestId,
    });

    try {
      const result = await params.run();
      this.emitEvent({
        key: params.resultKey,
        requestId: params.event.requestId,
        data: result,
      });
    } catch (error) {
      this.emitEvent({
        key: params.errorKey,
        requestId: params.event.requestId,
        err: serializeError(error),
      });
    }
  }

  private emitEvent(event: OnnxBridgeEvent): void {
    this.bridge.emit(ONNX_BRIDGE_EVENT_TYPE, event);
  }
}