import type { PluginError, PluginErrorCode } from "./definitions";

function generateCorrelationId(): string {
  if (typeof globalThis.crypto !== "undefined" && "randomUUID" in globalThis.crypto) {
    return globalThis.crypto.randomUUID();
  }

  return `${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function isRetryable(code: PluginErrorCode): boolean {
  switch (code) {
    case "NETWORK_ERROR":
    case "TIMEOUT":
    case "CANCELED":
    case "SESSION_INIT_ERROR":
      return true;
    default:
      return false;
  }
}

export class CapacitorOnnxError extends Error {
  readonly code: PluginErrorCode;
  readonly retryable: boolean;
  readonly correlationId: string;
  readonly details?: Record<string, unknown>;

  constructor(
    code: PluginErrorCode,
    message: string,
    options?: {
      retryable?: boolean;
      correlationId?: string;
      details?: Record<string, unknown>;
      cause?: unknown;
    },
  ) {
    super(message);
    this.name = "CapacitorOnnxError";
    this.code = code;
    this.retryable = options?.retryable ?? isRetryable(code);
    this.correlationId = options?.correlationId ?? generateCorrelationId();
    this.details = options?.details;
    if (options?.cause !== undefined) {
      (this as Error & { cause?: unknown }).cause = options.cause;
    }
  }

  toPluginError(): PluginError {
    return {
      code: this.code,
      message: this.message,
      retryable: this.retryable,
      correlationId: this.correlationId,
      details: this.details,
    };
  }

  static fromUnknown(
    code: PluginErrorCode,
    fallbackMessage: string,
    error: unknown,
    details?: Record<string, unknown>,
  ): CapacitorOnnxError {
    if (error instanceof CapacitorOnnxError) {
      return error;
    }

    const message =
      error instanceof Error && error.message ? error.message : fallbackMessage;
    return new CapacitorOnnxError(code, message, {
      details,
      cause: error,
    });
  }
}
