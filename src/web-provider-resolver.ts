import * as ort from "onnxruntime-web";
import type { SessionOptionsInput } from "./definitions";
import { CapacitorOnnxError } from "./errors";

export type WebExecutionProvider = "wasm" | "webgpu" | "webnn";

export async function createSessionWithFallback(
  model: ArrayBuffer,
  requestedProvider: SessionOptionsInput["executionProvider"] | undefined,
): Promise<{ session: ort.InferenceSession; providerUsed: WebExecutionProvider }> {
  const candidates = resolveProviderCandidates(requestedProvider);
  let lastError: unknown;

  for (const provider of candidates) {
    try {
      const session = await ort.InferenceSession.create(model, {
        executionProviders: [provider],
        graphOptimizationLevel: "all",
      });

      return { session, providerUsed: provider };
    } catch (error) {
      lastError = error;
    }
  }

  const reason =
    lastError instanceof Error && lastError.message
      ? ` Last error: ${lastError.message}`
      : "";

  throw new CapacitorOnnxError(
    "SESSION_INIT_ERROR",
    `Failed to initialize ONNX Runtime session with available providers.${reason}`,
  );
}

function resolveProviderCandidates(
  requested: SessionOptionsInput["executionProvider"] | undefined,
): WebExecutionProvider[] {
  const normalized = requested ?? "auto";

  if (normalized === "wasm") {
    return ["wasm"];
  }

  if (normalized === "webgpu") {
    if (!isWebGpuSupported()) {
      throw new CapacitorOnnxError(
        "SESSION_INIT_ERROR",
        "webgpu provider was requested but is not supported in this environment.",
      );
    }

    return ["webgpu", "wasm"];
  }

  if (normalized === "webnn") {
    if (!isWebNnSupported()) {
      throw new CapacitorOnnxError(
        "SESSION_INIT_ERROR",
        "webnn provider was requested but is not supported in this environment.",
      );
    }

    return ["webnn", "wasm"];
  }

  if (normalized === "cpu" || normalized === "nnapi") {
    return ["wasm"];
  }

  const candidates: WebExecutionProvider[] = [];

  if (isWebGpuSupported()) {
    candidates.push("webgpu");
  }

  if (isWebNnSupported()) {
    candidates.push("webnn");
  }

  candidates.push("wasm");

  return candidates;
}

function isWebGpuSupported(): boolean {
  return typeof navigator !== "undefined" && "gpu" in navigator;
}

function isWebNnSupported(): boolean {
  return (
    typeof navigator !== "undefined" && "ml" in (navigator as Navigator & { ml?: unknown })
  );
}
