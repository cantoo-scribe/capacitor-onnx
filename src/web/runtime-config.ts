import * as ort from "onnxruntime-web";

export function applyWebRuntimeConfig(wasmPath?: string, proxy?: boolean): void {
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = wasmPath || "";
  ort.env.wasm.proxy = proxy ?? false;
}

/**
 * Single-threaded mode with SIMD enabled. Use for models small enough that
 * thread-pool overhead outweighs the parallelism gain.
 */
export function applySingleThreadRuntime(): void {
  ort.env.wasm.simd = true;
  ort.env.wasm.numThreads = 1;
}

export function applyRuntimeThreads(numThreads: number = 0): void {
  if (!numThreads) {
    const threadsAvailable =
      typeof navigator !== "undefined" ? (navigator.hardwareConcurrency ?? 1) : 1;
    const safeThreadsAvailable = Math.floor(threadsAvailable * 0.75);

    numThreads =
      safeThreadsAvailable < 2
        ? 1
        : safeThreadsAvailable <= 5
          ? Math.max(1, Math.min(safeThreadsAvailable, safeThreadsAvailable - 2))
          : safeThreadsAvailable;
  }

  ort.env.wasm.numThreads = numThreads;
}
