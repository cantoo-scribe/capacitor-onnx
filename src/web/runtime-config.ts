import * as ort from "onnxruntime-web";

export function applyWebRuntimeConfig(wasmPath?: string): void {
  ort.env.logLevel = "error";
  ort.env.wasm.wasmPaths = wasmPath || "";
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
