# Architectural Decisions

## 2026-05-04 - Initial plugin baseline
- Decision: Start with high-level JS API (`prepareModel`, `classifyImage`, cache and diagnostics methods) instead of generic tensor bridge.
- Rationale: Reduce bridge overhead, avoid large payload transfers, and keep Web API simple.
- Consequence: Lower flexibility in V1, but better performance and operational safety.

## 2026-05-04 - Session concurrency strategy
- Decision: Reuse ONNX sessions and apply per-session serialization (mutex), allowing parallelism across different model sessions.
- Rationale: Improves predictability and thread safety while retaining horizontal concurrency.
- Consequence: Single model version requests are queued; multi-model workloads can scale better.

## 2026-05-04 - Model integrity and cache promotion
- Decision: Download to temporary file, validate SHA-256, then atomically promote to final cache file.
- Rationale: Prevent corrupted or partial artifacts from being used.
- Consequence: Slightly more IO steps but robust consistency.

## 2026-05-04 - Native image inference pipeline in V1
- Decision: Implement image inference in Android (URI/path decode, resize, RGB normalization to NCHW, ONNX run), without logits decoding in the plugin.
- Rationale: Keep bridge payloads small and avoid Float32 tensor transfers between JS and Kotlin.
- Consequence: The plugin returns raw logits only (data/shape/type); labels, topK, argmax, and other business rules remain in the application.

## 2026-05-04 - Raw logits as plugin contract
- Decision: The inference contract exposed to JS returns raw logits only.
- Rationale: Separate the inference engine (plugin) from output interpretation logic (application), reducing coupling and easing model-specific evolution.
- Consequence: The consumer app must implement appropriate post-processing for each model.

## 2026-05-04 - Raw tensor input as plugin contract
- Decision: The plugin receives a preprocessed input tensor (`inputTensor`) instead of an image URI.
- Rationale: Remove domain-specific preprocessing from the plugin and keep the native layer focused on ONNX execution.
- Consequence: The client application becomes responsible for decode/resize/normalization and for guaranteeing shape/type compatibility with the model.

## 2026-05-04 - Structured error envelope for JS
- Decision: Standardize all failures exposed by the bridge with a structured envelope (`code`, `message`, `retryable`, `correlationId`, `details`).
- Rationale: Enable consistent error handling in the app, observability, and category-based retry policies.
- Consequence: Internal error messages remain useful for diagnostics, but the official consumption contract depends primarily on `code`.

## 2026-05-04 - Temporary optional SHA-256 in prepareModel
- Decision: Make `sha256` temporarily optional in `prepareModel` and skip integrity validation when it is absent.
- Rationale: Current models do not yet provide JSON metadata with reliable hashes for all artifacts.
- Consequence: Preserves short-term integration speed with lower integrity guarantees until metadata is available.

## 2026-05-04 - High-level TS helper for tensor ergonomics
- Decision: Keep the native contract based on `inputTensor` in the plugin and add a convenience helper in the TS SDK (`getInputTensor(...)`) to build tensors automatically from normalized input.
- Rationale: Preserve low coupling and keep the plugin focused on ONNX execution without losing ergonomics for usage equivalent to the current onnxruntime-web flow.
- Consequence: The consumer app can operate at a high level (normalized data) while the bridge contract remains explicit, stable, and compatible with different domains (audio, vision, etc.).

## 2026-05-04 - Android runtime optimization strategy
- Decision: Adopt flexible execution configuration (`cpu`, `nnapi`, `auto`) with safe CPU fallback, reused sessions, and per-stage telemetry (download, session creation, pre, inference, post).
- Rationale: Maximize performance without compromising stability given Android driver/provider heterogeneity.
- Consequence: The app gets hardware acceleration when feasible (NNAPI), while keeping operational predictability through controlled fallback and continuous per-device benchmarking.

## 2026-05-04 - Session options exposed in prepareModel
- Decision: Expose `sessionOptions` in `prepareModel` with `executionProvider` (`cpu`/`nnapi`/`auto`) and simple thread knobs (`intraOpNumThreads`, `interOpNumThreads`).
- Rationale: Allow performance tuning without increasing inference API complexity.
- Consequence: The session continues to be created once and reused by `modelId+version`, with automatic CPU fallback in `auto` mode.

## 2026-05-05 - Package export strategy as ESM-only
- Decision: Publish the SDK as ESM-only with `type: module` and explicit `exports` for the main entrypoint.
- Rationale: The current build is already ESM, the target ecosystem (Capacitor + Vite) is ESM-first, and this reduces operational complexity versus a dual build (CJS + ESM).
- Consequence: Consumers must use `import` (not `require`), and the package's public surface is controlled by `exports`.

## 2026-05-05 - Host/iFrame postMessage bridge in SDK
- Decision: Expose TS utilities for Host and iFrame communication (`createHostBridge` and `createIFrameBridge`) with a channel-standardized message envelope.
- Rationale: Support web integration with a minimal, typed, reusable contract without coupling to native runtime.
- Consequence: Host and iFrame apps can emit/react to events through a common API, with origin filters (`targetOrigin`/`allowedOrigins`) for basic security.

## 2026-05-05 - Async request protocol for Host/iFrame operations
- Decision: Standardize bridge operations with an asynchronous protocol using `requestId` and a `requested/result/error` lifecycle, implemented by classes (`OnnxIFrameClient` and `OnnxHostDispatcher`).
- Rationale: Ensure consistent request/response semantics for critical methods (`isActive`, `prepareModel`, `warmupModel`, `classifyImage`) and allow a Promise-based API in the iFrame.
- Consequence: The iFrame waits for results deterministically, and the Host centralizes native dispatch with stable event keys.

## 2026-05-05 - RunInference-only audio-first contract
- Decision: Remove `classifyImage` from the public API and adopt `runInference` as the only inference method in the SDK, Host/iFrame bridge, and Android plugin.
- Rationale: The product's primary domain is audio, and image-oriented naming created contract ambiguity.
- Consequence: Consumers must migrate calls to `runInference`; the contract becomes audio-semantic while remaining neutral for other tensor-based domains.
