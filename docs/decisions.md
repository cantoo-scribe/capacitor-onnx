# Architectural Decisions

## 2026-05-04 - Initial plugin baseline
- Decision: Start with high-level JS API (`loadModel`, `run`, cache and diagnostics methods) instead of generic tensor bridge.
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

## 2026-05-04 - Raw logits as plugin contract
- Decision: The inference contract exposed to JS returns raw logits only.
- Rationale: Separate the inference engine (plugin) from output interpretation logic (application), reducing coupling and easing model-specific evolution.
- Consequence: The consumer app must implement appropriate post-processing for each model.

## 2026-05-04 - Raw tensor input as plugin contract
- Decision: The plugin receives a preprocessed input tensor (`inputTensor`) instead of an image URI.
- Rationale: Remove domain-specific preprocessing from the plugin and keep the native layer focused on ONNX execution.
- Consequence: The client application becomes responsible for decode/resize/normalization and for guaranteeing dims/type compatibility with the model.

## 2026-05-04 - Structured error envelope for JS
- Decision: Standardize all failures exposed by the bridge with a structured envelope (`code`, `message`, `retryable`, `correlationId`, `details`).
- Rationale: Enable consistent error handling in the app, observability, and category-based retry policies.
- Consequence: Internal error messages remain useful for diagnostics, but the official consumption contract depends primarily on `code`.

## 2026-05-04 - Temporary optional SHA-256 in loadModel
- Decision: Make `sha256` temporarily optional in `loadModel` and skip integrity validation when it is absent.
- Rationale: Current models do not yet provide JSON metadata with reliable hashes for all artifacts.
- Consequence: Preserves short-term integration speed with lower integrity guarantees until metadata is available.


## 2026-05-04 - Android runtime optimization strategy
- Decision: Adopt flexible execution configuration (`cpu`, `nnapi`, `auto`) with safe CPU fallback, reused sessions, and per-stage telemetry (download, session creation, pre, inference, post).
- Rationale: Maximize performance without compromising stability given Android driver/provider heterogeneity.
- Consequence: The app gets hardware acceleration when feasible (NNAPI), while keeping operational predictability through controlled fallback and continuous per-device benchmarking.

## 2026-05-04 - Session options exposed in loadModel
- Decision: Expose `sessionOptions` in `loadModel` with `executionProvider` (`cpu`/`nnapi`/`auto`) and simple thread knobs (`intraOpNumThreads`, `interOpNumThreads`).
- Rationale: Allow performance tuning without increasing inference API complexity.
- Consequence: The session continues to be created once and reused by `modelId+version`, with automatic CPU fallback in `auto` mode.

## 2026-05-05 - Package export strategy as ESM-only
- Decision: Publish the SDK as ESM-only with `type: module` and explicit `exports` for the main entrypoint.
- Rationale: The current build is already ESM, the target ecosystem (Capacitor + Vite) is ESM-first, and this reduces operational complexity versus a dual build (CJS + ESM).
- Consequence: Consumers must use `import` (not `require`), and the package's public surface is controlled by `exports`.

## 2026-05-06 - Web provider resolution with explicit fallback
- Decision: In Web, resolve execution provider from `sessionOptions.executionProvider` with support for `auto`, `wasm`, `webgpu`, and `webnn`, mapping Android aliases (`cpu`, `nnapi`) to `wasm`.
- Rationale: Keep cross-platform API compatibility while enabling accelerated Web providers when available.
- Consequence: In `auto`, the Web runtime attempts accelerated providers first and falls back to `wasm`; `loadModel` returns `executionProviderUsed` with the effective provider.

## 2026-05-06 - Web session/cache semantics aligned by model key
- Decision: Manage Web sessions by `modelId+version` and enforce strict `loadModel.status` semantics (`cache_hit` only for valid cache reuse, `downloaded` for network fetch).
- Rationale: Align Web behavior with Android expectations and avoid ambiguous cache telemetry.
- Consequence: `run`, `clearModel`, and cache operations are keyed per model version, reducing coupling to a single active session.

## 2026-05-06 - iOS native implementation
- Decision: Mirror the Android architecture in Swift: ModelStore (download/cache/SHA-256), SessionManager (ORTSession lifecycle, CoreML EP), InferenceService (per-session NSLock, run/warmup).
- Rationale: Consistent contract across platforms; same error codes, same session key scheme (`modelId::version`), same atomic cache promotion.
- Consequence: `executionProvider` mapping on iOS: `cpu`→CPU, `nnapi`/`coreml`→CoreML, `auto`→CoreML with CPU fallback, web providers→CPU. `coreml` added to TS types.

## 2026-05-06 - iOS distributed via SPM only (no CocoaPods podspec)
- Decision: Remove the CocoaPods podspec and distribute the iOS plugin exclusively via Swift Package Manager (`Package.swift`).
- Rationale: `onnxruntime` is not published on CocoaPods; a podspec without the dependency would not compile. SPM is the supported distribution channel for `onnxruntime-swift-package-manager`.
- Consequence: iOS consumers must add the plugin as an SPM package in Xcode. CocoaPods-based Capacitor apps cannot use the iOS native implementation.

## 2026-05-06 - Web runtime split by concern
- Decision: Extract Web runtime setup and provider resolution into dedicated modules: [src/web-runtime-config.ts](../src/web-runtime-config.ts) and [src/web-provider-resolver.ts](../src/web-provider-resolver.ts).
- Rationale: Keep plugin orchestration focused and make provider/runtime logic easier to test and evolve independently.
- Consequence: `src/web.ts` now orchestrates model lifecycle while platform/runtime concerns are centralized in dedicated helpers.

