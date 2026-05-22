# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

```bash
pnpm build       # Compile TypeScript to dist/
pnpm typecheck   # Type-check without emitting (alias: pnpm lint)
```

There are no automated tests in this repo. Manual testing flows are documented in [docs/testing-scripts.md](docs/testing-scripts.md).

### Publishing

```bash
npm version patch   # or minor/major — also updates CantooCapacitorOnnx.podspec automatically via package.json
pnpm build
pnpm typecheck
pnpm pack --dry-run
npm publish
```

After publishing, consumer Android apps should run `pnpm add @cantoo/capacitor-onnx@latest && pnpm cap sync android`.

## Architecture

This is a Capacitor plugin that exposes ONNX Runtime inference to JavaScript apps on **Android**, **iOS**, and **Web**.

### Plugin surface (`src/`)

- **`src/definitions.ts`** — All TypeScript interfaces and types for the public API (`CapacitorOnnxPlugin`, `LoadModelInput`, `RunInput`, `RawTensor`, etc.). This is the source of truth for the contract.
- **`src/plugin.ts`** — Calls `registerPlugin("CapacitorOnnx", { web: ... })`. On Android/iOS, Capacitor routes calls to the native plugin; on web, it instantiates `CapacitorOnnxWeb`.
- **`src/errors.ts`** — `CapacitorOnnxError` with structured fields (`code`, `message`, `retryable`, `correlationId`, `details`).
- **`src/web/index.ts`** — `CapacitorOnnxWeb extends WebPlugin`: implements the full plugin interface for web. Manages sessions in a `Map<modelKey, ort.InferenceSession>`. Session keys are `model-{modelId}-{version}`. `cacheStorage` is stored on `globalThis` so it survives multiple plugin instantiations (Capacitor may recreate the class).
- **`src/web/provider-resolver.ts`** — `createSessionWithFallback`: maps the requested provider string to an ordered candidate list and tries each until one succeeds. Native-only aliases (`cpu`, `nnapi`, `coreml`) map to `wasm` on Web.
- **`src/web/runtime-config.ts`** — Configures global `ort.env` (WASM path, number of threads).

### Android (`android/src/main/java/com/cantoo/capacitor/onnx/`)

- **`CapacitorOnnxPlugin.kt`** — Capacitor `@CapacitorPlugin` entry point; bridges JS calls to `InferenceService`.
- **`InferenceService.kt`** — Orchestrates download, caching, session creation, and inference; uses coroutines.
- **`SessionManager.kt`** — Manages ONNX `InferenceSession` instances keyed by `modelId+version`; applies per-session mutex for serialized inference.
- **`ModelStore.kt`** — Handles file system cache: download to temp file, SHA-256 validate, atomic promote to cache path.

### iOS (`ios/Plugin/`)

Distributed via both **CocoaPods** (`CantooCapacitorOnnx.podspec` at the repo root, picked up by `cap sync ios`; depends on `onnxruntime-objc`) and **Swift Package Manager** (`Package.swift`, library name `CantooCapacitorOnnx`, depends on `onnxruntime-swift-package-manager`). Both channels expose the same product name `CantooCapacitorOnnx` (matching what the Capacitor CLI derives from the npm package name `@cantoo/capacitor-onnx`). The SPM target is still `CapacitorOnnxPlugin` internally — that's the Swift module name, but Capacitor discovers the plugin via the ObjC runtime (`CAP_PLUGIN` macro), so host apps never `import` it. The two integrations coexist — host apps use CocoaPods by default; SPM is available for apps that opt out of the Podfile. Both pin to the 1.24.x ORT family.

- **`CapacitorOnnxPlugin.swift`** — `@objc(CapacitorOnnxPlugin)` Capacitor plugin bridge; validates input, dispatches async Tasks, rejects with structured errors.
- **`CapacitorOnnxPlugin.m`** — ObjC `CAP_PLUGIN` macro registration.
- **`InferenceService.swift`** — Orchestrates prepareModel/warmup/run; per-session `NSLock` for serialized inference.
- **`SessionManager.swift`** — Manages `ORTSession` instances; CoreML EP with CPU fallback in `auto` mode.
- **`ModelStore.swift`** — File cache, download, SHA-256 (via CryptoKit), atomic rename.

### Key design decisions

- The plugin contract is intentionally minimal: `run()` receives named `inputs` (a `Record<string, RawTensor>` keyed by ONNX input name) and returns named `outputs` (keyed by ONNX output name). Pre/post-processing lives in the consumer app. Android accepts `float32`, `int64`, `int32`, `bool`, `uint8` input tensors; iOS accepts the same set except `bool` (onnxruntime-objc exposes no bool tensor type); `float16`/`uint32` are web-only.
- Output shape and dtype come straight from ORT on every platform: Web reads each `ort.Tensor`'s `dims`/`type`, Android reads `OnnxTensor.info` (`shape`/`type`), iOS reads `tensorTypeAndShapeInfo()` (`shape`/`elementType`). No heuristic shape resolution — if you ever need to reconstruct dims, prefer adding a new ORT API call over reintroducing inference-by-element-count logic.
- Session concurrency: per-session mutex/lock allows parallel inference across different models, but queues calls to the same `modelId+version`.
- Model integrity: download → temp file → SHA-256 check → atomic rename (Android and iOS). On Web, integrity is verified by creating and immediately releasing an `InferenceSession`.
- `loadModel.status` is strict: `cache_hit` only when a valid cached artifact is reused; `downloaded` when a network fetch occurred.
- Published as ESM-only (`type: module`). Consumers must use `import`.
- `CapacitorOnnxWeb.setWebConfig({ cacheStorage, wasmPath })` must be called before `loadModel` in Web contexts to configure the WASM path and a custom cache backend.
- iOS provider mapping: `cpu`→CPU, `nnapi`/`coreml`→CoreML, `auto`→CoreML with CPU fallback, web providers→CPU.

## Linter

Biome is configured (`biome.json`) but has no npm script wired up. Run it manually with `pnpm biome check src/` if needed.
