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
npm version patch   # or minor/major — also updates CapacitorOnnx.podspec automatically via package.json
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

Distributed via **Swift Package Manager** (`Package.swift`). There is no CocoaPods podspec — `onnxruntime` is not published on CocoaPods. Consumers add the plugin as an SPM package in Xcode, which resolves `onnxruntime-swift-package-manager` automatically.

- **`CapacitorOnnxPlugin.swift`** — `@objc(CapacitorOnnxPlugin)` Capacitor plugin bridge; validates input, dispatches async Tasks, rejects with structured errors.
- **`CapacitorOnnxPlugin.m`** — ObjC `CAP_PLUGIN` macro registration.
- **`InferenceService.swift`** — Orchestrates prepareModel/warmup/run; per-session `NSLock` for serialized inference.
- **`SessionManager.swift`** — Manages `ORTSession` instances; CoreML EP with CPU fallback in `auto` mode.
- **`ModelStore.swift`** — File cache, download, SHA-256 (via CryptoKit), atomic rename.

### Key design decisions

- The plugin contract is intentionally minimal: it receives a preprocessed `inputTensor` and returns raw `logits`. Pre/post-processing lives in the consumer app.
- Session concurrency: per-session mutex/lock allows parallel inference across different models, but queues calls to the same `modelId+version`.
- Model integrity: download → temp file → SHA-256 check → atomic rename (Android and iOS). On Web, integrity is verified by creating and immediately releasing an `InferenceSession`.
- `loadModel.status` is strict: `cache_hit` only when a valid cached artifact is reused; `downloaded` when a network fetch occurred.
- Published as ESM-only (`type: module`). Consumers must use `import`.
- `CapacitorOnnxWeb.setWebConfig({ cacheStorage, wasmPath })` must be called before `loadModel` in Web contexts to configure the WASM path and a custom cache backend.
- iOS provider mapping: `cpu`→CPU, `nnapi`/`coreml`→CoreML, `auto`→CoreML with CPU fallback, web providers→CPU.

## Linter

Biome is configured (`biome.json`) but has no npm script wired up. Run it manually with `pnpm biome check src/` if needed.
