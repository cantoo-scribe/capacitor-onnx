# @cantoo/capacitor-onnx

Capacitor plugin for ONNX Runtime inference on Android, iOS and Web.

## Install

```bash
pnpm add @cantoo/capacitor-onnx
pnpm cap sync android
pnpm cap sync ios
```

### Android setup

`pnpm cap sync android` registers the plugin automatically; no manual `MainActivity` edits are required. The host app must satisfy:

- **`minSdk` ≥ 24** (Android 7.0).
- **`compileSdk` ≥ 34**.
- **JDK ≥ 17** on the build machine. The plugin targets Java 17 bytecode (`sourceCompatibility` / `targetCompatibility` / `kotlinOptions.jvmTarget = '17'`), so any newer JDK (e.g. 21) also works — 17 is just the floor.
- **`android.permission.INTERNET`** in the manifest if `loadModel` will fetch models over HTTPS (default in Capacitor templates).

The `com.microsoft.onnxruntime:onnxruntime-android` dependency is bundled by the plugin's `build.gradle` — you do not need to add it yourself. Tune execution providers and threading through `sessionOptions` (see [docs/android-optimization.md](docs/android-optimization.md)).

### iOS setup

iOS is distributed via **Swift Package Manager**. The `onnxruntime` dependency is not published on CocoaPods, so you must integrate the plugin through SPM:

1. After `pnpm cap sync ios`, open `ios/App/App.xcworkspace` in Xcode.
2. Select the **App** project → **Package Dependencies** → **+**.
3. Add the local package by pointing to the plugin folder: `node_modules/@cantoo/capacitor-onnx` (or the absolute path on disk). Xcode resolves [`onnxruntime-swift-package-manager`](https://github.com/microsoft/onnxruntime-swift-package-manager) transitively.
4. Add the **CapacitorOnnx** product to the **App** target.
5. Minimum deployment target: **iOS 14**.

The native bridge is registered automatically via `CapacitorOnnxPlugin.m`; no additional Swift code is required.

### Web setup

`onnxruntime-web` requires the page to be served as a **cross-origin isolated** context — without it the multi-threaded WASM backend falls back (or fails) and `SharedArrayBuffer` is unavailable. The host page **must** be served with the following response headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Plus, any cross-origin asset the page loads (model files, WASM artifacts, fonts, images) needs `Cross-Origin-Resource-Policy: cross-origin` (or `same-site`) on its response, otherwise it will be blocked under COEP. CDN/Storage hosting your `.onnx` artifacts must also send permissive **CORS** headers (`Access-Control-Allow-Origin`).

For Web-only hosts (without Capacitor), import from the dedicated Web entrypoint and configure WASM path/cache before `loadModel`:

```ts
import { CapacitorOnnxWeb } from '@cantoo/capacitor-onnx/web';

CapacitorOnnxWeb.setWebConfig({
  wasmPath: '/ort-wasm/',
  cacheStorage: myCacheBackend,
});
```

Symptoms of missing isolation/CORS: `SharedArrayBuffer is not defined`, `NetworkError when fetching .wasm`, or models silently downgrading to single-threaded execution.

## API

The package exports:

- `CapacitorOnnx`
- `CapacitorOnnxWeb` (from `@cantoo/capacitor-onnx/web` for non-Capacitor hosts)
- TypeScript interfaces from `definitions`

Host/iFrame bridge implementation is no longer part of this package and was moved to a dedicated package.

### Methods

| Method | Signature | Purpose | Notes |
| --- | --- | --- | --- |
| `loadModel` | `(input: LoadModelInput) => Promise<LoadModelResult>` | Downloads (or reuses cached) model bytes, validates them, creates the inference session, and optionally warms it up. Must be called once per `modelId+version` before `run`. | `status` returns `cache_hit` or `downloaded`. Pass `sha256` to enforce integrity, `warmupInput` (a `RawTensor` matching one valid input shape) to pay first-inference cost upfront, `forceRedownload: true` to bypass cache, `timeoutMs` to bound the network fetch, and `sessionOptions` to pick the execution provider / thread counts. The result includes `executionProviderUsed`. |
| `run` | `(input: RunInput) => Promise<RunResult>` | Runs inference on a previously loaded session. Resolves I/O names from session metadata, so the consumer only supplies `inputTensor`. | Calls to the same `modelId+version` are serialized by a per-session lock; different models run in parallel. Returns `{ logits, latencyMs }`. Pre/post-processing is the consumer's responsibility. |
| `release` | `(input: ClearModelInput) => Promise<void>` | Releases the in-memory ONNX session for the given `modelId+version`. The cached file on disk is **kept**. | Use to free RAM/GPU memory when you are done with a model but expect to use it again later (next `loadModel` will hit the cache). |
| `clearModel` | `(input: ClearModelInput) => Promise<ClearModelResult>` | Releases the session **and** removes the cached artifact for that `modelId+version` from disk / `cacheStorage`. | Returns `{ removed: boolean }`. Use when rotating a model version or invalidating a corrupted cache entry. |
| `clearAllCache` | `() => Promise<ClearAllCacheResult>` | Releases every active session and wipes all cached model artifacts. | Returns `{ removedModels: number }`. Useful for "log out" / "factory reset" flows. |

Type definitions for every input/result (e.g. `LoadModelInput`, `RawTensor`, `SessionOptionsInput`, `PluginError`) live in [src/definitions.ts](src/definitions.ts).

### Example

```ts
import { CapacitorOnnx } from '@cantoo/capacitor-onnx';

await CapacitorOnnx.loadModel({
  modelId: 'demo-model',
  version: '1.0.0',
  url: 'https://example.com/model.onnx',
});

const result = await CapacitorOnnx.run({
  modelId: 'demo-model',
  version: '1.0.0',
  inputTensor: {
    type: 'float32',
    dims: [1, 4],
    data: [0.1, 0.2, 0.3, 0.4],
  },
});

console.log(result.logits.dims, result.logits.data.length);

await CapacitorOnnx.clearModel({ modelId: 'demo-model', version: '1.0.0' });
```

## Runtime Notes

- `loadModel` supports optional `sha256` for integrity verification.
- `loadModel` supports optional `warmupInput: RawTensor` to pre-run the session with a sample tensor of the exact shape the model expects (e.g. `{ type: 'float32', dims: [1, 16000], data: [...] }`). Warmup is skipped when `warmupInput` is omitted.
- `loadModel.status` semantics are strict: `cache_hit` when loaded from valid cache, `downloaded` when network download is used.
- `loadModel` returns `executionProviderUsed` with the provider that was actually initialized.
- Web provider selection supports `sessionOptions.executionProvider` with `auto`, `wasm`, `webgpu`, `webnn` plus native aliases (`cpu`/`nnapi`/`coreml` mapped to `wasm` in Web).
- In Web `auto` mode, provider resolution tries accelerated providers first (`webgpu`, `webnn`) and falls back to `wasm`.
- iOS provider mapping: `cpu` → CPU, `nnapi`/`coreml` → CoreML, `auto` → CoreML with CPU fallback, web providers (`wasm`/`webgpu`/`webnn`) → CPU.
- `run` accepts `inputTensor` and resolves model I/O names from session metadata (`inputNames`/`outputNames`) instead of hardcoded names.
- **Output shape**: `RunResult.logits.dims` is the shape ORT materialized for the output tensor — Web reads `outputTensor.dims`, Android reads `OnnxTensor.info.shape`, iOS reads `tensorTypeAndShapeInfo().shape`. No heuristic, no symbolic dims (`-1`) in the result, no batch assumptions. Models with multiple independent dynamic axes are returned with their true runtime shape.
- Web runtime config is split by concern: [src/web-runtime-config.ts](src/web-runtime-config.ts) (global runtime/threads) and [src/web-provider-resolver.ts](src/web-provider-resolver.ts) (provider resolution and fallback).
- Errors are normalized with structured fields (`code`, `message`, `retryable`, `correlationId`, `details`).

## Docs

- Testing scripts and validation flow: [docs/testing-scripts.md](docs/testing-scripts.md)

## License

MIT
