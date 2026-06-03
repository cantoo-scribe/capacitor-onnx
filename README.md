# @cantoo/capacitor-onnx

Capacitor plugin for ONNX Runtime inference on Android, iOS and Web.

## Migration from 1.x to 2.0

`2.0.0` removes the plugin-side model cache. The plugin no longer downloads, validates, or stores model files — it is now a thin wrapper around ONNX Runtime sessions. The host app owns model storage and provides bytes (web) or a filesystem path (native).

### Contract changes

- `LoadModelInput` no longer accepts `url`, `sha256`, `forceRedownload`, or `timeoutMs`. Pass either `filePath` (iOS/Android) **or** `modelBuffer: Uint8Array` (web).
- `LoadModelResult` no longer includes `status` (`cache_hit` / `downloaded`).
- Methods `clearModel` and `clearAllCache` have been removed. `release(modelId, version)` still releases the in-memory ORT session.
- `CapacitorOnnxWeb.setWebConfig` no longer accepts `cacheStorage` — only `wasmPath`.
- Error codes `NETWORK_ERROR`, `INTEGRITY_ERROR`, and `MODEL_INTEGRITY_ERROR` are no longer reachable.

### Migration example

Before:

```ts
await CapacitorOnnx.loadModel({
  modelId: 'demo-model',
  version: '1.0.0',
  url: 'https://example.com/model.onnx',
  sha256: 'abc...',
});
```

After (native, iOS/Android):

```ts
// Download/cache the model in your app code, e.g. via @capacitor/filesystem.
// Then pass the absolute or file:// path to the plugin.
await CapacitorOnnx.loadModel({
  modelId: 'demo-model',
  version: '1.0.0',
  filePath: '/data/user/0/com.app/files/models/demo-model-1.0.0.onnx',
});
```

After (web):

```ts
const response = await fetch('https://example.com/model.onnx');
const modelBuffer = new Uint8Array(await response.arrayBuffer());

await CapacitorOnnx.loadModel({
  modelId: 'demo-model',
  version: '1.0.0',
  modelBuffer,
});
```

Passing `modelBuffer` on iOS/Android or `filePath` on web rejects with `MODEL_INVALID` — the Capacitor bridge serializes `Uint8Array` inefficiently (base64 / number array), so native callers must always use filesystem paths.

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

The `com.microsoft.onnxruntime:onnxruntime-android` dependency is bundled by the plugin's `build.gradle` — you do not need to add it yourself. Tune execution providers and threading through `sessionOptions` (see [docs/android-optimization.md](docs/android-optimization.md)).

### iOS setup

iOS supports both **CocoaPods** (default for Capacitor apps) and **Swift Package Manager**.

**CocoaPods (recommended for Capacitor apps).** `pnpm cap sync ios` registers the plugin automatically: the generated `Podfile` picks up `CantooCapacitorOnnx.podspec` from `node_modules/@cantoo/capacitor-onnx`, and `pod install` resolves [`onnxruntime-objc`](https://cocoapods.org/pods/onnxruntime-objc) transitively. No manual Xcode steps are required.

**Swift Package Manager (alternative).** If the host app prefers SPM, skip the Podfile entry and add the plugin as a local package in Xcode (**Package Dependencies → +**, pointing to `node_modules/@cantoo/capacitor-onnx`). Xcode resolves [`onnxruntime-swift-package-manager`](https://github.com/microsoft/onnxruntime-swift-package-manager) transitively. Add the **CapacitorOnnx** product to the **App** target.

Requirements either way:
- Minimum deployment target: **iOS 14**.
- The native bridge is registered automatically via `CapacitorOnnxPlugin.m`; no additional Swift code is required.

### Web setup

`onnxruntime-web` requires the page to be served as a **cross-origin isolated** context — without it the multi-threaded WASM backend falls back (or fails) and `SharedArrayBuffer` is unavailable. The host page **must** be served with the following response headers:

```
Cross-Origin-Opener-Policy: same-origin
Cross-Origin-Embedder-Policy: require-corp
```

Plus, any cross-origin asset the page loads (model files, WASM artifacts, fonts, images) needs `Cross-Origin-Resource-Policy: cross-origin` (or `same-site`) on its response, otherwise it will be blocked under COEP. CDN/Storage hosting your `.onnx` artifacts must also send permissive **CORS** headers (`Access-Control-Allow-Origin`).

For Web-only hosts (without Capacitor), import from the dedicated Web entrypoint and configure the WASM path before `loadModel`:

```ts
import { CapacitorOnnxWeb } from '@cantoo/capacitor-onnx/web';

CapacitorOnnxWeb.setWebConfig({
  wasmPath: '/ort-wasm/',
});
```

Symptoms of missing isolation/CORS: `SharedArrayBuffer is not defined`, `NetworkError when fetching .wasm`, or models silently downgrading to single-threaded execution.

## API

The package exports:

- `CapacitorOnnx`
- `CapacitorOnnxWeb` (from `@cantoo/capacitor-onnx/web` for non-Capacitor hosts)
- TypeScript interfaces from `definitions`

### Methods

| Method | Signature | Purpose | Notes |
| --- | --- | --- | --- |
| `loadModel` | `(input: LoadModelInput) => Promise<LoadModelResult>` | Creates an ONNX Runtime session from the model bytes (web) or file path (native), and optionally warms it up. Must be called once per `modelId+version` before `run`. | Native: pass `filePath` (absolute path or `file://` URI). Web: pass `modelBuffer: Uint8Array`. Pass `warmupInputs` (a `Record<string, RawTensor>` keyed by model input name) to pay first-inference cost upfront, and `sessionOptions` to pick the execution provider / thread counts. The result includes `executionProviderUsed`. |
| `run` | `(input: RunInput) => Promise<RunResult>` | Runs inference on a previously loaded session. | Pass `inputs` as a `Record<string, RawTensor>` keyed by the model's ONNX input names. Calls to the same `modelId+version` are serialized by a per-session lock; different models run in parallel. Returns `{ outputs, latencyMs }`, where `outputs` is keyed by the model's output names. Pre/post-processing is the consumer's responsibility. |
| `release` | `(input: ReleaseModelInput) => Promise<void>` | Releases the in-memory ONNX session for the given `modelId+version`. | Use to free RAM/GPU memory when you are done with a model. The host app is responsible for managing model files on disk. |

Type definitions for every input/result (e.g. `LoadModelInput`, `RawTensor`, `SessionOptionsInput`, `PluginError`) live in [src/definitions.ts](src/definitions.ts).

### Example

```ts
import { Capacitor } from '@capacitor/core';
import { CapacitorOnnx } from '@cantoo/capacitor-onnx';

async function loadDemoModel() {
  if (Capacitor.getPlatform() === 'web') {
    const response = await fetch('https://example.com/model.onnx');
    const modelBuffer = new Uint8Array(await response.arrayBuffer());
    await CapacitorOnnx.loadModel({
      modelId: 'demo-model',
      version: '1.0.0',
      modelBuffer,
    });
    return;
  }

  // On iOS/Android, the host app is responsible for downloading
  // the model to the filesystem (e.g. via @capacitor/filesystem).
  await CapacitorOnnx.loadModel({
    modelId: 'demo-model',
    version: '1.0.0',
    filePath: '/absolute/path/to/model.onnx',
  });
}

await loadDemoModel();

const { outputs } = await CapacitorOnnx.run({
  modelId: 'demo-model',
  version: '1.0.0',
  inputs: {
    input_values: {
      type: 'float32',
      dims: [1, 16000],
      data: [/* normalized audio samples */],
    },
    attention_mask: {
      type: 'int64',
      dims: [1, 16000],
      data: [/* 1s for real samples, 0s for padding */],
    },
  },
});

const logits = outputs.logits;
console.log(logits.dims, logits.data.length);

await CapacitorOnnx.release({ modelId: 'demo-model', version: '1.0.0' });
```

## Runtime Notes

- `loadModel` supports optional `warmupInputs: Record<string, RawTensor>` to pre-run the session with sample tensors keyed by model input name (e.g. `{ input_values: { type: 'float32', dims: [1, 16000], data: [...] } }`). Warmup is skipped when `warmupInputs` is omitted.
- `loadModel` returns `executionProviderUsed` with the provider that was actually initialized.
- Web provider selection supports `sessionOptions.executionProvider` with `auto`, `wasm`, `webgpu`, `webnn` plus native aliases (`cpu`/`nnapi`/`coreml` mapped to `wasm` in Web).
- In Web `auto` mode, provider resolution tries accelerated providers first (`webgpu`, `webnn`) and falls back to `wasm`.
- iOS provider mapping: `cpu` → CPU, `nnapi`/`coreml` → CoreML, `auto` → CoreML with CPU fallback, web providers (`wasm`/`webgpu`/`webnn`) → CPU.
- `run` takes `inputs` keyed by ONNX input name and returns every model output in `outputs` keyed by ONNX output name. Android accepts `float32`, `int64`, `int32`, `bool`, `uint8`; iOS accepts the same set **except `bool`** (the ONNX Runtime Obj-C API exposes no bool tensor type). `float16`/`uint32` are web-only. Unsupported types are rejected on native with a structured error.
- **Output shape & dtype**: each `RunResult.outputs` tensor carries the shape and dtype ORT materialized — Web reads `ort.Tensor.dims`/`.type`, Android reads `OnnxTensor.info.shape`/`.type`, iOS reads `tensorTypeAndShapeInfo().shape`/`.elementType`. No heuristic, no symbolic dims (`-1`) in the result.
- Errors are normalized with structured fields (`code`, `message`, `retryable`, `correlationId`, `details`).

## Model format: default `.onnx` vs reduced `.ort` (Android)

There are two ways to ship a model, and you pick per app:

### Default — `.onnx` (Android, iOS, Web)

The standard path used in all the examples above: the plugin bundles the **full**
`onnxruntime-android` AAR (and `onnxruntime-objc`/`onnxruntime-web` on the other
platforms), and you load a plain **`.onnx`** model. Works everywhere, **no extra tooling
or setup** — just `loadModel`. This is the default; if you do nothing, you get this.

### Reduced ops — `.ort` (Android only, opt-in)

For Android you can shrink the native runtime by compiling a `libonnxruntime.so` with
**only your model's operators** (~57–59% smaller on arm64) and loading a pre-optimized
**`.ort`** model instead of the `.onnx`. It is **opt-in and Android-only**; iOS/Web keep
the default path.

```bash
# in your app package (depends on @cantoo/capacitor-onnx)
pnpm exec cantoo-onnx-reduce      # or: npx cantoo-onnx-reduce
```

The **first** build (the "generator") needs a toolchain (Python ≥3.10 + `onnxruntime`/`onnx`,
Android NDK, CMake/Ninja, JDK 21; bash — on Windows use WSL2) and a small amount of app-side
wiring (load the `.ort`, whose filename must end in `.ort`). Once the generator publishes the
op-config, `.ort` and AAR (via the `onnx*UploadUrl` keys), **other devs/CI need none of that
toolchain** — with `onnxConfigUrl` + `onnxCacheUrl` set, their build just downloads the
op-config and the prebuilt AAR (no Python, no NDK, no model download). It stays fully opt-out —
clearing `onnxModel` reverts to the full AAR + `.onnx`.

**Full guide:** [docs/reduced-onnx.md](docs/reduced-onnx.md).

## Docs

- Reduced ONNX Runtime build (smaller Android binary): [docs/reduced-onnx.md](docs/reduced-onnx.md)
- Android optimization & execution providers: [docs/android-optimization.md](docs/android-optimization.md)
- Testing scripts and validation flow: [docs/testing-scripts.md](docs/testing-scripts.md)

## License

MIT
