# @cantoo/capacitor-onnx

Capacitor plugin for ONNX Runtime inference on Android and Web.

## Install

```bash
pnpm add @cantoo/capacitor-onnx
pnpm cap sync android
```

## API

The package exports:

- `CapacitorOnnx`
- TypeScript interfaces from `definitions`

Current `CapacitorOnnx` facade methods:

- `loadModel(input)`
- `run({ modelId, version, inputTensor })`
- `clear()` (returns a bound `clearModel` function)
- `clearAllCache()` (returns a bound function)

Host/iFrame bridge implementation is no longer part of this package and was moved to a dedicated package.

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

const clearModel = CapacitorOnnx.clear();
await clearModel({ modelId: 'demo-model', version: '1.0.0' });
```

## Runtime Notes

- `loadModel` supports optional `sha256` for integrity verification.
- `loadModel` supports optional `warmup` (`true`) to warm the session right after loading.
- `loadModel.status` semantics are strict: `cache_hit` when loaded from valid cache, `downloaded` when network download is used.
- `loadModel` returns `executionProviderUsed` with the provider that was actually initialized.
- Web provider selection supports `sessionOptions.executionProvider` with `auto`, `wasm`, `webgpu`, `webnn` plus Android aliases (`cpu`/`nnapi` mapped to `wasm` in Web).
- In Web `auto` mode, provider resolution tries accelerated providers first (`webgpu`, `webnn`) and falls back to `wasm`.
- `run` accepts `inputTensor` and resolves model I/O names from session metadata (`inputNames`/`outputNames`) instead of hardcoded names.
- Web runtime config is split by concern: [src/web-runtime-config.ts](src/web-runtime-config.ts) (global runtime/threads) and [src/web-provider-resolver.ts](src/web-provider-resolver.ts) (provider resolution and fallback).
- Errors are normalized with structured fields (`code`, `message`, `retryable`, `correlationId`, `details`).

## Docs

- Testing scripts and validation flow: [docs/testing-scripts.md](docs/testing-scripts.md)

## License

MIT
