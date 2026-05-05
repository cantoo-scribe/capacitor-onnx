# @cantoo/capacitor-onnx

Capacitor plugin for native ONNX Runtime inference on Android.

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
- `run({ modelId, version, audioData })`
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
  audioData: [0.1, 0.2, 0.3, 0.4],
});

console.log(result.logits.dims, result.logits.data.length);

const clearModel = CapacitorOnnx.clear();
await clearModel({ modelId: 'demo-model', version: '1.0.0' });
```

## Runtime Notes

- `loadModel` supports optional `sha256` for integrity verification.
- `loadModel` supports optional `warmup` (`true`) to warm the session right after loading.
- `run` accepts normalized `audioData`; tensor conversion is handled by the SDK helper.
- Errors are normalized with structured fields (`code`, `message`, `retryable`, `correlationId`, `details`).

## Docs

- Testing scripts and validation flow: [docs/testing-scripts.md](docs/testing-scripts.md)

## License

MIT
