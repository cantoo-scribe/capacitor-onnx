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
- `getInputTensor(...)`
- `createHostBridge(...)`
- `createIFrameBridge(...)`
- `OnnxIFrameClient`
- `OnnxHostDispatcher`
- `ONNX_BRIDGE_KEYS`
- TypeScript interfaces from `definitions`

Wrappers parity (`OnnxIFrameClient` and `OnnxHostDispatcher`):

- `isActive`
- `prepareModel`
- `warmupModel`
- `runInference`
- `getModelStatus`
- `clearModel`
- `clearAllCache`
- `getDiagnostics`
- `getInputTensor` (synchronous helper)

### Example

```ts
import { CapacitorOnnx } from '@cantoo/capacitor-onnx';

await CapacitorOnnx.prepareModel({
  modelId: 'demo-model',
  version: '1.0.0',
  url: 'https://example.com/model.onnx',
});

const inputTensor = CapacitorOnnx.getInputTensor([0.1, 0.2, 0.3, 0.4]);

const result = await CapacitorOnnx.runInference({
  modelId: 'demo-model',
  version: '1.0.0',
  inputTensor,
});

console.log(result.logits.shape, result.logits.data.length);
```

## Host / iFrame Bridge

For web integration scenarios, use the standardized async protocol over `postMessage`.

Protocol envelope:

```ts
{
  key: string;
  requestId?: string;
  data?: unknown;
  err?: { message: string; name?: string; stack?: string; details?: unknown };
}
```

Main request keys (`iFrame -> Host`):

- `plugin:active`
- `INITIALIZE_MODEL`
- `WARMUP_MODEL`
- `RUN_INFERENCE`

Main status keys (`Host -> iFrame`):

- `is-active:requested` / `is-active:result` / `is-active:error`
- `initialize-model:requested` / `initialize-model:result` / `initialize-model:error`
- `warmup-model:requested` / `warmup-model:result` / `warmup-model:error`
- `run-inference:requested` / `run-inference:result` / `run-inference:error`

### Host app

```ts
import { CapacitorOnnx, createHostBridge, OnnxHostDispatcher } from '@cantoo/capacitor-onnx';

const iframe = document.querySelector<HTMLIFrameElement>('#native-iframe');
if (!iframe?.contentWindow) {
  throw new Error('iFrame window not available');
}

const hostBridge = createHostBridge({
  iframeWindow: iframe.contentWindow,
  targetOrigin: 'https://iframe.example.com',
  allowedOrigins: ['https://iframe.example.com'],
});

const dispatcher = new OnnxHostDispatcher(hostBridge, {
  isActive: () => CapacitorOnnx.isActive(),
  prepareModel: (input) => CapacitorOnnx.prepareModel(input),
  warmupModel: (input) => CapacitorOnnx.warmupModel(input),
  runInference: (input) => CapacitorOnnx.runInference(input),
  getModelStatus: (input) => CapacitorOnnx.getModelStatus(input),
  clearModel: (input) => CapacitorOnnx.clearModel(input),
  clearAllCache: () => CapacitorOnnx.clearAllCache(),
  getDiagnostics: () => CapacitorOnnx.getDiagnostics(),
});

const stopDispatcher = dispatcher.start();

// Later: stopDispatcher(); or dispatcher.stop();
```

### iFrame app

```ts
import { createIFrameBridge, OnnxIFrameClient } from '@cantoo/capacitor-onnx';

const iframeBridge = createIFrameBridge({
  targetOrigin: 'https://host.example.com',
  allowedOrigins: ['https://host.example.com'],
});

const client = new OnnxIFrameClient(iframeBridge, {
  requestTimeoutMs: 15000,
});

const isActive = await client.isActive();
console.log('isActive', isActive);

const prepared = await client.prepareModel({
  modelId: 'demo-model',
  version: '1.0.0',
  url: 'https://example.com/model.onnx',
});

console.log('prepareModel', prepared);

const diagnostics = await client.getDiagnostics();
console.log('diagnostics', diagnostics);

// Later: client.destroy();
```

## Runtime Notes

- `prepareModel` supports optional `sha256` for integrity verification.
- Input to `runInference` is raw tensor data (`inputTensor`), keeping preprocessing in the app layer.
- Errors are normalized with structured fields (`code`, `message`, `retryable`, `correlationId`, `details`).

## License

MIT
