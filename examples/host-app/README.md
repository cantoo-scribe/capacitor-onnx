# Host App (Smoke Test)

Minimal host app to validate integration of the `@cantoo/capacitor-onnx` plugin with Android in a real environment.

## Prerequisites

- Node.js 18+
- pnpm 10+
- JDK 21
- Android SDK configurado

## Quick flow

1. Install dependencies:

```bash
pnpm install
```

2. Build web:

```bash
pnpm build
```

3. Add Android (once):

```bash
pnpm cap:add:android
```

4. Sync plugin/assets:

```bash
pnpm cap:sync
```

5. Build Android:

```bash
pnpm android:assemble
```

6. Full pipeline (after Android exists):

```bash
pnpm pipeline:validate
```

## Quick troubleshooting

- Android build failure due to missing SDK/NDK:
	- Check environment variables and Android SDK configuration on the host.
	- Run `pnpm cap:sync` again after fixing the environment.

## Functional smoke test

- **Get diagnostics** button: validates the bridge with a simple call.
- **Quick preset** field + **Apply preset** button:
	- Quickly fills `modelId`, `version`, and `normalized input`.
	- Keeps `url` and `sha256` (optional) for you to provide model values.
	- Includes short and long audio presets to speed up manual inference tests.
- **Generate mock audio** button:
	- Generates a synthetic audio signal (sine wave with harmonic) and automatically fills the `Normalized input (CSV)` field.
	- Lets you test inference without manually pasting thousands of samples.
- **Run success E2E** button:
	- Uses configuration fields (modelId, version, url, optional sha256, normalized input).
	- Automatically converts input to tensor via the `getInputTensor(...)` helper.
	- Runs the sequence `prepareModel -> warmupModel -> runInference`.
	- Validates minimal contract assertions (`sessionReady`, `warmed`, `logits.type`, shape/data consistency, numeric latency).
- **Run error E2E** button:
	- Forces `runInference` with a missing model.
	- Validates structured error contract (`code`, `message`, `retryable`, `correlationId`).
- **Clear model cache** button:
	- Clears the current model cache (`modelId` + `version`) on device.
- **Clear all cache** button:
	- Clears all prepared models on device.

## Concurrent call protection

- While `prepareModel` is running, action buttons are disabled.
- Concurrent calls for the same model load wait for the first call (deduplication by model/version/url/hash key).

## Note about E2E success

For the success flow, you must provide a real `url` to an ONNX model compatible with the configured input tensor. `sha256` can be temporarily omitted.
