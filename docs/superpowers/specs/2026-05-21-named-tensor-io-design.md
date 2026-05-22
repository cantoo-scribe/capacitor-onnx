# Design: Named multi-tensor I/O for `run()`

**Date:** 2026-05-21
**Status:** Approved — ready for implementation planning
**Target version:** 3.0.0 (breaking change)

## Problem

A partner team integrated a wav2vec2 ASR model and loads it successfully through
the plugin, but inference fails:

```
Name: '/wav2vec2/Shape_1'
Missing Input: attention_mask
```

The model requires **two** named inputs — `input_values` (audio, `float32`) and
`attention_mask` (`int64`) — but the current `run()` contract exposes a single
`inputTensor`.

Two distinct gaps surface from this:

1. **Single input only.** `run()` accepts one `inputTensor: RawTensor` and binds
   it to `session.inputNames[0]` implicitly. There is no way to pass additional
   named inputs.
2. **`float32`-only on native.** Android (`InferenceService.kt`) and iOS
   (`InferenceService.swift`) reject any tensor type other than `float32`. Even
   with multi-input support, `attention_mask` (`int64`) would still be rejected.

Resolving the report requires both: named multi-tensor inputs **and** integer
type support on native.

## Goals

- `run()` accepts an arbitrary set of named input tensors.
- `run()` returns all named output tensors with their real ONNX dtype.
- Native (Android/iOS) accepts the common input dtypes, not just `float32`.
- The contract is symmetric (named map in, named map out) and has a single,
  explicit shape — no implicit positional binding to `inputNames[0]`.

## Non-goals

- `float16` and `uint32` tensor support on native. `float16` requires manual
  half-precision encoding/decoding on both platforms; `uint32` is rare as a
  model input. Both remain declared in `RawTensor.type` but produce a clear
  structured error if passed to native. Web passes whatever `onnxruntime-web`
  accepts.
- `bool` tensor support on **iOS**. Discovered at verification time:
  `onnxruntime-objc`'s `ORTTensorElementDataType` has no `bool` case. `bool`
  is therefore Android/Web only; iOS rejects it with a structured error.
- Backward compatibility with the 2.x `inputTensor`/`logits` contract. There is
  a single known consumer, mid-integration; coordinating one breaking change now
  is cheaper than two later.
- Resolving the `int64` JSON-transport precision limit (see Risks).

## Decisions

| # | Decision | Rationale |
|---|----------|-----------|
| 1 | Replace `inputTensor: RawTensor` with `inputs: Record<string, RawTensor>` | The ONNX Runtime input model is already a name→tensor map. A named map removes the fragile implicit `inputNames[0]` binding and needs no further API change to add/remove model inputs. |
| 2 | Native accepts `float32`, `int64`, `int32`, `bool`, `uint8` | Covers `attention_mask` (`int64`) and common input dtypes (token ids, masks). **Implementation note:** the ONNX Runtime Obj-C API (`onnxruntime-objc`) has no bool tensor element type, so `bool` is supported on Android and Web only — iOS rejects `bool` with a structured `INFERENCE_ERROR`. |
| 3 | Replace `warmupInput?: RawTensor` with `warmupInputs?: Record<string, RawTensor>` | Keeps `loadModel` consistent with `run()`; warmup otherwise fails the same way for multi-input models. |
| 4 | Replace `logits: RawTensor` with `outputs: Record<string, RawTensor>`, dtype read from ORT | Symmetric contract; supports multi-output models; eliminates the latent bug where native hardcodes the output type to `float32`. Bundled into the same breaking release. |

## API contract (`src/definitions.ts`)

```ts
export interface RunInput {
  modelId: string;
  version: string;
  inputs: Record<string, RawTensor>;
}

export interface RunResult {
  outputs: Record<string, RawTensor>;
  latencyMs: number;
}

export interface LoadModelInput {
  modelId: string;
  version: string;
  filePath?: string;
  modelBuffer?: Uint8Array;
  warmupInputs?: Record<string, RawTensor>;
  sessionOptions?: SessionOptionsInput;
}
```

`RawTensor` is unchanged — it already declares all seven types. The map key is
the exact ONNX input/output name.

Consumer call after migration:

```ts
const { outputs, latencyMs } = await CapacitorOnnx.run({
  modelId, version,
  inputs: {
    input_values:   { type: "float32", dims, data },
    attention_mask: { type: "int64",   dims, data },
  },
});
const logits = outputs.logits; // RawTensor
```

## Architecture

The plugin keeps its current structure. The change touches the input-parsing and
result-mapping edges of each platform; session management, caching, download,
and concurrency are untouched.

### Web — `src/web/index.ts` + new `src/web/tensor.ts`

New file `src/web/tensor.ts` (mirrors the existing `provider-resolver.ts` /
`runtime-config.ts` split):

- `toOrtTensor(raw: RawTensor): ort.Tensor` — converts by `type`. Required
  because the `ort.Tensor` constructor does **not** accept `number[]` for every
  type:
  - `int64` → `BigInt64Array.from(data, BigInt)`
  - `bool`, `uint8` → `Uint8Array`
  - `float32`, `int32` → `number[]` accepted directly
  Shared by `run()` and `warmupSession()`.
- `fromOrtTensor(t: ort.Tensor): RawTensor` —
  `{ type: t.type, dims: t.dims, data: Array.from(t.data, Number) }`.
  `Number(bigint)` handles `int64` outputs.

`run()` builds `{ name: toOrtTensor(...) }` from `inputs`, calls
`session.run(map)`, and converts **every** entry of the result into `outputs`.
The `inputNames[0]` / `outputNames[0]` lookups are removed.

### Android — `InferenceService.kt`, `CapacitorOnnxPlugin.kt`

`RawTensorInternal.data` changes from `FloatArray` to `DoubleArray` — a
type-neutral staging buffer, since the Capacitor bridge delivers JSON numbers as
doubles. The concrete typed buffer is produced only at tensor-creation time.

- `CapacitorOnnxPlugin.run()` parses the `inputs` JSObject into
  `Map<String, RawTensorInternal>` (each value: `data` as `DoubleArray`, `dims`
  as `LongArray`, `type` as `String`). Same change for `warmupInput` parsing.
- `buildOnnxTensor(env, raw): OnnxTensor` — `when (type)`:
  - `float32` → `FloatBuffer`
  - `int32` → `IntBuffer`
  - `int64` → `LongBuffer`
  - `uint8` → `ByteBuffer` + `OnnxJavaType.UINT8`
  - `bool` → `ByteBuffer` (0/1) + `OnnxJavaType.BOOL`
- `InferenceService.run()` receives `Map<String, RawTensorInternal>`, validates
  each input against `session.inputInfo[name]` (the existing rank/shape check,
  now applied per name), runs, and reads **all** outputs.
- `readOnnxTensor(value): RawTensorInternal` — reads each output's real
  `OnnxJavaType`, flattens to `DoubleArray`, maps the type back to a
  `RawTensor.type` string. Replaces the `flattenToFloatArray` +
  hardcoded-`"float32"` path.

### iOS — `InferenceService.swift`, `CapacitorOnnxPlugin.swift`

`RawTensorInternal.data` changes from `[Float]` to `[Double]` for the same
staging reason.

- `CapacitorOnnxPlugin.run()` parses the `inputs` object into
  `[String: RawTensorInternal]`. Same change for `warmupInput` parsing.
- `buildOrtValue(raw): ORTValue` — packs an `NSMutableData` at the correct
  element size and constructs `ORTValue(tensorData:elementType:shape:)` with
  `.float` / `.int32` / `.int64` / `.uInt8` / `.bool`.
- `InferenceService.run()` receives `[String: RawTensorInternal]`, runs against
  the full `outputNames` set, and reads each output via
  `tensorTypeAndShapeInfo().elementType` mapped back to a `RawTensor.type`
  string.

## Validation & error handling

All structured errors use `INFERENCE_ERROR`, consistent across web/Android/iOS.

Per call:

- `inputs` missing or empty → error.

Per input tensor:

- `type` not in `{float32, int64, int32, bool, uint8}` (native) →
  `"unsupported tensor type '<type>' for input '<name>'"`.
- `data` or `dims` missing → error.
- any dimension `<= 0` → error.
- `product(dims) != data.length` → error.
- map key is not a real model input name →
  `"model has no input named '<name>'"` (catches typos).

Missing **required** model input: not pre-validated. ORT raises it, and its
message is already exactly `Missing Input: attention_mask`; it is surfaced as a
structured `INFERENCE_ERROR`. Pre-validating completeness is skipped because ORT
handles genuinely optional inputs correctly and we cannot cheaply distinguish
required from optional through the Java/Obj-C APIs.

## Versioning, docs, testing

- `npm version major` → `3.0.0`; the `package.json` script syncs
  `CantooCapacitorOnnx.podspec`.
- Update `CLAUDE.md`: the contract description ("receives a preprocessed
  `inputTensor` and returns raw `logits`") and the output-shape paragraph.
- Update `docs/testing-scripts.md` with a wav2vec2 multi-input example; update
  `README` if it shows `run()` usage.
- Verification: `pnpm build`, `pnpm typecheck`, `pnpm pack --dry-run`. No
  automated tests exist in the repo — a multi-input model is tested manually per
  `docs/testing-scripts.md`.

## Risks

- **`int64` JSON precision.** `RawTensor.data` is `number[]`; values transit the
  Capacitor bridge as JSON numbers (doubles). Integers above 2^53 lose
  precision. This is sufficient for `attention_mask` (0/1) and typical token ids,
  but is a documented limitation, not solved here.
- **Consumer migration.** The partner team must change `inputTensor: X` →
  `inputs: { "<name>": X }` and `result.logits` → `result.outputs["<name>"]`.
  They are already editing the call site for this feature, so the cost is
  marginal.
