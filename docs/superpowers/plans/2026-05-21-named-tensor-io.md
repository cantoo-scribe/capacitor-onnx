# Named Multi-Tensor I/O Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the single `inputTensor`/`logits` contract of `run()` with named `inputs`/`outputs` maps and add integer-dtype support on native, so multi-input models (e.g. wav2vec2 with `attention_mask`) work.

**Architecture:** The change touches only the input-parsing and result-mapping edges of each platform — session management, caching, download, and concurrency are untouched. Web gains a `tensor.ts` conversion helper; Android and iOS gain per-dtype tensor builders/readers and switch their internal `RawTensorInternal` staging buffer to a type-neutral `double`/`Double` array.

**Tech Stack:** TypeScript + `onnxruntime-web`; Kotlin + `onnxruntime-android`; Swift + `onnxruntime-objc`; Capacitor plugin bridge.

**Spec:** [docs/superpowers/specs/2026-05-21-named-tensor-io-design.md](../specs/2026-05-21-named-tensor-io-design.md)

---

## Conventions for this plan

- **No test harness.** This repo has no automated test framework (see `CLAUDE.md`). The TDD "write a failing test" step is replaced by **compile/build verification**: `pnpm typecheck` + `pnpm build` for TypeScript, and host-app compilation for native code. Each task ends with an explicit verification step.
- **No commit steps.** Commits are intentionally omitted per project preference. After a task's verification passes, the maintainer commits at their discretion.
- **Breaking change.** This targets `3.0.0`. The version bump and publish are a maintainer action described in the Release section, not a task.

## File Structure

| File | Change | Responsibility |
| --- | --- | --- |
| `src/definitions.ts` | Modify | Public contract: `RunInput.inputs`, `RunResult.outputs`, `LoadModelInput.warmupInputs`. |
| `src/web/tensor.ts` | Create | `toOrtTensor` / `fromOrtTensor` — `RawTensor` ⇄ `ort.Tensor` conversion. |
| `src/web/index.ts` | Modify | Web `run()` / `loadModel()` warmup over named maps. |
| `examples/host-app/src/main.ts` | Modify | Smoke runner updated to the named-map contract. |
| `android/src/main/java/com/cantoo/capacitor/onnx/InferenceService.kt` | Modify | Per-dtype `OnnxTensor` build/read; `run`/`warmup` over named maps. |
| `android/src/main/java/com/cantoo/capacitor/onnx/CapacitorOnnxPlugin.kt` | Modify | Parse `inputs`/`warmupInputs` JSON maps; serialize `outputs`. |
| `ios/Plugin/InferenceService.swift` | Modify | Per-dtype `ORTValue` build/read; `run`/`warmup` over named maps. |
| `ios/Plugin/CapacitorOnnxPlugin.swift` | Modify | Parse `inputs`/`warmupInputs` objects; serialize `outputs`. |
| `CLAUDE.md`, `README.md`, `docs/testing-scripts.md` | Modify | Documentation. |

Task order: TypeScript contract → example app (restores green build) → Android → iOS → docs. Each native task is verified by compiling the example host-app.

---

## Task 1: TypeScript contract + Web implementation

**Files:**
- Modify: `src/definitions.ts`
- Create: `src/web/tensor.ts`
- Modify: `src/web/index.ts`

- [ ] **Step 1: Update the contract in `src/definitions.ts`**

Replace the `LoadModelInput`, `RunInput`, and `RunResult` interfaces (lines 21-28, 44-48, 56-59) with:

```ts
export interface LoadModelInput {
  modelId: string;
  version: string;
  filePath?: string;
  modelBuffer?: Uint8Array;
  warmupInputs?: Record<string, RawTensor>;
  sessionOptions?: SessionOptionsInput;
}
```

```ts
export interface RunInput {
  modelId: string;
  version: string;
  inputs: Record<string, RawTensor>;
}
```

```ts
export interface RunResult {
  outputs: Record<string, RawTensor>;
  latencyMs: number;
}
```

Leave `RawTensor`, `LoadModelResult`, `SessionOptionsInput`, `ReleaseModelInput`, and `CapacitorOnnxPlugin` unchanged. The map key is the exact ONNX input/output name.

- [ ] **Step 2: Create `src/web/tensor.ts`**

```ts
import * as ort from "onnxruntime-web";
import type { RawTensor } from "../definitions";

/**
 * Converts a transport RawTensor into an ort.Tensor. The ort.Tensor constructor
 * does not accept number[] for every dtype: int64 requires BigInt64Array, and
 * bool/uint8 require typed arrays.
 */
export function toOrtTensor(raw: RawTensor): ort.Tensor {
  switch (raw.type) {
    case "int64":
      return new ort.Tensor("int64", BigInt64Array.from(raw.data, BigInt), raw.dims);
    case "uint32":
      return new ort.Tensor("uint32", Uint32Array.from(raw.data), raw.dims);
    case "int32":
      return new ort.Tensor("int32", Int32Array.from(raw.data), raw.dims);
    case "uint8":
      return new ort.Tensor("uint8", Uint8Array.from(raw.data), raw.dims);
    case "bool":
      return new ort.Tensor("bool", Uint8Array.from(raw.data, (v) => (v ? 1 : 0)), raw.dims);
    case "float16":
      return new ort.Tensor("float16", Uint16Array.from(raw.data), raw.dims);
    default:
      return new ort.Tensor("float32", Float32Array.from(raw.data), raw.dims);
  }
}

/** Converts an ort.Tensor result into a transport RawTensor. Number() handles int64 (bigint). */
export function fromOrtTensor(tensor: ort.Tensor): RawTensor {
  return {
    type: tensor.type as RawTensor["type"],
    dims: tensor.dims,
    data: Array.from(tensor.data as ArrayLike<number | bigint>, Number),
  };
}
```

- [ ] **Step 3: Update imports and `run()` in `src/web/index.ts`**

Add to the existing import block (after the `./provider-resolver` import):

```ts
import { fromOrtTensor, toOrtTensor } from "./tensor";
```

Replace the entire `run()` method (lines 76-115) with:

```ts
  async run(_input: RunInput): Promise<RunResult> {
    const startTime = nowMs();
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const session = this.sessions.get(modelKey);

    if (!session) {
      throw new CapacitorOnnxError(
        "SESSION_INIT_ERROR",
        "ONNX Runtime session is not initialized. Please load a model before running inference.",
      );
    }

    const inputNames = Object.keys(_input.inputs ?? {});
    if (inputNames.length === 0) {
      throw new CapacitorOnnxError(
        "INFERENCE_ERROR",
        "run() requires at least one input tensor in `inputs`.",
      );
    }

    const feeds: Record<string, ort.Tensor> = {};
    for (const name of inputNames) {
      if (!session.inputNames.includes(name)) {
        throw new CapacitorOnnxError(
          "INFERENCE_ERROR",
          `Model has no input named '${name}'.`,
        );
      }
      feeds[name] = toOrtTensor(_input.inputs[name]);
    }

    const inferenceResult = await session.run(feeds);

    const outputs: Record<string, RawTensor> = {};
    for (const name of session.outputNames) {
      const tensor = inferenceResult[name];
      if (tensor) {
        outputs[name] = fromOrtTensor(tensor);
      }
    }

    if (Object.keys(outputs).length === 0) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model produced no outputs.");
    }

    return {
      outputs,
      latencyMs: elapsedMs(startTime),
    };
  }
```

- [ ] **Step 4: Update warmup in `src/web/index.ts`**

In `loadModel()`, replace the warmup block (lines 61-65):

```ts
    if (_input.warmupInputs && Object.keys(_input.warmupInputs).length > 0) {
      const warmupStartTime = nowMs();
      warmed = await CapacitorOnnxWeb.warmupSession(session, _input.warmupInputs);
      warmupLatencyMs = elapsedMs(warmupStartTime);
    }
```

Replace the entire `warmupSession()` method (lines 126-146) with:

```ts
  private static async warmupSession(
    session: ort.InferenceSession,
    warmupInputs: Record<string, RawTensor>,
  ): Promise<boolean> {
    const names = Object.keys(warmupInputs);
    if (names.length === 0) {
      return false;
    }

    try {
      const feeds: Record<string, ort.Tensor> = {};
      for (const name of names) {
        feeds[name] = toOrtTensor(warmupInputs[name]);
      }
      await session.run(feeds);
      return true;
    } catch (err) {
      console.warn(
        "[CapacitorOnnxWeb] Warmup inference failed; continuing without warmup.",
        err,
      );
      return false;
    }
  }
```

- [ ] **Step 5: Verify TypeScript compiles**

Run: `pnpm typecheck`
Expected: no errors (exit 0). If `examples/host-app` is type-checked by this command it will fail — `pnpm typecheck` runs `tsc --noEmit` on the plugin `src/` only, so it should pass.

Run: `pnpm build`
Expected: compiles to `dist/`, exit 0.

---

## Task 2: Update the example host-app smoke runner

The example calls the old contract and will not compile after Task 1. This task restores a green example build.

**Files:**
- Modify: `examples/host-app/src/main.ts`

- [ ] **Step 1: Add an "Input name" config field to the form**

In the `app.innerHTML` template, insert this `<label>` immediately after the Version label block (after the closing `</label>` that follows `<input id="input-version" .../>`):

```html
        <label>
          Input name
          <input id="input-tensor-name" value="input" />
        </label>
```

- [ ] **Step 2: Read the new field in `getFormFields()`**

In `getFormFields()`, add after the `versionInput` lookup:

```ts
  const inputNameInput = document.querySelector<HTMLInputElement>("#input-tensor-name");
```

Add `!inputNameInput ||` to the `if (...)` guard condition, and add `inputNameInput,` to the returned object.

- [ ] **Step 3: Add `inputName` to `RunInferenceConfig` and `getRunConfigFromForm()`**

Change the `RunInferenceConfig` type (lines 20-24) to:

```ts
type RunInferenceConfig = {
  modelId: string;
  version: string;
  inputName: string;
  normalizedData: number[];
};
```

In `getRunConfigFromForm()`, change the destructuring and return:

```ts
function getRunConfigFromForm(): RunInferenceConfig {
  const { modelIdInput, versionInput, inputNameInput, normalizedInput } = getFormFields();

  const modelId = modelIdInput.value.trim();
  const version = versionInput.value.trim();
  const inputName = inputNameInput.value.trim() || "input";
  const normalizedData = parseTensorData(normalizedInput.value);
  validateNormalizedAudioLength(normalizedData);

  if (!modelId || !version) {
    throw new Error("modelId and version are required to run inference");
  }

  return {
    modelId,
    version,
    inputName,
    normalizedData,
  };
}
```

- [ ] **Step 4: Update the success-e2e handler**

In the `successE2EButton` click handler, replace the `CapacitorOnnx.run(...)` call and the assertions/`writeOutput` block (lines 553-589) with:

```ts
    const inference: RunResult = await CapacitorOnnx.run({
      modelId: config.modelId,
      version: config.version,
      inputs: {
        [config.inputName]: {
          data: config.normalizedData,
          dims: [1, config.normalizedData.length],
          type: "float32",
        },
      },
    });

    const outputEntries = Object.entries(inference.outputs);
    const firstOutput = outputEntries[0]?.[1];

    const assertions: AssertionResult[] = [
      assertion(
        "runInference returns at least one output",
        outputEntries.length > 0,
        outputEntries.map(([name]) => name),
      ),
      assertion(
        "runInference.latencyMs is numeric",
        Number.isFinite(inference.latencyMs),
        inference.latencyMs,
      ),
    ];

    const allPassed = assertions.every((item) => item.ok);

    writeOutput({
      platform: Capacitor.getPlatform(),
      operation: "success-e2e",
      passed: allPassed,
      durationMs: Date.now() - startedAt,
      assertions,
      inferenceSummary: {
        outputNames: outputEntries.map(([name]) => name),
        firstOutputType: firstOutput?.type,
        firstOutputDims: firstOutput?.dims,
        firstOutputLength: firstOutput?.data.length,
      },
    });
```

- [ ] **Step 5: Update the error-e2e handler**

In the `errorE2EButton` click handler, replace the `CapacitorOnnx.run(...)` call (lines 638-646) with:

```ts
    await CapacitorOnnx.run({
      modelId: "missing-model",
      version: "0.0.0",
      inputs: {
        input: {
          data: [0, 0, 0, 0],
          dims: [1, 4],
          type: "float32",
        },
      },
    });
```

- [ ] **Step 6: Verify the example builds**

Run: `pnpm build` (from repo root — refreshes `dist/` so the example resolves the new contract)
Then run: `cd examples/host-app && pnpm install && pnpm build`
Expected: Vite + `tsc` build succeeds, exit 0. No references to `inputTensor` or `.logits` remain — confirm with `grep -n "inputTensor\|\.logits" examples/host-app/src/main.ts` returning nothing.

---

## Task 3: Android native implementation

**Files:**
- Modify: `android/src/main/java/com/cantoo/capacitor/onnx/InferenceService.kt`
- Modify: `android/src/main/java/com/cantoo/capacitor/onnx/CapacitorOnnxPlugin.kt`

- [ ] **Step 1: Update imports and `RawTensorInternal` in `InferenceService.kt`**

Replace the import block (lines 3-10) with:

```kotlin
import ai.onnxruntime.NodeInfo
import ai.onnxruntime.OnnxJavaType
import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.TensorInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.nio.ByteBuffer
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.concurrent.ConcurrentHashMap
```

Replace the `RawTensorInternal` data class (lines 12-16) with:

```kotlin
data class RawTensorInternal(
    val data: DoubleArray,
    val dims: LongArray,
    val type: String,
)
```

- [ ] **Step 2: Replace `warmup()` and `run()` in `InferenceService.kt`**

Replace both the `warmup()` (lines 33-59) and `run()` (lines 61-126) methods with:

```kotlin
    suspend fun warmup(
        modelId: String,
        version: String,
        warmupInputs: Map<String, RawTensorInternal>,
    ) = withContext(Dispatchers.Default) {
        val sessionRef = sessionManager.getSession(modelId, version)
            ?: throw IllegalStateException("SESSION_INIT_ERROR: model not loaded, call loadModel first")
        if (warmupInputs.isEmpty()) return@withContext

        val lock = perSessionLock.computeIfAbsent("$modelId::$version") { Mutex() }
        lock.withLock {
            val env = sessionManager.ortEnvironment()
            val tensors = HashMap<String, OnnxTensor>()
            try {
                for ((name, raw) in warmupInputs) {
                    tensors[name] = buildOnnxTensor(env, name, raw)
                }
                sessionRef.session.run(tensors).use { /* discard output */ }
            } catch (_: Throwable) {
                // best-effort warmup; surface failures via subsequent run() calls
            } finally {
                tensors.values.forEach { it.close() }
            }
        }
    }

    suspend fun run(
        modelId: String,
        version: String,
        inputs: Map<String, RawTensorInternal>,
    ): Map<String, RawTensorInternal> = withContext(Dispatchers.Default) {
        val sessionRef = sessionManager.getSession(modelId, version)
            ?: throw IllegalStateException("SESSION_INIT_ERROR: model not loaded, call loadModel first")
        val lock = perSessionLock.computeIfAbsent("$modelId::$version") { Mutex() }

        lock.withLock {
            if (inputs.isEmpty()) {
                throw IllegalStateException("INFERENCE_ERROR: run() requires at least one input tensor")
            }

            val session = sessionRef.session
            val inputInfo = session.inputInfo
            val env = sessionManager.ortEnvironment()
            val tensors = HashMap<String, OnnxTensor>()
            try {
                for ((name, raw) in inputs) {
                    val info = inputInfo[name]
                        ?: throw IllegalStateException("INFERENCE_ERROR: model has no input named '$name'")
                    validateTensor(name, raw, info)
                    tensors[name] = buildOnnxTensor(env, name, raw)
                }

                session.run(tensors).use { result ->
                    val outputs = LinkedHashMap<String, RawTensorInternal>()
                    for (entry in result) {
                        val tensor = entry.value as? OnnxTensor ?: continue
                        outputs[entry.key] = readOnnxTensor(tensor)
                    }
                    if (outputs.isEmpty()) {
                        throw IllegalStateException("MODEL_INVALID: model produced no outputs")
                    }
                    return@withLock outputs
                }
            } finally {
                tensors.values.forEach { it.close() }
            }
        }
    }

    private fun validateTensor(name: String, raw: RawTensorInternal, info: NodeInfo) {
        if (raw.data.isEmpty()) {
            throw IllegalStateException("INFERENCE_ERROR: input '$name' data is empty")
        }
        if (raw.dims.isEmpty()) {
            throw IllegalStateException("INFERENCE_ERROR: input '$name' dims is empty")
        }
        var elementCount = 1L
        for (dim in raw.dims) {
            if (dim <= 0L) {
                throw IllegalStateException("INFERENCE_ERROR: input '$name' dims must have positive dimensions")
            }
            elementCount *= dim
        }
        if (elementCount != raw.data.size.toLong()) {
            throw IllegalStateException("INFERENCE_ERROR: input '$name' data size does not match dims")
        }
        val tensorInfo = info.info as? TensorInfo
            ?: throw IllegalStateException("MODEL_INVALID: model input '$name' is not a tensor")
        val modelShape = tensorInfo.shape
        if (modelShape.size != raw.dims.size) {
            throw IllegalStateException("INFERENCE_ERROR: input '$name' rank does not match model input rank")
        }
        for (i in modelShape.indices) {
            if (modelShape[i] > 0L && modelShape[i] != raw.dims[i]) {
                throw IllegalStateException("INFERENCE_ERROR: input '$name' dims are incompatible with model input")
            }
        }
    }

    private fun buildOnnxTensor(
        env: OrtEnvironment,
        name: String,
        raw: RawTensorInternal,
    ): OnnxTensor {
        return when (raw.type) {
            "float32" -> {
                val arr = FloatArray(raw.data.size) { raw.data[it].toFloat() }
                OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), raw.dims)
            }
            "int32" -> {
                val arr = IntArray(raw.data.size) { raw.data[it].toInt() }
                OnnxTensor.createTensor(env, IntBuffer.wrap(arr), raw.dims)
            }
            "int64" -> {
                val arr = LongArray(raw.data.size) { raw.data[it].toLong() }
                OnnxTensor.createTensor(env, LongBuffer.wrap(arr), raw.dims)
            }
            "uint8" -> {
                val arr = ByteArray(raw.data.size) { raw.data[it].toInt().toByte() }
                OnnxTensor.createTensor(env, ByteBuffer.wrap(arr), raw.dims, OnnxJavaType.UINT8)
            }
            "bool" -> {
                val arr = ByteArray(raw.data.size) { if (raw.data[it] != 0.0) 1 else 0 }
                OnnxTensor.createTensor(env, ByteBuffer.wrap(arr), raw.dims, OnnxJavaType.BOOL)
            }
            else -> throw IllegalStateException(
                "INFERENCE_ERROR: unsupported tensor type '${raw.type}' for input '$name'",
            )
        }
    }

    private fun readOnnxTensor(tensor: OnnxTensor): RawTensorInternal {
        val info = tensor.info
        val type = when (info.type) {
            OnnxJavaType.FLOAT -> "float32"
            OnnxJavaType.INT32 -> "int32"
            OnnxJavaType.INT64 -> "int64"
            OnnxJavaType.UINT8 -> "uint8"
            OnnxJavaType.BOOL -> "bool"
            else -> throw IllegalStateException(
                "INFERENCE_ERROR: unsupported output tensor type ${info.type}",
            )
        }
        return RawTensorInternal(
            data = flattenToDoubleArray(tensor.value),
            dims = info.shape,
            type = type,
        )
    }
```

- [ ] **Step 3: Replace `flattenToFloatArray` with `flattenToDoubleArray` in `InferenceService.kt`**

Replace the `flattenToFloatArray` method (lines 128-149) with:

```kotlin
    private fun flattenToDoubleArray(value: Any?): DoubleArray {
        val output = ArrayList<Double>(1024)

        fun visit(node: Any?) {
            when (node) {
                null -> Unit
                is Float -> output.add(node.toDouble())
                is Double -> output.add(node)
                is Int -> output.add(node.toDouble())
                is Long -> output.add(node.toDouble())
                is Byte -> output.add(node.toDouble())
                is Boolean -> output.add(if (node) 1.0 else 0.0)
                is FloatArray -> node.forEach { output.add(it.toDouble()) }
                is DoubleArray -> node.forEach { output.add(it) }
                is IntArray -> node.forEach { output.add(it.toDouble()) }
                is LongArray -> node.forEach { output.add(it.toDouble()) }
                is ByteArray -> node.forEach { output.add(it.toDouble()) }
                is BooleanArray -> node.forEach { output.add(if (it) 1.0 else 0.0) }
                is Array<*> -> node.forEach { visit(it) }
                else -> throw IllegalStateException(
                    "INFERENCE_ERROR: unsupported output type ${node::class.java.name}",
                )
            }
        }

        visit(value)
        return output.toDoubleArray()
    }
```

- [ ] **Step 4: Update `CapacitorOnnxPlugin.kt` imports and `loadModel()` warmup parsing**

Add to the import block (after `import com.getcapacitor.PluginMethod`):

```kotlin
import org.json.JSONObject
```

In `loadModel()`, replace the `warmupInputJson` line (line 52):

```kotlin
        val warmupInputsJson = call.getObject("warmupInputs")
```

Replace the `parseWarmupInput` usage block (lines 109-113) with:

```kotlin
        val warmupInputs = if (warmupInputsJson != null) parseTensorMap(warmupInputsJson) else null
        if (warmupInputsJson != null && (warmupInputs == null || warmupInputs.isEmpty())) {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid warmupInputs: each entry must include numeric data and positive dims")
            return
        }
```

Inside the `pluginScope.launch` block of `loadModel()`, replace the warmup section (lines 124-130) with:

```kotlin
                val warmupLatencyMs = if (warmupInputs != null && warmupInputs.isNotEmpty()) {
                    val warmupStartMs = System.currentTimeMillis()
                    inferenceService.warmup(modelId, version, warmupInputs)
                    System.currentTimeMillis() - warmupStartMs
                } else {
                    null
                }
```

And in the same block change the `warmed` line (line 135):

```kotlin
                    put("warmed", warmupInputs != null && warmupInputs.isNotEmpty())
```

- [ ] **Step 5: Replace `parseWarmupInput` with map helpers in `CapacitorOnnxPlugin.kt`**

Replace the entire `parseWarmupInput` method (lines 146-168) with:

```kotlin
    private fun parseTensorMap(json: JSObject): Map<String, RawTensorInternal>? {
        val result = LinkedHashMap<String, RawTensorInternal>()
        val keys = json.keys()
        while (keys.hasNext()) {
            val name = keys.next()
            val tensorJson = json.optJSONObject(name) ?: return null
            val tensor = parseTensor(tensorJson) ?: return null
            result[name] = tensor
        }
        return result
    }

    private fun parseTensor(json: JSONObject): RawTensorInternal? {
        val type = json.optString("type", "")
        if (type.isBlank()) return null
        val dataJson = json.optJSONArray("data") ?: return null
        val dimsJson = json.optJSONArray("dims") ?: return null

        val data = DoubleArray(dataJson.length()) { i -> dataJson.optDouble(i, Double.NaN) }
        if (data.any { it.isNaN() }) return null

        val dims = LongArray(dimsJson.length()) { i -> dimsJson.optLong(i, -1L) }
        if (dims.isEmpty() || dims.any { it <= 0L }) return null

        var elementCount = 1L
        for (d in dims) elementCount *= d
        if (elementCount != data.size.toLong()) return null

        return RawTensorInternal(data = data, dims = dims, type = type)
    }

    private fun tensorToJson(tensor: RawTensorInternal): JSObject {
        val dataArray = JSArray()
        tensor.data.forEach { dataArray.put(it) }
        val dimsArray = JSArray()
        tensor.dims.forEach { dimsArray.put(it.toDouble()) }
        return JSObject().apply {
            put("data", dataArray)
            put("dims", dimsArray)
            put("type", tensor.type)
        }
    }
```

- [ ] **Step 6: Replace `run()` in `CapacitorOnnxPlugin.kt`**

Replace the entire `run()` method (lines 170-237) with:

```kotlin
    @PluginMethod
    fun run(call: PluginCall) {
        val modelId = call.getString("modelId")
        val version = call.getString("version")
        val inputsJson = call.getObject("inputs")

        if (modelId == null || version == null || inputsJson == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required fields: modelId, version, inputs")
            return
        }

        val inputs = parseTensorMap(inputsJson)
        if (inputs == null || inputs.isEmpty()) {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid inputs: each entry must include numeric data and positive dims")
            return
        }

        pluginScope.launch {
            try {
                val startMs = System.currentTimeMillis()
                val predictions = inferenceService.run(
                    modelId = modelId,
                    version = version,
                    inputs = inputs,
                )

                val outputsJson = JSObject()
                for ((name, tensor) in predictions) {
                    outputsJson.put(name, tensorToJson(tensor))
                }

                val result = JSObject().apply {
                    put("outputs", outputsJson)
                    put("latencyMs", System.currentTimeMillis() - startMs)
                }
                call.resolve(result)
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }
```

- [ ] **Step 7: Verify Android compiles**

Run: `pnpm build` (from repo root)
Then run: `cd examples/host-app && pnpm cap:sync && pnpm android:assemble`
Expected: Gradle assembles the debug APK, exit 0. This compiles the modified Kotlin (`InferenceService.kt`, `CapacitorOnnxPlugin.kt`) against `onnxruntime-android`.

If the `OnnxTensor.createTensor(env, ByteBuffer, long[], OnnxJavaType.BOOL)` overload is rejected by the pinned ORT version, that is the only at-risk line — confirm against the `onnxruntime-android` 1.24.x API and report back before continuing.

---

## Task 4: iOS native implementation

**Files:**
- Modify: `ios/Plugin/InferenceService.swift`
- Modify: `ios/Plugin/CapacitorOnnxPlugin.swift`

- [ ] **Step 1: Update `RawTensorInternal` in `InferenceService.swift`**

Replace the `RawTensorInternal` struct (lines 35-39) with:

```swift
struct RawTensorInternal {
    let data: [Double]
    let dims: [Int64]
    let type: String
}
```

- [ ] **Step 2: Replace `warmup()`, `run()`, and `runWarmup()` in `InferenceService.swift`**

Replace the `warmup()` (lines 65-70), `run()` (lines 72-137), and `runWarmup()` (lines 139-158) methods with:

```swift
    func warmup(modelId: String, version: String, warmupInputs: [String: RawTensorInternal]) {
        guard let sessionRef = sessionManager.getSession(modelId: modelId, version: version) else {
            return
        }
        guard !warmupInputs.isEmpty else { return }
        do {
            var feeds: [String: ORTValue] = [:]
            for (name, raw) in warmupInputs {
                feeds[name] = try buildOrtValue(name: name, raw: raw)
            }
            let outputNames = Set(try sessionRef.session.outputNames())
            _ = try sessionRef.session.run(withInputs: feeds, outputNames: outputNames, runOptions: nil)
        } catch {
            // best-effort warmup; surface failures via subsequent run() calls
        }
    }

    func run(
        modelId: String,
        version: String,
        inputs: [String: RawTensorInternal]
    ) throws -> [String: RawTensorInternal] {
        guard !inputs.isEmpty else {
            throw OnnxPluginError.inferenceError("run() requires at least one input tensor")
        }
        guard let sessionRef = sessionManager.getSession(modelId: modelId, version: version) else {
            throw OnnxPluginError.sessionInitError("model not loaded, call loadModel first")
        }

        let lock = sessionLock(for: modelId, version: version)
        lock.lock()
        defer { lock.unlock() }

        let session = sessionRef.session
        let modelInputNames = Set(try session.inputNames())
        let outputNames = try session.outputNames()
        guard !outputNames.isEmpty else {
            throw OnnxPluginError.modelInvalid("model has no outputs")
        }

        var feeds: [String: ORTValue] = [:]
        for (name, raw) in inputs {
            guard modelInputNames.contains(name) else {
                throw OnnxPluginError.inferenceError("model has no input named '\(name)'")
            }
            try validateTensor(name: name, raw: raw)
            feeds[name] = try buildOrtValue(name: name, raw: raw)
        }

        let runOutputs = try session.run(
            withInputs: feeds,
            outputNames: Set(outputNames),
            runOptions: nil
        )

        var result: [String: RawTensorInternal] = [:]
        for name in outputNames {
            guard let value = runOutputs[name] else { continue }
            result[name] = try readOrtValue(value)
        }
        guard !result.isEmpty else {
            throw OnnxPluginError.modelInvalid("model produced no output")
        }
        return result
    }

    private func validateTensor(name: String, raw: RawTensorInternal) throws {
        guard !raw.data.isEmpty else {
            throw OnnxPluginError.inferenceError("input '\(name)' data is empty")
        }
        guard !raw.dims.isEmpty else {
            throw OnnxPluginError.inferenceError("input '\(name)' dims is empty")
        }
        guard raw.dims.allSatisfy({ $0 > 0 }) else {
            throw OnnxPluginError.inferenceError("input '\(name)' dims must have positive dimensions")
        }
        let elementCount = raw.dims.reduce(1, *)
        guard elementCount == Int64(raw.data.count) else {
            throw OnnxPluginError.inferenceError("input '\(name)' data size does not match dims")
        }
    }

    private func buildOrtValue(name: String, raw: RawTensorInternal) throws -> ORTValue {
        let shape = raw.dims.map { NSNumber(value: $0) }
        let data = NSMutableData()
        let elementType: ORTTensorElementDataType

        switch raw.type {
        case "float32":
            var values = raw.data.map { Float($0) }
            data.append(&values, length: values.count * MemoryLayout<Float>.stride)
            elementType = .float
        case "int32":
            var values = raw.data.map { Int32($0) }
            data.append(&values, length: values.count * MemoryLayout<Int32>.stride)
            elementType = .int32
        case "int64":
            var values = raw.data.map { Int64($0) }
            data.append(&values, length: values.count * MemoryLayout<Int64>.stride)
            elementType = .int64
        case "uint8":
            var values = raw.data.map { UInt8($0) }
            data.append(&values, length: values.count * MemoryLayout<UInt8>.stride)
            elementType = .uInt8
        case "bool":
            var values = raw.data.map { $0 != 0.0 }
            data.append(&values, length: values.count * MemoryLayout<Bool>.stride)
            elementType = .bool
        default:
            throw OnnxPluginError.inferenceError(
                "unsupported tensor type '\(raw.type)' for input '\(name)'"
            )
        }

        return try ORTValue(tensorData: data, elementType: elementType, shape: shape)
    }

    private func readOrtValue(_ value: ORTValue) throws -> RawTensorInternal {
        let info = try value.tensorTypeAndShapeInfo()
        let dims = info.shape.map { $0.int64Value }
        let raw = try value.tensorData() as Data

        let data: [Double]
        let type: String
        switch info.elementType {
        case .float:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }.map { Double($0) }
            type = "float32"
        case .int32:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }.map { Double($0) }
            type = "int32"
        case .int64:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }.map { Double($0) }
            type = "int64"
        case .uInt8:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }.map { Double($0) }
            type = "uint8"
        case .bool:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Bool.self)) }.map { $0 ? 1.0 : 0.0 }
            type = "bool"
        default:
            throw OnnxPluginError.inferenceError("unsupported output tensor type")
        }

        return RawTensorInternal(data: data, dims: dims, type: type)
    }
```

- [ ] **Step 3: Update `run()` in `CapacitorOnnxPlugin.swift`**

Replace the entire `run()` method (lines 112-158) with:

```swift
    @objc func run(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version"),
              let inputsObj = call.getObject("inputs") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version, inputs")
            return
        }

        guard let inputs = Self.parseTensorMap(inputsObj), !inputs.isEmpty else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid inputs: each entry must include numeric data and positive dims")
            return
        }

        Task {
            do {
                let startMs = currentTimeMs()
                let result = try inferenceService.run(
                    modelId: modelId, version: version, inputs: inputs
                )
                var outputs: [String: Any] = [:]
                for (name, tensor) in result {
                    outputs[name] = Self.tensorToDict(tensor)
                }
                call.resolve([
                    "outputs": outputs,
                    "latencyMs": currentTimeMs() - startMs,
                ])
            } catch {
                rejectStructured(call, error: error)
            }
        }
    }
```

- [ ] **Step 4: Update `loadModel()` warmup in `CapacitorOnnxPlugin.swift`**

In `loadModel()`, replace the `warmupInputJson` line (line 46):

```swift
        let warmupInputsJson = call.getObject("warmupInputs")
```

Replace the warmup parsing block (lines 68-77) with:

```swift
        let warmupInputs: [String: RawTensorInternal]?
        if let warmupInputsJson = warmupInputsJson {
            guard let parsed = Self.parseTensorMap(warmupInputsJson), !parsed.isEmpty else {
                rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid warmupInputs: each entry must include numeric data and positive dims")
                return
            }
            warmupInputs = parsed
        } else {
            warmupInputs = nil
        }
```

Inside the `Task` block of `loadModel()`, replace the warmup section (lines 89-96) with:

```swift
                var warmupLatencyMs: Double? = nil
                var warmed = false
                if let warmupInputs = warmupInputs {
                    let warmupStart = currentTimeMs()
                    inferenceService.warmup(modelId: modelId, version: version, warmupInputs: warmupInputs)
                    warmupLatencyMs = currentTimeMs() - warmupStart
                    warmed = true
                }
```

- [ ] **Step 5: Replace `parseWarmupInput` with map helpers in `CapacitorOnnxPlugin.swift`**

Replace the entire `parseWarmupInput` static method (lines 203-217) with:

```swift
    private static func parseTensorMap(_ obj: [String: Any]) -> [String: RawTensorInternal]? {
        var result: [String: RawTensorInternal] = [:]
        for (name, value) in obj {
            guard let tensorObj = value as? [String: Any],
                  let tensor = parseTensor(tensorObj) else {
                return nil
            }
            result[name] = tensor
        }
        return result
    }

    private static func parseTensor(_ json: [String: Any]) -> RawTensorInternal? {
        guard let type = json["type"] as? String, !type.isEmpty else { return nil }
        guard let dataRaw = json["data"] as? [Any],
              let dimsRaw = json["dims"] as? [Any] else { return nil }

        let data = dataRaw.compactMap { ($0 as? NSNumber)?.doubleValue }
        guard data.count == dataRaw.count else { return nil }

        let dims = dimsRaw.compactMap { ($0 as? NSNumber)?.int64Value }
        guard dims.count == dimsRaw.count, !dims.isEmpty, dims.allSatisfy({ $0 > 0 }) else { return nil }

        let elementCount = dims.reduce(1, *)
        guard elementCount == Int64(data.count) else { return nil }

        return RawTensorInternal(data: data, dims: dims, type: type)
    }

    private static func tensorToDict(_ tensor: RawTensorInternal) -> [String: Any] {
        return [
            "data": tensor.data,
            "dims": tensor.dims.map { NSNumber(value: $0) },
            "type": tensor.type,
        ]
    }
```

- [ ] **Step 6: Verify iOS compiles**

iOS has no generated project in `examples/host-app`. Verify by compiling in a Capacitor iOS host:

Run: `pnpm build` (repo root), then `cd examples/host-app && npx cap add ios && npx cap sync ios`
Then build: `xcodebuild -workspace examples/host-app/ios/App/App.xcworkspace -scheme App -sdk iphonesimulator -configuration Debug build` (requires macOS + Xcode + CocoaPods).
Expected: build succeeds, exit 0. This compiles `InferenceService.swift` and `CapacitorOnnxPlugin.swift` against `onnxruntime-objc`.

The at-risk lines are the `ORTTensorElementDataType` cases `.uInt8` and `.bool`. If the compiler reports either case does not exist on the pinned `onnxruntime-objc` 1.24.x, stop and report back — do not silently drop a type.

---

## Task 5: Documentation

**Files:**
- Modify: `README.md`
- Modify: `CLAUDE.md`
- Modify: `docs/testing-scripts.md`

- [ ] **Step 1: Update the `README.md` methods table**

Replace the `loadModel` and `run` rows (lines 122-123) with:

```markdown
| `loadModel` | `(input: LoadModelInput) => Promise<LoadModelResult>` | Creates an ONNX Runtime session from the model bytes (web) or file path (native), and optionally warms it up. Must be called once per `modelId+version` before `run`. | Native: pass `filePath` (absolute path or `file://` URI). Web: pass `modelBuffer: Uint8Array`. Pass `warmupInputs` (a `Record<string, RawTensor>` keyed by model input name) to pay first-inference cost upfront, and `sessionOptions` to pick the execution provider / thread counts. The result includes `executionProviderUsed`. |
| `run` | `(input: RunInput) => Promise<RunResult>` | Runs inference on a previously loaded session. | Pass `inputs` as a `Record<string, RawTensor>` keyed by the model's ONNX input names. Calls to the same `modelId+version` are serialized by a per-session lock; different models run in parallel. Returns `{ outputs, latencyMs }`, where `outputs` is keyed by the model's output names. Pre/post-processing is the consumer's responsibility. |
```

- [ ] **Step 2: Update the `README.md` example**

Replace the `run`/`logits` block (lines 157-167) with:

```ts
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
```

- [ ] **Step 3: Update the `README.md` Runtime Notes**

Replace the warmup note (line 174) with:

```markdown
- `loadModel` supports optional `warmupInputs: Record<string, RawTensor>` to pre-run the session with sample tensors keyed by model input name (e.g. `{ input_values: { type: 'float32', dims: [1, 16000], data: [...] } }`). Warmup is skipped when `warmupInputs` is omitted.
```

Replace the `run` note (line 179) with:

```markdown
- `run` takes `inputs` keyed by ONNX input name and returns every model output in `outputs` keyed by ONNX output name. Native (Android/iOS) accepts `float32`, `int64`, `int32`, `bool`, and `uint8` tensors; `float16`/`uint32` are web-only and rejected on native with a structured error.
```

Replace the output-shape note (line 180) with:

```markdown
- **Output shape & dtype**: each `RunResult.outputs` tensor carries the shape and dtype ORT materialized — Web reads `ort.Tensor.dims`/`.type`, Android reads `OnnxTensor.info.shape`/`.type`, iOS reads `tensorTypeAndShapeInfo().shape`/`.elementType`. No heuristic, no symbolic dims (`-1`) in the result.
```

- [ ] **Step 4: Update `CLAUDE.md`**

In the "Key design decisions" section, replace the first bullet:

```markdown
- The plugin contract is intentionally minimal: `run()` receives named `inputs` (a `Record<string, RawTensor>` keyed by ONNX input name) and returns named `outputs` (keyed by ONNX output name). Pre/post-processing lives in the consumer app.
```

Replace the output-shape bullet:

```markdown
- Output shape and dtype come straight from ORT on every platform: Web reads each `ort.Tensor`'s `dims`/`type`, Android reads `OnnxTensor.info` (`shape`/`type`), iOS reads `tensorTypeAndShapeInfo()` (`shape`/`elementType`). No heuristic shape resolution.
```

In the iOS provider-mapping bullet at the end, no change is required.

- [ ] **Step 5: Update `docs/testing-scripts.md`**

Append a new section before "## Quick troubleshooting":

```markdown
## Multi-input models

`run()` takes `inputs` keyed by ONNX input name. A model with two inputs (e.g. wav2vec2):

\`\`\`ts
const { outputs } = await CapacitorOnnx.run({
  modelId, version,
  inputs: {
    input_values:   { type: 'float32', dims: [1, 16000], data: audioSamples },
    attention_mask: { type: 'int64',   dims: [1, 16000], data: maskValues },
  },
});
\`\`\`

Native accepts `float32`, `int64`, `int32`, `bool`, `uint8`. The exact input/output
names must match the model — inspect the `.onnx` file (e.g. with Netron) if unsure.
```

(Replace the `\`\`\`` fences above with real triple backticks when writing the file.)

- [ ] **Step 6: Verify docs**

Run: `grep -rn "inputTensor\|warmupInput\b\|\.logits\|RunResult.logits" README.md CLAUDE.md docs/testing-scripts.md`
Expected: no matches (every reference to the old contract has been replaced).

---

## Release (maintainer action — not a task)

After all tasks pass verification, the maintainer publishes `3.0.0`:

```bash
npm version major   # 2.x.x -> 3.0.0; syncs CantooCapacitorOnnx.podspec; creates a commit + tag
pnpm build
pnpm typecheck
pnpm pack --dry-run
npm publish
```

`npm version` creates a commit and tag by design — this is the documented publish flow and is the maintainer's decision, distinct from the no-auto-commit convention for implementation work. Consumer apps then run `pnpm add @cantoo/capacitor-onnx@latest && pnpm cap sync`.

---

## Self-Review

**Spec coverage:**
- Decision 1 (`inputs` map) → Task 1 Step 1, 3; Task 3 Step 6; Task 4 Step 3.
- Decision 2 (native dtypes `float32/int64/int32/bool/uint8`) → Task 3 Step 2 (`buildOnnxTensor`/`readOnnxTensor`); Task 4 Step 2 (`buildOrtValue`/`readOrtValue`).
- Decision 3 (`warmupInputs` map) → Task 1 Step 4; Task 3 Steps 4-5; Task 4 Steps 4-5.
- Decision 4 (`outputs` map + real dtype) → Task 1 Step 3 (`fromOrtTensor`); Task 3 Step 2 (`readOnnxTensor`); Task 4 Step 2 (`readOrtValue`).
- Validation rules (empty inputs, unknown name, bad dims, unsupported type) → Task 1 Step 3; Task 3 Step 2 (`validateTensor` + `buildOnnxTensor` else); Task 4 Step 2 (`validateTensor` + `buildOrtValue` default).
- Docs/versioning → Task 5 + Release section.

**Type consistency:** `RawTensorInternal` is `DoubleArray`/`LongArray` (Kotlin) and `[Double]`/`[Int64]` (Swift) across all native tasks. `inferenceService.run` returns `Map<String, RawTensorInternal>` / `[String: RawTensorInternal]`, consumed by `tensorToJson` / `tensorToDict`. `warmup` is non-throwing on iOS (best-effort) and the plugin call site uses no `try`. Web `toOrtTensor`/`fromOrtTensor` signatures match their call sites in `run`/`warmupSession`.

**Known risk points flagged in-plan:** Android `OnnxJavaType.BOOL` buffer overload (Task 3 Step 7); iOS `ORTTensorElementDataType.uInt8`/`.bool` cases (Task 4 Step 6). Both are isolated to the `bool`/`uint8` paths; the wav2vec2 driver model needs only `float32`+`int64`.
