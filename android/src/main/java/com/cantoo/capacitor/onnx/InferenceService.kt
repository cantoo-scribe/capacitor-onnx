package com.cantoo.capacitor.onnx

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.TensorInfo
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.nio.FloatBuffer
import java.util.concurrent.ConcurrentHashMap

data class RawTensorInternal(
    val data: FloatArray,
    val dims: LongArray,
    val type: String,
)

data class PrepareModelRuntimeResultInternal(
    val cacheHit: Boolean,
    val executionProviderUsed: String,
)

class InferenceService(
    private val sessionManager: SessionManager,
    private val modelStore: ModelStore,
) {
    private val perSessionLock = ConcurrentHashMap<String, Mutex>()

    suspend fun prepareModel(
        modelId: String,
        version: String,
        url: String,
        sha256: String?,
        forceRedownload: Boolean,
        sessionConfig: SessionConfig,
    ): PrepareModelRuntimeResultInternal = withContext(Dispatchers.IO) {
        val prepare = modelStore.prepare(modelId, version, url, sha256, forceRedownload)
        val session = sessionManager.ensureSession(prepare.modelRef, sessionConfig)
        PrepareModelRuntimeResultInternal(
            cacheHit = prepare.cacheHit,
            executionProviderUsed = session.executionProviderUsed,
        )
    }

    suspend fun warmup(
        modelId: String,
        version: String,
        warmupInput: RawTensorInternal? = null,
    ) = withContext(Dispatchers.Default) {
        val modelRef = modelStore.resolve(modelId, version)
        val sessionRef = sessionManager.ensureSession(modelRef)

        if (warmupInput == null) {
            return@withContext
        }

        val lock = perSessionLock.computeIfAbsent("$modelId::$version") { Mutex() }
        lock.withLock {
            val input = sessionRef.session.inputInfo.entries.firstOrNull()
                ?: return@withLock
            val env = sessionManager.ortEnvironment()
            try {
                val tensor = OnnxTensor.createTensor(
                    env,
                    FloatBuffer.wrap(warmupInput.data),
                    warmupInput.dims,
                )
                tensor.use { t ->
                    sessionRef.session.run(mapOf(input.key to t)).use { /* discard output */ }
                }
            } catch (_: Throwable) {
                // best-effort warmup; surface failures via subsequent run() calls
            }
        }
    }

    suspend fun run(
        modelId: String,
        version: String,
        inputTensorData: FloatArray,
        inputTensorDims: LongArray,
        inputTensorType: String,
    ): RawTensorInternal = withContext(Dispatchers.Default) {
        val modelRef = modelStore.resolve(modelId, version)
        val lock = perSessionLock.computeIfAbsent("$modelId::$version") { Mutex() }

        lock.withLock {
            val sessionRef = sessionManager.ensureSession(modelRef)
            if (inputTensorType != "float32") {
                throw IllegalStateException("INFERENCE_ERROR: only float32 tensors are supported")
            }
            if (inputTensorData.isEmpty()) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor data is empty")
            }
            if (inputTensorDims.isEmpty()) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor dims is empty")
            }

            var elementCount = 1L
            for (dim in inputTensorDims) {
                if (dim <= 0L) {
                    throw IllegalStateException("INFERENCE_ERROR: input tensor dims must have positive dimensions")
                }
                elementCount *= dim
            }
            if (elementCount != inputTensorData.size.toLong()) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor data size does not match dims")
            }

            val input = sessionRef.session.inputInfo.entries.firstOrNull()
                ?: throw IllegalStateException("MODEL_INVALID: model has no inputs")
            val tensorInfo = input.value.info as? TensorInfo
                ?: throw IllegalStateException("MODEL_INVALID: model input is not a tensor")
            val modelInputShape = tensorInfo.shape
            if (modelInputShape.size != inputTensorDims.size) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor rank does not match model input rank")
            }
            for (i in modelInputShape.indices) {
                if (modelInputShape[i] > 0L && modelInputShape[i] != inputTensorDims[i]) {
                    throw IllegalStateException("INFERENCE_ERROR: input tensor dims are incompatible with model input")
                }
            }

            val env = sessionManager.ortEnvironment()
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensorData), inputTensorDims)

            tensor.use { t ->
                sessionRef.session.run(mapOf(input.key to t)).use { result ->
                    val outputValue = result[0]
                    val outputTensor = outputValue as? OnnxTensor
                        ?: throw IllegalStateException("INFERENCE_ERROR: model output is not a tensor")
                    val dims = outputTensor.info.shape
                    val logits = flattenToFloatArray(outputValue.value)
                    return@withLock RawTensorInternal(
                        data = logits,
                        dims = dims,
                        type = "float32",
                    )
                }
            }
        }
    }

    private fun flattenToFloatArray(value: Any?): FloatArray {
        val output = ArrayList<Float>(1024)

        fun visit(node: Any?) {
            when (node) {
                null -> Unit
                is Float -> output.add(node)
                is Double -> output.add(node.toFloat())
                is Int -> output.add(node.toFloat())
                is Long -> output.add(node.toFloat())
                is FloatArray -> node.forEach { output.add(it) }
                is DoubleArray -> node.forEach { output.add(it.toFloat()) }
                is IntArray -> node.forEach { output.add(it.toFloat()) }
                is LongArray -> node.forEach { output.add(it.toFloat()) }
                is Array<*> -> node.forEach { visit(it) }
                else -> throw IllegalStateException("INFERENCE_ERROR: unsupported output type ${node::class.java.name}")
            }
        }

        visit(value)
        return output.toFloatArray()
    }
}
