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
    val shape: LongArray,
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

    suspend fun warmup(modelId: String, version: String) = withContext(Dispatchers.Default) {
        val modelRef = modelStore.resolve(modelId, version)
        sessionManager.ensureSession(modelRef)
    }

    suspend fun runInference(
        modelId: String,
        version: String,
        inputTensorData: FloatArray,
        inputTensorShape: LongArray,
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
            if (inputTensorShape.isEmpty()) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor shape is empty")
            }

            var elementCount = 1L
            for (dim in inputTensorShape) {
                if (dim <= 0L) {
                    throw IllegalStateException("INFERENCE_ERROR: input tensor shape must have positive dimensions")
                }
                elementCount *= dim
            }
            if (elementCount != inputTensorData.size.toLong()) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor data size does not match shape")
            }

            val input = sessionRef.session.inputInfo.entries.firstOrNull()
                ?: throw IllegalStateException("MODEL_INVALID: model has no inputs")
            val tensorInfo = input.value.info as? TensorInfo
                ?: throw IllegalStateException("MODEL_INVALID: model input is not a tensor")
            val modelInputShape = tensorInfo.shape
            if (modelInputShape.size != inputTensorShape.size) {
                throw IllegalStateException("INFERENCE_ERROR: input tensor rank does not match model input rank")
            }
            for (i in modelInputShape.indices) {
                if (modelInputShape[i] > 0L && modelInputShape[i] != inputTensorShape[i]) {
                    throw IllegalStateException("INFERENCE_ERROR: input tensor shape is incompatible with model input")
                }
            }

            val env = sessionManager.ortEnvironment()
            val tensor = OnnxTensor.createTensor(env, FloatBuffer.wrap(inputTensorData), inputTensorShape)
            val outputShapeHint = resolveOutputShapeHint(sessionRef.session)

            tensor.use { t ->
                sessionRef.session.run(mapOf(input.key to t)).use { result ->
                    val logits = flattenToFloatArray(result[0].value)
                    return@withLock RawTensorInternal(
                        data = logits,
                        shape = resolveOutputShape(logits.size, outputShapeHint),
                        type = "float32",
                    )
                }
            }
        }
    }

    private fun resolveOutputShapeHint(session: ai.onnxruntime.OrtSession): LongArray {
        val output = session.outputInfo.entries.firstOrNull()
            ?: return longArrayOf(-1)
        val info = output.value.info as? TensorInfo
            ?: return longArrayOf(-1)
        return info.shape
    }

    private fun resolveOutputShape(totalValues: Int, shapeHint: LongArray): LongArray {
        if (shapeHint.isEmpty()) {
            return longArrayOf(totalValues.toLong())
        }

        var knownProduct = 1L
        var unknownCount = 0
        for (d in shapeHint) {
            if (d <= 0L) {
                unknownCount += 1
            } else {
                knownProduct *= d
            }
        }

        if (unknownCount == 0 && knownProduct == totalValues.toLong()) {
            return shapeHint
        }

        if (unknownCount == 1 && knownProduct > 0L && totalValues % knownProduct.toInt() == 0) {
            val resolved = shapeHint.copyOf()
            val missing = (totalValues / knownProduct).toLong()
            for (i in resolved.indices) {
                if (resolved[i] <= 0L) {
                    resolved[i] = missing
                    break
                }
            }
            return resolved
        }

        return longArrayOf(1L, totalValues.toLong())
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
