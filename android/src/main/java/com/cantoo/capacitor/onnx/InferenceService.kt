package com.cantoo.capacitor.onnx

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
import java.nio.ByteOrder
import java.nio.FloatBuffer
import java.nio.IntBuffer
import java.nio.LongBuffer
import java.util.concurrent.ConcurrentHashMap

/**
 * A tensor as it crosses the JS bridge: its elements either as JSON numbers (`data`) or as
 * their little-endian bytes (`bytes`, carried as base64), exactly one of the two.
 */
data class RawTensorInternal(
    val dims: LongArray,
    val type: String,
    val data: DoubleArray? = null,
    val bytes: ByteArray? = null,
) {
    val elementCount: Int
        get() = data?.size ?: ((bytes?.size ?: 0) / elementByteSize(type))
}

/** Bytes per element of a tensor type, as the JS side lays them out (TypedTensorData). */
fun elementByteSize(type: String): Int = when (type) {
    "int64" -> 8
    "float16" -> 2
    "uint8", "bool" -> 1
    else -> 4
}

class InferenceService(
    private val sessionManager: SessionManager,
) {
    private val perSessionLock = ConcurrentHashMap<String, Mutex>()

    suspend fun prepareModel(
        modelId: String,
        version: String,
        filePath: String,
        sessionConfig: SessionConfig,
    ): String = withContext(Dispatchers.IO) {
        val session = sessionManager.ensureSession(modelId, version, filePath, sessionConfig)
        session.executionProviderUsed
    }

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
        typedOutputs: Boolean = false,
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
                        outputs[entry.key] = readOnnxTensor(tensor, typedOutputs)
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
        if (raw.elementCount == 0) {
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
        if (elementCount != raw.elementCount.toLong()) {
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
        val bytes = raw.bytes
        if (bytes != null) return buildOnnxTensorFromBytes(env, name, raw, bytes)
        val data = raw.data
            ?: throw IllegalStateException("INFERENCE_ERROR: input '$name' has no data")
        return when (raw.type) {
            "float32" -> {
                val arr = FloatArray(data.size) { data[it].toFloat() }
                OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), raw.dims)
            }
            "int32" -> {
                val arr = IntArray(data.size) { data[it].toInt() }
                OnnxTensor.createTensor(env, IntBuffer.wrap(arr), raw.dims)
            }
            "int64" -> {
                val arr = LongArray(data.size) { data[it].toLong() }
                OnnxTensor.createTensor(env, LongBuffer.wrap(arr), raw.dims)
            }
            "uint8" -> {
                val arr = ByteArray(data.size) { data[it].toInt().toByte() }
                OnnxTensor.createTensor(env, ByteBuffer.wrap(arr), raw.dims, OnnxJavaType.UINT8)
            }
            "bool" -> {
                val arr = ByteArray(data.size) { if (data[it] != 0.0) 1 else 0 }
                OnnxTensor.createTensor(env, ByteBuffer.wrap(arr), raw.dims, OnnxJavaType.BOOL)
            }
            else -> throw IllegalStateException(
                "INFERENCE_ERROR: unsupported tensor type '${raw.type}' for input '$name'",
            )
        }
    }

    // The bytes come from the JS typed arrays, little-endian (every target platform is).
    private fun buildOnnxTensorFromBytes(
        env: OrtEnvironment,
        name: String,
        raw: RawTensorInternal,
        bytes: ByteArray,
    ): OnnxTensor {
        val count = raw.elementCount
        val littleEndian = ByteBuffer.wrap(bytes).order(ByteOrder.LITTLE_ENDIAN)
        return when (raw.type) {
            "float32" -> {
                val arr = FloatArray(count)
                littleEndian.asFloatBuffer().get(arr)
                OnnxTensor.createTensor(env, FloatBuffer.wrap(arr), raw.dims)
            }
            "int32" -> {
                val arr = IntArray(count)
                littleEndian.asIntBuffer().get(arr)
                OnnxTensor.createTensor(env, IntBuffer.wrap(arr), raw.dims)
            }
            "int64" -> {
                val arr = LongArray(count)
                littleEndian.asLongBuffer().get(arr)
                OnnxTensor.createTensor(env, LongBuffer.wrap(arr), raw.dims)
            }
            "uint8" -> {
                OnnxTensor.createTensor(env, ByteBuffer.wrap(bytes), raw.dims, OnnxJavaType.UINT8)
            }
            "bool" -> {
                val arr = ByteArray(count) { if (bytes[it] != 0.toByte()) 1 else 0 }
                OnnxTensor.createTensor(env, ByteBuffer.wrap(arr), raw.dims, OnnxJavaType.BOOL)
            }
            else -> throw IllegalStateException(
                "INFERENCE_ERROR: unsupported tensor type '${raw.type}' for input '$name'",
            )
        }
    }

    private fun readOnnxTensor(tensor: OnnxTensor, asBytes: Boolean): RawTensorInternal {
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
        if (asBytes) {
            // A copy of the tensor's native (little-endian) bytes.
            val buffer = tensor.byteBuffer
            val bytes = ByteArray(buffer.remaining())
            buffer.get(bytes)
            return RawTensorInternal(dims = info.shape, type = type, bytes = bytes)
        }
        return RawTensorInternal(
            data = flattenToDoubleArray(tensor.value),
            dims = info.shape,
            type = type,
        )
    }

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
}
