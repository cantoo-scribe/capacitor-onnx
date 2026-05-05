package com.cantoo.capacitor.onnx

import com.getcapacitor.JSArray
import com.getcapacitor.JSObject
import com.getcapacitor.Plugin
import com.getcapacitor.PluginCall
import com.getcapacitor.PluginMethod
import com.getcapacitor.annotation.CapacitorPlugin
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.cancel
import kotlinx.coroutines.launch
import java.io.File
import java.util.UUID

@CapacitorPlugin(name = "CapacitorOnnx")
class CapacitorOnnxPlugin : Plugin() {
    private val pluginScope = CoroutineScope(SupervisorJob() + Dispatchers.Main.immediate)

    private lateinit var modelStore: ModelStore
    private lateinit var sessionManager: SessionManager
    private lateinit var inferenceService: InferenceService

    private data class ParsedPluginError(
        val code: String,
        val message: String,
    )

    override fun load() {
        super.load()
        val root = File(context.filesDir, "capacitor_onnx")
        modelStore = ModelStore(context, root)
        sessionManager = SessionManager()
        inferenceService = InferenceService(sessionManager, modelStore)
    }

    override fun handleOnDestroy() {
        pluginScope.cancel()
        sessionManager.closeAll()
        super.handleOnDestroy()
    }

    @PluginMethod
    fun isActive(call: PluginCall) {
        call.resolve(JSObject().apply { put("value", true) })
    }

    @PluginMethod
    fun loadModel(call: PluginCall) {
        val modelId = call.getString("modelId")
        val version = call.getString("version")
        val url = call.getString("url")
        val sha256 = call.getString("sha256")?.trim()?.takeIf { it.isNotEmpty() }
        val forceRedownload = call.getBoolean("forceRedownload", false) ?: false
        val warmup = call.getBoolean("warmup", false) ?: false
        val sessionOptions = call.getObject("sessionOptions")
        val executionProvider = (sessionOptions?.optString("executionProvider", "auto") ?: "auto").lowercase()

        val intraOpNumThreads = if (sessionOptions?.has("intraOpNumThreads") == true) {
            sessionOptions.optInt("intraOpNumThreads", -1)
        } else {
            null
        }
        val interOpNumThreads = if (sessionOptions?.has("interOpNumThreads") == true) {
            sessionOptions.optInt("interOpNumThreads", -1)
        } else {
            null
        }

        if (modelId == null || version == null || url == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required fields: modelId, version, url")
            return
        }

        if (executionProvider != "cpu" && executionProvider != "nnapi" && executionProvider != "auto") {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid sessionOptions.executionProvider: use cpu, nnapi or auto")
            return
        }

        if (intraOpNumThreads != null && intraOpNumThreads <= 0) {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid sessionOptions.intraOpNumThreads: must be > 0")
            return
        }

        if (interOpNumThreads != null && interOpNumThreads <= 0) {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid sessionOptions.interOpNumThreads: must be > 0")
            return
        }

        val sessionConfig = SessionConfig(
            executionProvider = executionProvider,
            intraOpNumThreads = intraOpNumThreads,
            interOpNumThreads = interOpNumThreads,
        )

        pluginScope.launch {
            try {
                val startMs = System.currentTimeMillis()
                val prepare = inferenceService.prepareModel(
                    modelId = modelId,
                    version = version,
                    url = url,
                    sha256 = sha256,
                    forceRedownload = forceRedownload,
                    sessionConfig = sessionConfig,
                )
                val warmupLatencyMs = if (warmup) {
                    val warmupStartMs = System.currentTimeMillis()
                    inferenceService.warmup(modelId, version)
                    System.currentTimeMillis() - warmupStartMs
                } else {
                    null
                }

                val result = JSObject().apply {
                    put("status", if (prepare.cacheHit) "cache_hit" else "downloaded")
                    put("sessionReady", true)
                    put("executionProviderUsed", prepare.executionProviderUsed)
                    put("warmed", warmup)
                    warmupLatencyMs?.let { put("warmupLatencyMs", it) }
                    put("latencyMs", System.currentTimeMillis() - startMs)
                }
                call.resolve(result)
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }

    @PluginMethod
    fun warmupModel(call: PluginCall) {
        val modelId = call.getString("modelId")
        val version = call.getString("version")

        if (modelId == null || version == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required fields: modelId, version")
            return
        }

        pluginScope.launch {
            try {
                val startMs = System.currentTimeMillis()
                inferenceService.warmup(modelId, version)
                val result = JSObject().apply {
                    put("warmed", true)
                    put("latencyMs", System.currentTimeMillis() - startMs)
                }
                call.resolve(result)
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }

    @PluginMethod
    fun run(call: PluginCall) {
        val modelId = call.getString("modelId")
        val version = call.getString("version")
        val inputTensor = call.getObject("inputTensor")

        if (modelId == null || version == null || inputTensor == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required fields: modelId, version, inputTensor")
            return
        }

        val inputType = inputTensor.optString("type", "")
        val inputDataJson = inputTensor.optJSONArray("data")
        val inputDimsJson = inputTensor.optJSONArray("dims")
        if (inputDataJson == null || inputDimsJson == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required inputTensor fields: data, dims")
            return
        }

        val inputData = FloatArray(inputDataJson.length()) { i ->
            inputDataJson.optDouble(i, Double.NaN).toFloat()
        }
        if (inputData.any { it.isNaN() }) {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid inputTensor.data: values must be numeric")
            return
        }

        val inputDims = LongArray(inputDimsJson.length()) { i ->
            inputDimsJson.optLong(i, -1L)
        }
        if (inputDims.any { it <= 0L }) {
            rejectStructured(call, "INFERENCE_ERROR", "Invalid inputTensor.dims: all dimensions must be positive integers")
            return
        }

        pluginScope.launch {
            try {
                val startMs = System.currentTimeMillis()
                val predictions = inferenceService.run(
                    modelId = modelId,
                    version = version,
                    inputTensorData = inputData,
                    inputTensorDims = inputDims,
                    inputTensorType = inputType,
                )

                val result = JSObject().apply {
                    val logitsData = JSArray()
                    predictions.data.forEach { value ->
                        logitsData.put(value.toDouble())
                    }
                    val logitsDims = JSArray()
                    predictions.dims.forEach { dim ->
                        logitsDims.put(dim.toDouble())
                    }
                    put("logits", JSObject().apply {
                        put("data", logitsData)
                        put("dims", logitsDims)
                        put("type", predictions.type)
                    })
                    put("latencyMs", System.currentTimeMillis() - startMs)
                }
                call.resolve(result)
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }

    @PluginMethod
    fun getModelStatus(call: PluginCall) {
        val modelId = call.getString("modelId")
        val version = call.getString("version")

        if (modelId == null || version == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required fields: modelId, version")
            return
        }

        pluginScope.launch {
            try {
                val status = modelStore.getModelStatus(modelId, version)
                val result = JSObject().apply {
                    put("exists", status.exists)
                    put("integrityOk", status.integrityOk)
                    put("sessionLoaded", sessionManager.hasSession(modelId, version))
                    status.sizeBytes?.let { put("sizeBytes", it) }
                }
                call.resolve(result)
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }

    @PluginMethod
    fun clearModel(call: PluginCall) {
        val modelId = call.getString("modelId")
        val version = call.getString("version")

        if (modelId == null || version == null) {
            rejectStructured(call, "INFERENCE_ERROR", "Missing required fields: modelId, version")
            return
        }

        pluginScope.launch {
            try {
                sessionManager.closeSession(modelId, version)
                val removed = modelStore.clearModel(modelId, version)
                call.resolve(JSObject().apply { put("removed", removed) })
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }

    @PluginMethod
    fun clearAllCache(call: PluginCall) {
        pluginScope.launch {
            try {
                sessionManager.closeAll()
                val removedModels = modelStore.clearAll()
                call.resolve(JSObject().apply { put("removedModels", removedModels) })
            } catch (e: Throwable) {
                rejectStructured(call, e)
            }
        }
    }

    @PluginMethod
    fun getDiagnostics(call: PluginCall) {
        val result = JSObject().apply {
            put("activeSessions", sessionManager.activeSessionCount())
            put("cacheEntries", modelStore.cacheEntryCount())
        }
        call.resolve(result)
    }

    private fun rejectStructured(call: PluginCall, error: Throwable) {
        val parsed = parsePluginError(error.message)
        val payload = buildErrorPayload(parsed.code, parsed.message)
        call.reject(parsed.message, parsed.code, error as? Exception, payload)
    }

    private fun rejectStructured(call: PluginCall, code: String, message: String) {
        val payload = buildErrorPayload(code, message)
        call.reject(message, code, null, payload)
    }

    private fun parsePluginError(rawMessage: String?): ParsedPluginError {
        if (rawMessage.isNullOrBlank()) {
            return ParsedPluginError("INTERNAL_ERROR", "Unexpected internal error")
        }

        val separatorIndex = rawMessage.indexOf(":")
        if (separatorIndex > 0) {
            val prefix = rawMessage.substring(0, separatorIndex).trim()
            val message = rawMessage.substring(separatorIndex + 1).trim()
            if (isKnownErrorCode(prefix) && message.isNotBlank()) {
                return ParsedPluginError(prefix, message)
            }
        }

        return ParsedPluginError("INTERNAL_ERROR", rawMessage)
    }

    private fun buildErrorPayload(code: String, message: String): JSObject {
        return JSObject().apply {
            put("code", code)
            put("message", message)
            put("retryable", isRetryable(code))
            put("correlationId", UUID.randomUUID().toString())
            put("details", JSObject())
        }
    }

    private fun isKnownErrorCode(code: String): Boolean {
        return when (code) {
            "NETWORK_ERROR",
            "INTEGRITY_ERROR",
            "MODEL_INVALID",
            "SESSION_INIT_ERROR",
            "INFERENCE_ERROR",
            "TIMEOUT",
            "CANCELED",
            "INTERNAL_ERROR" -> true
            else -> false
        }
    }

    private fun isRetryable(code: String): Boolean {
        return when (code) {
            "NETWORK_ERROR", "TIMEOUT", "CANCELED", "SESSION_INIT_ERROR" -> true
            else -> false
        }
    }
}
