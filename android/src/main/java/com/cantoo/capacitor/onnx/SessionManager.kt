package com.cantoo.capacitor.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import java.util.concurrent.ConcurrentHashMap

data class SessionConfig(
    val executionProvider: String = "auto",
    val intraOpNumThreads: Int? = null,
    val interOpNumThreads: Int? = 1,
) {
    fun normalized(): SessionConfig {
        val provider = when (executionProvider.lowercase()) {
            "cpu", "nnapi", "auto" -> executionProvider.lowercase()
            else -> "auto"
        }
        val intra = intraOpNumThreads?.takeIf { it > 0 }
        val inter = interOpNumThreads?.takeIf { it > 0 }
        return SessionConfig(provider, intra, inter)
    }
}

data class SessionRef(
    val modelId: String,
    val version: String,
    val session: OrtSession,
    val config: SessionConfig,
    val executionProviderUsed: String,
)

class SessionManager {
    private val environment: OrtEnvironment = OrtEnvironment.getEnvironment()
    private val sessions = ConcurrentHashMap<String, SessionRef>()

    fun ortEnvironment(): OrtEnvironment = environment

    fun ensureSession(modelId: String, version: String, filePath: String, config: SessionConfig): SessionRef {
        val key = key(modelId, version)
        val normalized = config.normalized()

        synchronized(this) {
            val existing = sessions[key]
            if (existing != null && existing.config == normalized) {
                return existing
            }

            if (existing != null) {
                existing.session.close()
                sessions.remove(key)
            }

            val created = createSession(modelId, version, filePath, normalized)
            sessions[key] = created
            return created
        }
    }

    fun getSession(modelId: String, version: String): SessionRef? = sessions[key(modelId, version)]

    fun closeSession(modelId: String, version: String) {
        sessions.remove(key(modelId, version))?.session?.close()
    }

    fun closeAll() {
        sessions.values.forEach { it.session.close() }
        sessions.clear()
    }

    private fun createSession(modelId: String, version: String, filePath: String, config: SessionConfig): SessionRef {
        return when (config.executionProvider) {
            "cpu" -> createCpuSession(modelId, version, filePath, config)
            "nnapi" -> createNnapiSession(modelId, version, filePath, config)
            else -> createAutoSession(modelId, version, filePath, config)
        }
    }

    private fun createAutoSession(modelId: String, version: String, filePath: String, config: SessionConfig): SessionRef {
        return try {
            createNnapiSession(modelId, version, filePath, config)
        } catch (_: Throwable) {
            createCpuSession(modelId, version, filePath, config)
        }
    }

    private fun createNnapiSession(modelId: String, version: String, filePath: String, config: SessionConfig): SessionRef {
        val options = baseOptions(config)
        if (!tryEnableNnapi(options)) {
            throw IllegalStateException("SESSION_INIT_ERROR: NNAPI provider is not available on this device/build")
        }
        val session = environment.createSession(filePath, options)
        return SessionRef(modelId, version, session, config, "nnapi")
    }

    private fun createCpuSession(modelId: String, version: String, filePath: String, config: SessionConfig): SessionRef {
        val options = baseOptions(config)
        val session = environment.createSession(filePath, options)
        return SessionRef(modelId, version, session, config, "cpu")
    }

    private fun baseOptions(config: SessionConfig): OrtSession.SessionOptions {
        val defaultIntra = Runtime.getRuntime().availableProcessors().coerceAtMost(4).coerceAtLeast(1)
        val intra = config.intraOpNumThreads ?: defaultIntra
        val inter = config.interOpNumThreads ?: 1

        return OrtSession.SessionOptions().apply {
            setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
            setIntraOpNumThreads(intra)
            setInterOpNumThreads(inter)
        }
    }

    private fun tryEnableNnapi(options: OrtSession.SessionOptions): Boolean {
        return try {
            val addNnapi = options::class.java.methods.firstOrNull { method ->
                method.name == "addNnapi" && method.parameterCount == 0
            } ?: return false
            addNnapi.invoke(options)
            true
        } catch (_: Throwable) {
            false
        }
    }

    private fun key(modelId: String, version: String): String = "$modelId::$version"
}
