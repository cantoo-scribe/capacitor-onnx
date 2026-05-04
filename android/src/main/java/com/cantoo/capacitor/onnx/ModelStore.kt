package com.cantoo.capacitor.onnx

import android.content.Context
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import okhttp3.OkHttpClient
import okhttp3.Request
import java.io.File
import java.io.FileOutputStream
import java.security.MessageDigest

data class ModelRef(
    val modelId: String,
    val version: String,
    val sha256: String?,
    val file: File,
)

data class PrepareModelResultInternal(
    val modelRef: ModelRef,
    val cacheHit: Boolean,
)

data class ModelStatus(
    val exists: Boolean,
    val integrityOk: Boolean,
    val sizeBytes: Long?,
)

class ModelStore(
    private val context: Context,
    private val rootDir: File,
    private val client: OkHttpClient = OkHttpClient(),
) {
    init {
        rootDir.mkdirs()
    }

    suspend fun prepare(
        modelId: String,
        version: String,
        url: String,
        sha256: String?,
        forceRedownload: Boolean,
    ): PrepareModelResultInternal = withContext(Dispatchers.IO) {
        val modelDir = File(rootDir, "$modelId/$version")
        val finalFile = File(modelDir, "model.onnx")
        val hashFile = File(modelDir, "sha256.txt")
        val normalizedHash = sha256?.trim()?.takeIf { it.isNotEmpty() }

        if (!forceRedownload && finalFile.exists()) {
            if (normalizedHash == null) {
                return@withContext PrepareModelResultInternal(
                    modelRef = ModelRef(modelId, version, null, finalFile),
                    cacheHit = true,
                )
            }

            if (hashFile.exists()) {
                val cachedHash = hashFile.readText().trim()
                if (cachedHash.equals(normalizedHash, ignoreCase = true) && verifySha256(finalFile, normalizedHash)) {
                    return@withContext PrepareModelResultInternal(
                        modelRef = ModelRef(modelId, version, normalizedHash, finalFile),
                        cacheHit = true,
                    )
                }
            }
        }

        modelDir.mkdirs()
        val tmpFile = File(modelDir, "model.tmp")
        downloadToFile(url, tmpFile)

        if (normalizedHash != null && !verifySha256(tmpFile, normalizedHash)) {
            tmpFile.delete()
            throw IllegalStateException("INTEGRITY_ERROR: sha256 mismatch for downloaded model")
        }

        if (finalFile.exists()) {
            finalFile.delete()
        }
        if (!tmpFile.renameTo(finalFile)) {
            throw IllegalStateException("Failed to promote temp model file")
        }

        if (normalizedHash != null) {
            hashFile.writeText(normalizedHash.lowercase())
        } else if (hashFile.exists()) {
            hashFile.delete()
        }

        PrepareModelResultInternal(
            modelRef = ModelRef(modelId, version, normalizedHash, finalFile),
            cacheHit = false,
        )
    }

    suspend fun resolve(modelId: String, version: String): ModelRef = withContext(Dispatchers.IO) {
        val modelDir = File(rootDir, "$modelId/$version")
        val finalFile = File(modelDir, "model.onnx")
        val hashFile = File(modelDir, "sha256.txt")

        if (!finalFile.exists()) {
            throw IllegalStateException("MODEL_INVALID: model is not prepared")
        }

        val hash = if (hashFile.exists()) hashFile.readText().trim().ifEmpty { null } else null
        ModelRef(modelId, version, hash, finalFile)
    }

    suspend fun getModelStatus(modelId: String, version: String): ModelStatus = withContext(Dispatchers.IO) {
        val modelDir = File(rootDir, "$modelId/$version")
        val finalFile = File(modelDir, "model.onnx")
        val hashFile = File(modelDir, "sha256.txt")

        if (!finalFile.exists()) {
            return@withContext ModelStatus(false, false, null)
        }

        val hash = if (hashFile.exists()) hashFile.readText().trim() else ""
        ModelStatus(
            exists = true,
            integrityOk = if (hash.isBlank()) true else verifySha256(finalFile, hash),
            sizeBytes = finalFile.length(),
        )
    }

    suspend fun clearModel(modelId: String, version: String): Boolean = withContext(Dispatchers.IO) {
        val dir = File(rootDir, "$modelId/$version")
        deleteRecursively(dir)
    }

    suspend fun clearAll(): Int = withContext(Dispatchers.IO) {
        val count = cacheEntryCount()
        deleteRecursively(rootDir)
        rootDir.mkdirs()
        count
    }

    fun cacheEntryCount(): Int {
        if (!rootDir.exists()) return 0
        var count = 0
        rootDir.listFiles()?.forEach { modelIdDir ->
            modelIdDir.listFiles()?.forEach { versionDir ->
                if (File(versionDir, "model.onnx").exists()) count += 1
            }
        }
        return count
    }

    private fun downloadToFile(url: String, destination: File) {
        val request = Request.Builder().url(url).build()
        client.newCall(request).execute().use { response ->
            if (!response.isSuccessful) {
                throw IllegalStateException("NETWORK_ERROR: HTTP ${response.code}")
            }
            val body = response.body ?: throw IllegalStateException("NETWORK_ERROR: empty response body")
            FileOutputStream(destination).use { output ->
                body.byteStream().copyTo(output)
            }
        }
    }

    private fun verifySha256(file: File, expectedSha256: String): Boolean {
        val digest = MessageDigest.getInstance("SHA-256")
        file.inputStream().use { stream ->
            val buffer = ByteArray(DEFAULT_BUFFER_SIZE)
            while (true) {
                val bytesRead = stream.read(buffer)
                if (bytesRead < 0) break
                digest.update(buffer, 0, bytesRead)
            }
        }
        val actualHash = digest.digest().joinToString("") { "%02x".format(it) }
        return actualHash.equals(expectedSha256, ignoreCase = true)
    }

    private fun deleteRecursively(file: File): Boolean {
        if (!file.exists()) return false
        file.listFiles()?.forEach { child -> deleteRecursively(child) }
        return file.delete()
    }
}
