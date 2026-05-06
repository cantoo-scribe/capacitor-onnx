import Foundation
import CryptoKit

enum OnnxPluginError: Error {
    case networkError(String)
    case integrityError(String)
    case modelInvalid(String)
    case sessionInitError(String)
    case inferenceError(String)
    case internalError(String)

    var code: String {
        switch self {
        case .networkError: return "NETWORK_ERROR"
        case .integrityError: return "INTEGRITY_ERROR"
        case .modelInvalid: return "MODEL_INVALID"
        case .sessionInitError: return "SESSION_INIT_ERROR"
        case .inferenceError: return "INFERENCE_ERROR"
        case .internalError: return "INTERNAL_ERROR"
        }
    }

    var message: String {
        switch self {
        case .networkError(let m), .integrityError(let m), .modelInvalid(let m),
             .sessionInitError(let m), .inferenceError(let m), .internalError(let m):
            return m
        }
    }

    var retryable: Bool {
        switch self {
        case .networkError, .sessionInitError: return true
        default: return false
        }
    }
}

struct ModelRef {
    let modelId: String
    let version: String
    let sha256: String?
    let filePath: String
    let cacheHit: Bool
}

struct ModelStatus {
    let exists: Bool
    let integrityOk: Bool
    let sizeBytes: Int64?
}

class ModelStore {
    private let rootDir: URL

    init(rootDir: URL) {
        self.rootDir = rootDir
        try? FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
    }

    func prepare(modelId: String, version: String, url: String, sha256: String?, forceRedownload: Bool) async throws -> ModelRef {
        let modelDir = rootDir.appendingPathComponent("\(modelId)/\(version)")
        let finalFile = modelDir.appendingPathComponent("model.onnx")
        let hashFile = modelDir.appendingPathComponent("sha256.txt")
        let normalizedHash = sha256?.trimmingCharacters(in: .whitespaces).nilIfEmpty

        if !forceRedownload && FileManager.default.fileExists(atPath: finalFile.path) {
            if let expected = normalizedHash {
                if let cached = try? String(contentsOf: hashFile, encoding: .utf8).trimmingCharacters(in: .whitespaces),
                   cached.lowercased() == expected.lowercased(),
                   verifySha256(file: finalFile, expected: expected) {
                    return ModelRef(modelId: modelId, version: version, sha256: normalizedHash, filePath: finalFile.path, cacheHit: true)
                }
            } else {
                return ModelRef(modelId: modelId, version: version, sha256: nil, filePath: finalFile.path, cacheHit: true)
            }
        }

        try FileManager.default.createDirectory(at: modelDir, withIntermediateDirectories: true)
        let tmpFile = modelDir.appendingPathComponent("model.tmp")

        try await downloadToFile(url: url, destination: tmpFile)

        if let expected = normalizedHash {
            guard verifySha256(file: tmpFile, expected: expected) else {
                try? FileManager.default.removeItem(at: tmpFile)
                throw OnnxPluginError.integrityError("sha256 mismatch for downloaded model")
            }
        }

        if FileManager.default.fileExists(atPath: finalFile.path) {
            try FileManager.default.removeItem(at: finalFile)
        }
        do {
            try FileManager.default.moveItem(at: tmpFile, to: finalFile)
        } catch {
            throw OnnxPluginError.internalError("Failed to promote temp model file: \(error.localizedDescription)")
        }

        if let hash = normalizedHash {
            try? hash.lowercased().write(to: hashFile, atomically: true, encoding: .utf8)
        } else if FileManager.default.fileExists(atPath: hashFile.path) {
            try? FileManager.default.removeItem(at: hashFile)
        }

        return ModelRef(modelId: modelId, version: version, sha256: normalizedHash, filePath: finalFile.path, cacheHit: false)
    }

    func resolve(modelId: String, version: String) throws -> ModelRef {
        let finalFile = rootDir.appendingPathComponent("\(modelId)/\(version)/model.onnx")
        let hashFile = rootDir.appendingPathComponent("\(modelId)/\(version)/sha256.txt")

        guard FileManager.default.fileExists(atPath: finalFile.path) else {
            throw OnnxPluginError.modelInvalid("model is not prepared")
        }

        let hash = (try? String(contentsOf: hashFile, encoding: .utf8).trimmingCharacters(in: .whitespaces)).nilIfEmpty
        return ModelRef(modelId: modelId, version: version, sha256: hash, filePath: finalFile.path, cacheHit: true)
    }

    func getModelStatus(modelId: String, version: String) -> ModelStatus {
        let finalFile = rootDir.appendingPathComponent("\(modelId)/\(version)/model.onnx")
        let hashFile = rootDir.appendingPathComponent("\(modelId)/\(version)/sha256.txt")

        guard FileManager.default.fileExists(atPath: finalFile.path) else {
            return ModelStatus(exists: false, integrityOk: false, sizeBytes: nil)
        }

        let hash = (try? String(contentsOf: hashFile, encoding: .utf8).trimmingCharacters(in: .whitespaces)) ?? ""
        let integrityOk = hash.isEmpty ? true : verifySha256(file: finalFile, expected: hash)
        let sizeBytes = (try? FileManager.default.attributesOfItem(atPath: finalFile.path)[.size] as? Int64) ?? 0

        return ModelStatus(exists: true, integrityOk: integrityOk, sizeBytes: sizeBytes)
    }

    func clearModel(modelId: String, version: String) -> Bool {
        let dir = rootDir.appendingPathComponent("\(modelId)/\(version)")
        guard FileManager.default.fileExists(atPath: dir.path) else { return false }
        try? FileManager.default.removeItem(at: dir)
        return true
    }

    func clearAll() -> Int {
        let count = cacheEntryCount()
        try? FileManager.default.removeItem(at: rootDir)
        try? FileManager.default.createDirectory(at: rootDir, withIntermediateDirectories: true)
        return count
    }

    func cacheEntryCount() -> Int {
        guard let modelIdDirs = try? FileManager.default.contentsOfDirectory(at: rootDir, includingPropertiesForKeys: nil) else { return 0 }
        var count = 0
        for modelIdDir in modelIdDirs {
            guard let versionDirs = try? FileManager.default.contentsOfDirectory(at: modelIdDir, includingPropertiesForKeys: nil) else { continue }
            for versionDir in versionDirs {
                if FileManager.default.fileExists(atPath: versionDir.appendingPathComponent("model.onnx").path) {
                    count += 1
                }
            }
        }
        return count
    }

    private func downloadToFile(url urlString: String, destination: URL) async throws {
        guard let url = URL(string: urlString) else {
            throw OnnxPluginError.networkError("Invalid URL: \(urlString)")
        }

        let (data, response) = try await URLSession.shared.data(from: url)

        if let httpResponse = response as? HTTPURLResponse, !(200..<300).contains(httpResponse.statusCode) {
            throw OnnxPluginError.networkError("HTTP \(httpResponse.statusCode)")
        }

        try data.write(to: destination)
    }

    private func verifySha256(file: URL, expected: String) -> Bool {
        guard let stream = InputStream(url: file) else { return false }
        stream.open()
        defer { stream.close() }

        var hasher = SHA256()
        let bufferSize = 65536
        let buffer = UnsafeMutablePointer<UInt8>.allocate(capacity: bufferSize)
        defer { buffer.deallocate() }

        while stream.hasBytesAvailable {
            let bytesRead = stream.read(buffer, maxLength: bufferSize)
            if bytesRead <= 0 { break }
            hasher.update(data: Data(bytes: buffer, count: bytesRead))
        }

        let digest = hasher.finalize()
        let actual = digest.map { String(format: "%02x", $0) }.joined()
        return actual.lowercased() == expected.lowercased()
    }
}

private extension String {
    var nilIfEmpty: String? { isEmpty ? nil : self }
}

private extension Optional where Wrapped == String {
    var nilIfEmpty: String? { self?.nilIfEmpty }
}
