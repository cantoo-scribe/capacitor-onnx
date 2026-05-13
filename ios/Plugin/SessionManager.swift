import Foundation
import onnxruntime

struct SessionConfig {
    let executionProvider: String
    let intraOpNumThreads: Int?
    let interOpNumThreads: Int?

    init(executionProvider: String = "auto", intraOpNumThreads: Int? = nil, interOpNumThreads: Int? = nil) {
        self.executionProvider = executionProvider.lowercased()
        self.intraOpNumThreads = intraOpNumThreads
        self.interOpNumThreads = interOpNumThreads
    }

    func normalized() -> SessionConfig {
        let provider: String
        switch executionProvider {
        case "cpu", "coreml", "nnapi", "auto": provider = executionProvider
        default: provider = "auto"
        }
        return SessionConfig(
            executionProvider: provider,
            intraOpNumThreads: intraOpNumThreads.flatMap { $0 > 0 ? $0 : nil },
            interOpNumThreads: interOpNumThreads.flatMap { $0 > 0 ? $0 : nil }
        )
    }
}

struct SessionRef {
    let modelId: String
    let version: String
    let session: ORTSession
    let config: SessionConfig
    let executionProviderUsed: String
}

class SessionManager {
    private let environment: ORTEnvironment
    private var sessions: [String: SessionRef] = [:]
    private let lock = NSLock()

    init() throws {
        self.environment = try ORTEnvironment(loggingLevel: .warning)
    }

    func ensureSession(modelId: String, version: String, filePath: String, config: SessionConfig) throws -> SessionRef {
        let key = sessionKey(modelId, version)
        let normalized = config.normalized()

        lock.lock()
        defer { lock.unlock() }

        if let existing = sessions[key], configsMatch(existing.config, normalized) {
            return existing
        }

        sessions.removeValue(forKey: key)

        let created = try createSession(modelId: modelId, version: version, filePath: filePath, config: normalized)
        sessions[key] = created
        return created
    }

    func getSession(modelId: String, version: String) -> SessionRef? {
        lock.lock()
        defer { lock.unlock() }
        return sessions[sessionKey(modelId, version)]
    }

    func closeSession(modelId: String, version: String) {
        lock.lock()
        defer { lock.unlock() }
        sessions.removeValue(forKey: sessionKey(modelId, version))
    }

    func closeAll() {
        lock.lock()
        defer { lock.unlock() }
        sessions.removeAll()
    }

    private func createSession(modelId: String, version: String, filePath: String, config: SessionConfig) throws -> SessionRef {
        switch config.executionProvider {
        case "cpu":
            return try createCpuSession(modelId: modelId, version: version, filePath: filePath, config: config)
        case "coreml", "nnapi":
            return try createCoreMLSession(modelId: modelId, version: version, filePath: filePath, config: config)
        default:
            return try createAutoSession(modelId: modelId, version: version, filePath: filePath, config: config)
        }
    }

    private func createAutoSession(modelId: String, version: String, filePath: String, config: SessionConfig) throws -> SessionRef {
        do {
            return try createCoreMLSession(modelId: modelId, version: version, filePath: filePath, config: config)
        } catch {
            return try createCpuSession(modelId: modelId, version: version, filePath: filePath, config: config)
        }
    }

    private func createCoreMLSession(modelId: String, version: String, filePath: String, config: SessionConfig) throws -> SessionRef {
        let options = try buildOptions(config: config)
        do {
            try options.appendCoreMLExecutionProvider(withFlags: 0)
        } catch {
            throw OnnxPluginError.sessionInitError("CoreML provider is not available on this device")
        }
        let session = try ORTSession(env: environment, modelPath: filePath, sessionOptions: options)
        return SessionRef(modelId: modelId, version: version, session: session, config: config, executionProviderUsed: "coreml")
    }

    private func createCpuSession(modelId: String, version: String, filePath: String, config: SessionConfig) throws -> SessionRef {
        let options = try buildOptions(config: config)
        let session = try ORTSession(env: environment, modelPath: filePath, sessionOptions: options)
        return SessionRef(modelId: modelId, version: version, session: session, config: config, executionProviderUsed: "cpu")
    }

    private func buildOptions(config: SessionConfig) throws -> ORTSessionOptions {
        let options = try ORTSessionOptions()
        try options.setGraphOptimizationLevel(.all)
        let defaultIntra = min(ProcessInfo.processInfo.activeProcessorCount, 4)
        let intra = config.intraOpNumThreads ?? defaultIntra
        let inter = config.interOpNumThreads ?? 1
        try options.setIntraOpNumThreads(Int32(intra))
        try options.setInterOpNumThreads(Int32(inter))
        return options
    }

    private func configsMatch(_ a: SessionConfig, _ b: SessionConfig) -> Bool {
        return a.executionProvider == b.executionProvider &&
            a.intraOpNumThreads == b.intraOpNumThreads &&
            a.interOpNumThreads == b.interOpNumThreads
    }

    private func sessionKey(_ modelId: String, _ version: String) -> String {
        return "\(modelId)::\(version)"
    }
}
