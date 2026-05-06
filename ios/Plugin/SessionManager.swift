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
    private var desiredConfigs: [String: SessionConfig] = [:]
    private let lock = NSLock()

    init() throws {
        self.environment = try ORTEnvironment(loggingLevel: .warning)
    }

    func ensureSession(modelRef: ModelRef, config: SessionConfig? = nil) throws -> SessionRef {
        let key = sessionKey(modelRef.modelId, modelRef.version)
        let normalized = (config?.normalized() ?? desiredConfigs[key] ?? SessionConfig()).normalized()

        lock.lock()
        defer { lock.unlock() }

        desiredConfigs[key] = normalized

        if let existing = sessions[key], configsMatch(existing.config, normalized) {
            return existing
        }

        sessions.removeValue(forKey: key)

        let created = try createSession(modelRef: modelRef, config: normalized)
        sessions[key] = created
        return created
    }

    func hasSession(modelId: String, version: String) -> Bool {
        lock.lock()
        defer { lock.unlock() }
        return sessions[sessionKey(modelId, version)] != nil
    }

    func closeSession(modelId: String, version: String) {
        lock.lock()
        defer { lock.unlock() }
        let key = sessionKey(modelId, version)
        desiredConfigs.removeValue(forKey: key)
        sessions.removeValue(forKey: key)
    }

    func closeAll() {
        lock.lock()
        defer { lock.unlock() }
        sessions.removeAll()
        desiredConfigs.removeAll()
    }

    func activeSessionCount() -> Int {
        lock.lock()
        defer { lock.unlock() }
        return sessions.count
    }

    private func createSession(modelRef: ModelRef, config: SessionConfig) throws -> SessionRef {
        switch config.executionProvider {
        case "cpu":
            return try createCpuSession(modelRef: modelRef, config: config)
        case "coreml", "nnapi":
            return try createCoreMLSession(modelRef: modelRef, config: config)
        default:
            return try createAutoSession(modelRef: modelRef, config: config)
        }
    }

    private func createAutoSession(modelRef: ModelRef, config: SessionConfig) throws -> SessionRef {
        do {
            return try createCoreMLSession(modelRef: modelRef, config: config)
        } catch {
            return try createCpuSession(modelRef: modelRef, config: config)
        }
    }

    private func createCoreMLSession(modelRef: ModelRef, config: SessionConfig) throws -> SessionRef {
        let options = try buildOptions(config: config)
        do {
            try options.appendCoreMLExecutionProvider(withFlags: 0)
        } catch {
            throw OnnxPluginError.sessionInitError("CoreML provider is not available on this device")
        }
        let session = try ORTSession(env: environment, modelPath: modelRef.filePath, sessionOptions: options)
        return SessionRef(modelId: modelRef.modelId, version: modelRef.version, session: session, config: config, executionProviderUsed: "coreml")
    }

    private func createCpuSession(modelRef: ModelRef, config: SessionConfig) throws -> SessionRef {
        let options = try buildOptions(config: config)
        let session = try ORTSession(env: environment, modelPath: modelRef.filePath, sessionOptions: options)
        return SessionRef(modelId: modelRef.modelId, version: modelRef.version, session: session, config: config, executionProviderUsed: "cpu")
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
