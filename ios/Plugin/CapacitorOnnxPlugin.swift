import Capacitor
import Foundation

@objc(CapacitorOnnxPlugin)
public class CapacitorOnnxPlugin: CAPPlugin {
    private var modelStore: ModelStore!
    private var sessionManager: SessionManager!
    private var inferenceService: InferenceService!

    override public func load() {
        let root = FileManager.default.urls(for: .documentDirectory, in: .userDomainMask)[0]
            .appendingPathComponent("capacitor_onnx")
        modelStore = ModelStore(rootDir: root)
        do {
            sessionManager = try SessionManager()
        } catch {
            print("[CapacitorOnnx] Failed to initialize SessionManager: \(error)")
            return
        }
        inferenceService = InferenceService(sessionManager: sessionManager, modelStore: modelStore)
    }

    @objc func isActive(_ call: CAPPluginCall) {
        call.resolve(["value": true])
    }

    @objc func loadModel(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version"),
              let url = call.getString("url") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version, url")
            return
        }

        let sha256 = call.getString("sha256")
        let forceRedownload = call.getBool("forceRedownload") ?? false
        let warmup = call.getBool("warmup") ?? false
        let sessionOptions = call.getObject("sessionOptions")
        let providerRaw = (sessionOptions?["executionProvider"] as? String ?? "auto").lowercased()
        let validProviders = ["cpu", "nnapi", "coreml", "auto", "wasm", "webgpu", "webnn"]

        guard validProviders.contains(providerRaw) else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid sessionOptions.executionProvider")
            return
        }

        let intraOp = sessionOptions?["intraOpNumThreads"] as? Int
        let interOp = sessionOptions?["interOpNumThreads"] as? Int

        if let v = intraOp, v <= 0 {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid sessionOptions.intraOpNumThreads: must be > 0")
            return
        }
        if let v = interOp, v <= 0 {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid sessionOptions.interOpNumThreads: must be > 0")
            return
        }

        let config = SessionConfig(executionProvider: providerRaw, intraOpNumThreads: intraOp, interOpNumThreads: interOp)

        Task {
            do {
                let startMs = currentTimeMs()
                let result = try await inferenceService.prepareModel(
                    modelId: modelId, version: version, url: url,
                    sha256: sha256, forceRedownload: forceRedownload,
                    sessionConfig: config
                )

                var warmupLatencyMs: Double? = nil
                var warmed = false
                if warmup {
                    let warmupStart = currentTimeMs()
                    try inferenceService.warmup(modelId: modelId, version: version)
                    warmupLatencyMs = currentTimeMs() - warmupStart
                    warmed = true
                }

                var response: [String: Any] = [
                    "status": result.cacheHit ? "cache_hit" : "downloaded",
                    "sessionReady": true,
                    "executionProviderUsed": result.executionProviderUsed,
                    "warmed": warmed,
                    "latencyMs": currentTimeMs() - startMs,
                ]
                if let wl = warmupLatencyMs { response["warmupLatencyMs"] = wl }
                call.resolve(response)
            } catch {
                rejectStructured(call, error: error)
            }
        }
    }

    @objc func warmupModel(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version")
            return
        }

        Task {
            do {
                let startMs = currentTimeMs()
                try inferenceService.warmup(modelId: modelId, version: version)
                call.resolve(["warmed": true, "latencyMs": currentTimeMs() - startMs])
            } catch {
                rejectStructured(call, error: error)
            }
        }
    }

    @objc func run(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version"),
              let inputTensor = call.getObject("inputTensor") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version, inputTensor")
            return
        }

        let inputType = inputTensor["type"] as? String ?? ""
        guard let inputDataRaw = inputTensor["data"] as? [Any],
              let inputDimsRaw = inputTensor["dims"] as? [Any] else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required inputTensor fields: data, dims")
            return
        }

        let inputData = inputDataRaw.compactMap { $0 as? Double }.map { Float($0) }
        if inputData.count != inputDataRaw.count {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid inputTensor.data: values must be numeric")
            return
        }

        let inputDims = inputDimsRaw.compactMap { ($0 as? NSNumber)?.int64Value }
        if inputDims.count != inputDimsRaw.count || inputDims.contains(where: { $0 <= 0 }) {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid inputTensor.dims: all dimensions must be positive integers")
            return
        }

        Task {
            do {
                let startMs = currentTimeMs()
                let result = try inferenceService.run(
                    modelId: modelId, version: version,
                    inputData: inputData, inputDims: inputDims, inputType: inputType
                )
                call.resolve([
                    "logits": [
                        "data": result.data.map { Double($0) },
                        "dims": result.dims.map { NSNumber(value: $0) },
                        "type": result.type,
                    ],
                    "latencyMs": currentTimeMs() - startMs,
                ])
            } catch {
                rejectStructured(call, error: error)
            }
        }
    }

    @objc func release(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version")
            return
        }
        sessionManager.closeSession(modelId: modelId, version: version)
        call.resolve()
    }

    @objc func clearModel(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version")
            return
        }
        sessionManager.closeSession(modelId: modelId, version: version)
        let removed = modelStore.clearModel(modelId: modelId, version: version)
        call.resolve(["removed": removed])
    }

    @objc func clearAllCache(_ call: CAPPluginCall) {
        sessionManager.closeAll()
        let removed = modelStore.clearAll()
        call.resolve(["removedModels": removed])
    }

    @objc func getModelStatus(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version")
            return
        }
        let status = modelStore.getModelStatus(modelId: modelId, version: version)
        var response: [String: Any] = [
            "exists": status.exists,
            "integrityOk": status.integrityOk,
            "sessionLoaded": sessionManager.hasSession(modelId: modelId, version: version),
        ]
        if let size = status.sizeBytes { response["sizeBytes"] = size }
        call.resolve(response)
    }

    @objc func getDiagnostics(_ call: CAPPluginCall) {
        call.resolve([
            "activeSessions": sessionManager.activeSessionCount(),
            "cacheEntries": modelStore.cacheEntryCount(),
        ])
    }

    // MARK: - Error helpers

    private func rejectStructured(_ call: CAPPluginCall, error: Error) {
        if let pluginError = error as? OnnxPluginError {
            rejectStructured(call, code: pluginError.code, message: pluginError.message, retryable: pluginError.retryable)
        } else {
            rejectStructured(call, code: "INTERNAL_ERROR", message: error.localizedDescription)
        }
    }

    private func rejectStructured(_ call: CAPPluginCall, code: String, message: String, retryable: Bool? = nil) {
        let isRetryable = retryable ?? ["NETWORK_ERROR", "TIMEOUT", "CANCELED", "SESSION_INIT_ERROR"].contains(code)
        let correlationId = UUID().uuidString
        call.reject(message, code, nil, [
            "code": code,
            "message": message,
            "retryable": isRetryable,
            "correlationId": correlationId,
            "details": [:] as [String: Any],
        ])
    }

    private func currentTimeMs() -> Double {
        return Date().timeIntervalSince1970 * 1000
    }
}
