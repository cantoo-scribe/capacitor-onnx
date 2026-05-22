import Capacitor
import Foundation

@objc(CapacitorOnnxPlugin)
public class CapacitorOnnxPlugin: CAPPlugin {
    private var sessionManager: SessionManager!
    private var inferenceService: InferenceService!

    override public func load() {
        do {
            sessionManager = try SessionManager()
        } catch {
            print("[CapacitorOnnx] Failed to initialize SessionManager: \(error)")
            return
        }
        inferenceService = InferenceService(sessionManager: sessionManager)
    }

    @objc func isActive(_ call: CAPPluginCall) {
        call.resolve(["value": true])
    }

    @objc func loadModel(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version")
            return
        }

        if call.hasOption("modelBuffer") {
            rejectStructured(call, code: "MODEL_INVALID", message: "modelBuffer is not supported on iOS; pass filePath (file:// URI or absolute path) instead")
            return
        }

        guard let rawFilePath = call.getString("filePath"), !rawFilePath.isEmpty else {
            rejectStructured(call, code: "MODEL_INVALID", message: "Expected exactly one of filePath or modelBuffer")
            return
        }

        let filePath = Self.resolveFilePath(rawFilePath)
        guard FileManager.default.fileExists(atPath: filePath) else {
            rejectStructured(call, code: "MODEL_INVALID", message: "Model file not found at filePath: \(rawFilePath)")
            return
        }

        let warmupInputsJson = call.getObject("warmupInputs")
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

        let warmupInputs: [String: RawTensorInternal]?
        if let warmupInputsJson = warmupInputsJson {
            guard let parsed = Self.parseTensorMap(warmupInputsJson), !parsed.isEmpty else {
                rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid warmupInputs: each entry must include numeric data and positive dims")
                return
            }
            warmupInputs = parsed
        } else {
            warmupInputs = nil
        }

        let config = SessionConfig(executionProvider: providerRaw, intraOpNumThreads: intraOp, interOpNumThreads: interOp)

        Task {
            do {
                let startMs = currentTimeMs()
                let executionProviderUsed = try inferenceService.prepareModel(
                    modelId: modelId, version: version, filePath: filePath,
                    sessionConfig: config
                )

                var warmupLatencyMs: Double? = nil
                var warmed = false
                if let warmupInputs = warmupInputs {
                    let warmupStart = currentTimeMs()
                    inferenceService.warmup(modelId: modelId, version: version, warmupInputs: warmupInputs)
                    warmupLatencyMs = currentTimeMs() - warmupStart
                    warmed = true
                }

                var response: [String: Any] = [
                    "sessionReady": true,
                    "executionProviderUsed": executionProviderUsed,
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

    @objc func run(_ call: CAPPluginCall) {
        guard let modelId = call.getString("modelId"),
              let version = call.getString("version"),
              let inputsObj = call.getObject("inputs") else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Missing required fields: modelId, version, inputs")
            return
        }

        guard let inputs = Self.parseTensorMap(inputsObj), !inputs.isEmpty else {
            rejectStructured(call, code: "INFERENCE_ERROR", message: "Invalid inputs: each entry must include numeric data and positive dims")
            return
        }

        Task {
            do {
                let startMs = currentTimeMs()
                let result = try inferenceService.run(
                    modelId: modelId, version: version, inputs: inputs
                )
                var outputs: [String: Any] = [:]
                for (name, tensor) in result {
                    outputs[name] = Self.tensorToDict(tensor)
                }
                call.resolve([
                    "outputs": outputs,
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

    // MARK: - Helpers

    private static func resolveFilePath(_ raw: String) -> String {
        if raw.hasPrefix("file://") {
            return URL(string: raw)?.path ?? raw
        }
        return raw
    }

    private func rejectStructured(_ call: CAPPluginCall, error: Error) {
        if let pluginError = error as? OnnxPluginError {
            rejectStructured(call, code: pluginError.code, message: pluginError.message, retryable: pluginError.retryable)
        } else {
            rejectStructured(call, code: "INTERNAL_ERROR", message: error.localizedDescription)
        }
    }

    private func rejectStructured(_ call: CAPPluginCall, code: String, message: String, retryable: Bool? = nil) {
        let isRetryable = retryable ?? ["TIMEOUT", "CANCELED", "SESSION_INIT_ERROR"].contains(code)
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

    private static func parseTensorMap(_ obj: [String: Any]) -> [String: RawTensorInternal]? {
        var result: [String: RawTensorInternal] = [:]
        for (name, value) in obj {
            guard let tensorObj = value as? [String: Any],
                  let tensor = parseTensor(tensorObj) else {
                return nil
            }
            result[name] = tensor
        }
        return result
    }

    private static func parseTensor(_ json: [String: Any]) -> RawTensorInternal? {
        guard let type = json["type"] as? String, !type.isEmpty else { return nil }
        guard let dataRaw = json["data"] as? [Any],
              let dimsRaw = json["dims"] as? [Any] else { return nil }

        let data = dataRaw.compactMap { ($0 as? NSNumber)?.doubleValue }
        guard data.count == dataRaw.count else { return nil }

        let dims = dimsRaw.compactMap { ($0 as? NSNumber)?.int64Value }
        guard dims.count == dimsRaw.count, !dims.isEmpty, dims.allSatisfy({ $0 > 0 }) else { return nil }

        let elementCount = dims.reduce(1, *)
        guard elementCount == Int64(data.count) else { return nil }

        return RawTensorInternal(data: data, dims: dims, type: type)
    }

    private static func tensorToDict(_ tensor: RawTensorInternal) -> [String: Any] {
        return [
            "data": tensor.data,
            "dims": tensor.dims.map { NSNumber(value: $0) },
            "type": tensor.type,
        ]
    }
}
