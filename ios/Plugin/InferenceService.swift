import Foundation
// The ONNX Runtime Objective-C API ships as the `onnxruntime` module via SPM
// (onnxruntime-swift-package-manager) and as `onnxruntime_objc` via CocoaPods
// (the onnxruntime-objc pod). Resolve whichever channel the host app uses.
#if canImport(onnxruntime)
import onnxruntime
#else
import onnxruntime_objc
#endif

enum OnnxPluginError: Error {
    case modelInvalid(String)
    case sessionInitError(String)
    case inferenceError(String)
    case internalError(String)

    var code: String {
        switch self {
        case .modelInvalid: return "MODEL_INVALID"
        case .sessionInitError: return "SESSION_INIT_ERROR"
        case .inferenceError: return "INFERENCE_ERROR"
        case .internalError: return "INTERNAL_ERROR"
        }
    }

    var message: String {
        switch self {
        case .modelInvalid(let m), .sessionInitError(let m),
             .inferenceError(let m), .internalError(let m):
            return m
        }
    }

    var retryable: Bool {
        switch self {
        case .sessionInitError: return true
        default: return false
        }
    }
}

struct RawTensorInternal {
    let data: [Double]
    let dims: [Int64]
    let type: String
}

class InferenceService {
    private let sessionManager: SessionManager
    private var perSessionLocks: [String: NSLock] = [:]
    private let locksLock = NSLock()

    init(sessionManager: SessionManager) {
        self.sessionManager = sessionManager
    }

    func prepareModel(
        modelId: String,
        version: String,
        filePath: String,
        sessionConfig: SessionConfig
    ) throws -> String {
        let sessionRef = try sessionManager.ensureSession(
            modelId: modelId,
            version: version,
            filePath: filePath,
            config: sessionConfig
        )
        return sessionRef.executionProviderUsed
    }

    func warmup(modelId: String, version: String, warmupInputs: [String: RawTensorInternal]) {
        guard let sessionRef = sessionManager.getSession(modelId: modelId, version: version) else {
            return
        }
        guard !warmupInputs.isEmpty else { return }
        do {
            var feeds: [String: ORTValue] = [:]
            for (name, raw) in warmupInputs {
                feeds[name] = try buildOrtValue(name: name, raw: raw)
            }
            let outputNames = Set(try sessionRef.session.outputNames())
            _ = try sessionRef.session.run(withInputs: feeds, outputNames: outputNames, runOptions: nil)
        } catch {
            // best-effort warmup; surface failures via subsequent run() calls
        }
    }

    func run(
        modelId: String,
        version: String,
        inputs: [String: RawTensorInternal]
    ) throws -> [String: RawTensorInternal] {
        guard !inputs.isEmpty else {
            throw OnnxPluginError.inferenceError("run() requires at least one input tensor")
        }
        guard let sessionRef = sessionManager.getSession(modelId: modelId, version: version) else {
            throw OnnxPluginError.sessionInitError("model not loaded, call loadModel first")
        }

        let lock = sessionLock(for: modelId, version: version)
        lock.lock()
        defer { lock.unlock() }

        let session = sessionRef.session
        let modelInputNames = Set(try session.inputNames())
        let outputNames = try session.outputNames()
        guard !outputNames.isEmpty else {
            throw OnnxPluginError.modelInvalid("model has no outputs")
        }

        var feeds: [String: ORTValue] = [:]
        for (name, raw) in inputs {
            guard modelInputNames.contains(name) else {
                throw OnnxPluginError.inferenceError("model has no input named '\(name)'")
            }
            try validateTensor(name: name, raw: raw)
            feeds[name] = try buildOrtValue(name: name, raw: raw)
        }

        let runOutputs = try session.run(
            withInputs: feeds,
            outputNames: Set(outputNames),
            runOptions: nil
        )

        var result: [String: RawTensorInternal] = [:]
        for name in outputNames {
            guard let value = runOutputs[name] else { continue }
            result[name] = try readOrtValue(value)
        }
        guard !result.isEmpty else {
            throw OnnxPluginError.modelInvalid("model produced no output")
        }
        return result
    }

    private func validateTensor(name: String, raw: RawTensorInternal) throws {
        guard !raw.data.isEmpty else {
            throw OnnxPluginError.inferenceError("input '\(name)' data is empty")
        }
        guard !raw.dims.isEmpty else {
            throw OnnxPluginError.inferenceError("input '\(name)' dims is empty")
        }
        guard raw.dims.allSatisfy({ $0 > 0 }) else {
            throw OnnxPluginError.inferenceError("input '\(name)' dims must have positive dimensions")
        }
        let elementCount = raw.dims.reduce(1, *)
        guard elementCount == Int64(raw.data.count) else {
            throw OnnxPluginError.inferenceError("input '\(name)' data size does not match dims")
        }
    }

    private func buildOrtValue(name: String, raw: RawTensorInternal) throws -> ORTValue {
        let shape = raw.dims.map { NSNumber(value: $0) }
        let data = NSMutableData()
        let elementType: ORTTensorElementDataType

        switch raw.type {
        case "float32":
            var values = raw.data.map { Float($0) }
            data.append(&values, length: values.count * MemoryLayout<Float>.stride)
            elementType = .float
        case "int32":
            var values = raw.data.map { Int32($0) }
            data.append(&values, length: values.count * MemoryLayout<Int32>.stride)
            elementType = .int32
        case "int64":
            var values = raw.data.map { Int64($0) }
            data.append(&values, length: values.count * MemoryLayout<Int64>.stride)
            elementType = .int64
        case "uint8":
            var values = raw.data.map { UInt8($0) }
            data.append(&values, length: values.count * MemoryLayout<UInt8>.stride)
            elementType = .uInt8
        default:
            // Note: onnxruntime-objc's ORTTensorElementDataType has no `bool`
            // case, so `bool` tensors are not supported on iOS (they are on
            // Android/Web). `float16`/`uint32` are likewise unsupported here.
            throw OnnxPluginError.inferenceError(
                "unsupported tensor type '\(raw.type)' for input '\(name)'"
            )
        }

        return try ORTValue(tensorData: data, elementType: elementType, shape: shape)
    }

    private func readOrtValue(_ value: ORTValue) throws -> RawTensorInternal {
        let info = try value.tensorTypeAndShapeInfo()
        let dims = info.shape.map { $0.int64Value }
        let raw = try value.tensorData() as Data

        let data: [Double]
        let type: String
        switch info.elementType {
        case .float:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Float.self)) }.map { Double($0) }
            type = "float32"
        case .int32:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Int32.self)) }.map { Double($0) }
            type = "int32"
        case .int64:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: Int64.self)) }.map { Double($0) }
            type = "int64"
        case .uInt8:
            data = raw.withUnsafeBytes { Array($0.bindMemory(to: UInt8.self)) }.map { Double($0) }
            type = "uint8"
        default:
            throw OnnxPluginError.inferenceError("unsupported output tensor type")
        }

        return RawTensorInternal(data: data, dims: dims, type: type)
    }

    private func sessionLock(for modelId: String, version: String) -> NSLock {
        let key = "\(modelId)::\(version)"
        locksLock.lock()
        defer { locksLock.unlock() }
        if let existing = perSessionLocks[key] { return existing }
        let newLock = NSLock()
        perSessionLocks[key] = newLock
        return newLock
    }
}
