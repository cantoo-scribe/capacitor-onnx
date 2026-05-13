import Foundation
import onnxruntime

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
    let data: [Float]
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

    func warmup(modelId: String, version: String, warmupInput: RawTensorInternal) throws {
        guard let sessionRef = sessionManager.getSession(modelId: modelId, version: version) else {
            throw OnnxPluginError.sessionInitError("model not loaded, call loadModel first")
        }
        try runWarmup(session: sessionRef.session, warmupInput: warmupInput)
    }

    func run(
        modelId: String,
        version: String,
        inputData: [Float],
        inputDims: [Int64],
        inputType: String
    ) throws -> RawTensorInternal {
        guard inputType == "float32" else {
            throw OnnxPluginError.inferenceError("only float32 tensors are supported")
        }
        guard !inputData.isEmpty else {
            throw OnnxPluginError.inferenceError("input tensor data is empty")
        }
        guard !inputDims.isEmpty else {
            throw OnnxPluginError.inferenceError("input tensor dims is empty")
        }

        let elementCount = inputDims.reduce(1, *)
        guard elementCount > 0 else {
            throw OnnxPluginError.inferenceError("input tensor dims must have positive dimensions")
        }
        guard elementCount == Int64(inputData.count) else {
            throw OnnxPluginError.inferenceError("input tensor data size does not match dims")
        }

        guard let sessionRef = sessionManager.getSession(modelId: modelId, version: version) else {
            throw OnnxPluginError.sessionInitError("model not loaded, call loadModel first")
        }
        let lock = sessionLock(for: modelId, version: version)

        lock.lock()
        defer { lock.unlock() }

        let session = sessionRef.session

        let inputNames = try session.inputNames()
        guard let inputName = inputNames.first else {
            throw OnnxPluginError.modelInvalid("model has no inputs")
        }

        let outputNames = try session.outputNames()
        guard let outputName = outputNames.first else {
            throw OnnxPluginError.modelInvalid("model has no outputs")
        }

        let nsInputDims = inputDims.map { NSNumber(value: $0) }
        var inputDataCopy = inputData
        let tensorData = NSMutableData(bytes: &inputDataCopy, length: inputDataCopy.count * MemoryLayout<Float>.size)
        let inputTensor = try ORTValue(tensorData: tensorData, elementType: .float, shape: nsInputDims)

        let outputs = try session.run(withInputs: [inputName: inputTensor], outputNames: Set([outputName]), runOptions: nil)

        guard let outputValue = outputs[outputName] else {
            throw OnnxPluginError.modelInvalid("model produced no output")
        }

        let outputData = try outputValue.tensorData() as Data
        let floatData = outputData.withUnsafeBytes { ptr -> [Float] in
            Array(ptr.bindMemory(to: Float.self))
        }

        let typeInfo = try outputValue.tensorTypeAndShapeInfo()
        let dims = typeInfo.shape.map { $0.int64Value }

        return RawTensorInternal(data: floatData, dims: dims, type: "float32")
    }

    private func runWarmup(session: ORTSession, warmupInput: RawTensorInternal) throws {
        let inputNames = try session.inputNames()
        guard let inputName = inputNames.first else {
            throw OnnxPluginError.modelInvalid("model has no inputs to warm up")
        }

        let nsDims = warmupInput.dims.map { NSNumber(value: $0) }
        var dataCopy = warmupInput.data
        let tensorData = NSMutableData(
            bytes: &dataCopy,
            length: dataCopy.count * MemoryLayout<Float>.size,
        )
        let tensor = try ORTValue(tensorData: tensorData, elementType: .float, shape: nsDims)

        _ = try? session.run(
            withInputs: [inputName: tensor],
            outputNames: Set(try session.outputNames()),
            runOptions: nil,
        )
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
