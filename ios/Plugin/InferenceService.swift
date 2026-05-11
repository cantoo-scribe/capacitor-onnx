import Foundation
import onnxruntime

struct RawTensorInternal {
    let data: [Float]
    let dims: [Int64]
    let type: String
}

class InferenceService {
    private let sessionManager: SessionManager
    private let modelStore: ModelStore
    private var perSessionLocks: [String: NSLock] = [:]
    private let locksLock = NSLock()

    init(sessionManager: SessionManager, modelStore: ModelStore) {
        self.sessionManager = sessionManager
        self.modelStore = modelStore
    }

    func prepareModel(
        modelId: String,
        version: String,
        url: String,
        sha256: String?,
        forceRedownload: Bool,
        sessionConfig: SessionConfig
    ) async throws -> (cacheHit: Bool, executionProviderUsed: String) {
        let modelRef = try await modelStore.prepare(modelId: modelId, version: version, url: url, sha256: sha256, forceRedownload: forceRedownload)
        let sessionRef = try sessionManager.ensureSession(modelRef: modelRef, config: sessionConfig)
        return (cacheHit: modelRef.cacheHit, executionProviderUsed: sessionRef.executionProviderUsed)
    }

    func warmup(modelId: String, version: String, warmupInput: RawTensorInternal? = nil) throws {
        let modelRef = try modelStore.resolve(modelId: modelId, version: version)
        let sessionRef = try sessionManager.ensureSession(modelRef: modelRef)
        if let warmupInput = warmupInput {
            try runWarmup(session: sessionRef.session, warmupInput: warmupInput)
        }
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

        let modelRef = try modelStore.resolve(modelId: modelId, version: version)
        let lock = sessionLock(for: modelId, version: version)

        lock.lock()
        defer { lock.unlock() }

        let sessionRef = try sessionManager.ensureSession(modelRef: modelRef)
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
