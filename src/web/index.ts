import * as ort from "onnxruntime-web";
import type {
  CapacitorOnnxPlugin,
  LoadModelInput,
  LoadModelResult,
  RawTensor,
  ReleaseModelInput,
  RunInput,
  RunResult,
  WebConfig,
} from "../definitions";
import { CapacitorOnnxError } from "../errors";
import { elapsedMs, nowMs } from "../helpers/time";
import { createSessionWithFallback } from "./provider-resolver";
import { applyRuntimeThreads, applyWebRuntimeConfig } from "./runtime-config";

export class CapacitorOnnxWeb implements CapacitorOnnxPlugin {
  private sessions = new Map<string, ort.InferenceSession>();

  private static modelKey(modelId: string, version: string): string {
    return `model-${modelId}-${version}`;
  }

  static setWebConfig(config?: WebConfig): void {
    applyWebRuntimeConfig(config?.wasmPath);
  }

  async loadModel(_input: LoadModelInput): Promise<LoadModelResult> {
    if (_input.filePath !== undefined) {
      throw new CapacitorOnnxError(
        "MODEL_INVALID",
        "filePath is not supported on web; pass modelBuffer (Uint8Array) instead",
      );
    }
    if (!_input.modelBuffer) {
      throw new CapacitorOnnxError(
        "MODEL_INVALID",
        "Expected exactly one of filePath or modelBuffer",
      );
    }

    applyRuntimeThreads(_input.sessionOptions?.intraOpNumThreads);

    const startTime = nowMs();
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const executionProvider = _input.sessionOptions?.executionProvider ?? "auto";

    await this.sessions
      .get(modelKey)
      ?.release()
      .catch(() => {});

    const created = await createSessionWithFallback(_input.modelBuffer, executionProvider);
    const session = created.session;

    this.sessions.set(modelKey, session);

    let warmed = false;
    let warmupLatencyMs: number | undefined;

    if (_input.warmupInput) {
      const warmupStartTime = nowMs();
      warmed = await CapacitorOnnxWeb.warmupSession(session, _input.warmupInput);
      warmupLatencyMs = elapsedMs(warmupStartTime);
    }

    return {
      sessionReady: true,
      latencyMs: elapsedMs(startTime),
      warmed,
      warmupLatencyMs,
      executionProviderUsed: created.providerUsed,
    };
  }

  async run(_input: RunInput): Promise<RunResult> {
    const startTime = nowMs();
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const session = this.sessions.get(modelKey);

    if (!session) {
      throw new CapacitorOnnxError(
        "SESSION_INIT_ERROR",
        "ONNX Runtime session is not initialized. Please load a model before running inference.",
      );
    }

    const inputTensor = new ort.Tensor(
      _input.inputTensor.type,
      _input.inputTensor.data,
      _input.inputTensor.dims,
    );

    const inputName = session.inputNames[0];
    if (!inputName) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model has no inputs.");
    }

    const inferenceResult = await session.run({ [inputName]: inputTensor });
    const outputName = session.outputNames[0] ?? Object.keys(inferenceResult)[0];
    const outputTensor = outputName ? inferenceResult[outputName] : undefined;

    if (!outputTensor) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model has no outputs.");
    }

    return {
      logits: {
        type: outputTensor.type as RawTensor["type"],
        data: Array.from(outputTensor.data as ArrayLike<number | bigint>, Number),
        dims: outputTensor.dims,
      },
      latencyMs: elapsedMs(startTime),
    };
  }

  async release(_input: ReleaseModelInput): Promise<void> {
    const modelKey = CapacitorOnnxWeb.modelKey(_input.modelId, _input.version);
    const session = this.sessions.get(modelKey);
    if (session) {
      await session.release().catch(() => {});
      this.sessions.delete(modelKey);
    }
  }

  private static async warmupSession(
    session: ort.InferenceSession,
    warmupInput: RawTensor,
  ): Promise<boolean> {
    const inputName = session.inputNames[0];
    if (!inputName) {
      return false;
    }

    try {
      const tensor = new ort.Tensor(warmupInput.type, warmupInput.data, warmupInput.dims);
      await session.run({ [inputName]: tensor });
      return true;
    } catch (err) {
      console.warn(
        "[CapacitorOnnxWeb] Warmup inference failed; continuing without warmup.",
        err,
      );
      return false;
    }
  }
}
