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
import { fromOrtTensor, toOrtTensor } from "./tensor";

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

    if (_input.warmupInputs && Object.keys(_input.warmupInputs).length > 0) {
      const warmupStartTime = nowMs();
      warmed = await CapacitorOnnxWeb.warmupSession(session, _input.warmupInputs);
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

    const inputNames = Object.keys(_input.inputs ?? {});
    if (inputNames.length === 0) {
      throw new CapacitorOnnxError(
        "INFERENCE_ERROR",
        "run() requires at least one input tensor in `inputs`.",
      );
    }

    const feeds: Record<string, ort.Tensor> = {};
    for (const name of inputNames) {
      if (!session.inputNames.includes(name)) {
        throw new CapacitorOnnxError(
          "INFERENCE_ERROR",
          `Model has no input named '${name}'.`,
        );
      }
      feeds[name] = toOrtTensor(_input.inputs[name]);
    }

    const inferenceResult = await session.run(feeds);

    const outputs: Record<string, RawTensor> = {};
    for (const name of session.outputNames) {
      const tensor = inferenceResult[name];
      if (tensor) {
        outputs[name] = fromOrtTensor(tensor);
      }
    }

    if (Object.keys(outputs).length === 0) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model produced no outputs.");
    }

    return {
      outputs,
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
    warmupInputs: Record<string, RawTensor>,
  ): Promise<boolean> {
    const names = Object.keys(warmupInputs);
    if (names.length === 0) {
      return false;
    }

    try {
      const feeds: Record<string, ort.Tensor> = {};
      for (const name of names) {
        feeds[name] = toOrtTensor(warmupInputs[name]);
      }
      await session.run(feeds);
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
