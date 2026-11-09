import type { InferenceSession, Tensor } from "onnxruntime-web";
import type {
  CapacitorOnnxPlugin,
  InputTensor,
  LoadModelInput,
  LoadModelResult,
  RawTensor,
  ReleaseModelInput,
  RunInput,
  RunResult,
  TypedRawTensor,
  TypedRunResult,
} from "../definitions";
import { CapacitorOnnxError } from "../errors";
import { elapsedMs, nowMs } from "../helpers/time";
import { createSessionWithFallback } from "./provider-resolver";
import {
  applyRuntimeThreads,
  applySingleThreadRuntime,
  applyWebRuntimeConfig,
} from "./runtime-config";
import { fromOrtTensor, fromOrtTensorTyped, toOrtTensor } from "./tensor";

export class CapacitorOnnxWeb implements CapacitorOnnxPlugin {
  private sessions = new Map<string, InferenceSession>();

  private static modelKey(modelId: string, version: string): string {
    return `model-${modelId}-${version}`;
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

    applyWebRuntimeConfig(_input.sessionOptions?.web?.wasmPath, _input.sessionOptions?.web?.proxy);

    if (_input.sessionOptions?.web?.multithread === false) {
      applySingleThreadRuntime();
    } else {
      applyRuntimeThreads(_input.sessionOptions?.intraOpNumThreads);
    }

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

  async run(input: RunInput & { typedOutputs: true }): Promise<TypedRunResult>;
  async run(input: RunInput): Promise<RunResult>;
  async run(_input: RunInput): Promise<RunResult | TypedRunResult> {
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

    const feeds: Record<string, Tensor> = {};
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

    const toOutput = _input.typedOutputs ? fromOrtTensorTyped : fromOrtTensor;
    const outputs: Record<string, RawTensor | TypedRawTensor> = {};
    for (const name of session.outputNames) {
      const tensor = inferenceResult[name];
      if (tensor) {
        outputs[name] = toOutput(tensor);
      }
    }

    if (Object.keys(outputs).length === 0) {
      throw new CapacitorOnnxError("MODEL_INVALID", "Model produced no outputs.");
    }

    return {
      outputs,
      latencyMs: elapsedMs(startTime),
    } as RunResult | TypedRunResult;
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
    session: InferenceSession,
    warmupInputs: Record<string, InputTensor>,
  ): Promise<boolean> {
    const names = Object.keys(warmupInputs);
    if (names.length === 0) {
      return false;
    }

    try {
      const feeds: Record<string, Tensor> = {};
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
