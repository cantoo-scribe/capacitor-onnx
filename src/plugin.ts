import { Capacitor, registerPlugin } from "@capacitor/core";

import type {
  CapacitorOnnxPlugin,
  InputTensor,
  LoadModelInput,
  RunInput,
  RunResult,
  TensorType,
  TypedRawTensor,
  TypedRunResult,
} from "./definitions";
import { asTypedTensorData, base64ToTypedArray, typedArrayToBase64 } from "./tensor-data";

const BridgedCapacitorOnnx = registerPlugin<CapacitorOnnxPlugin>("CapacitorOnnx", {
  web: () => import("./web/wrapper").then((m) => new m.WrapperCapacitorOnnxWeb()),
});

/**
 * A tensor as it crosses the Android/iOS bridge, which only carries JSON: a typed array goes
 * as the base64 of its bytes (`dataBase64`), a number[] as is (`data`).
 */
type NativeTensor = {
  type: TensorType;
  dims?: readonly number[];
  data?: number[];
  dataBase64?: string;
};

const toNativeTensor = (tensor: InputTensor): NativeTensor =>
  Array.isArray(tensor.data)
    ? (tensor as NativeTensor)
    : {
        type: tensor.type,
        dims: tensor.dims,
        dataBase64: typedArrayToBase64(asTypedTensorData(tensor.type, tensor.data)),
      };

const toNativeTensors = (tensors: Record<string, InputTensor>) =>
  Object.fromEntries(
    Object.entries(tensors).map(([name, tensor]) => [name, toNativeTensor(tensor)]),
  );

const fromNativeTypedTensor = ({
  type,
  dims,
  data,
  dataBase64,
}: NativeTensor): TypedRawTensor => ({
  type,
  dims,
  data:
    dataBase64 !== undefined
      ? base64ToTypedArray(type, dataBase64)
      : asTypedTensorData(type, data ?? []),
});

async function run(input: RunInput & { typedOutputs: true }): Promise<TypedRunResult>;
async function run(input: RunInput): Promise<RunResult>;
async function run(input: RunInput): Promise<RunResult | TypedRunResult> {
  if (!Capacitor.isNativePlatform()) return BridgedCapacitorOnnx.run(input);

  const result = (await BridgedCapacitorOnnx.run({
    ...input,
    inputs: toNativeTensors(input.inputs) as RunInput["inputs"],
  })) as unknown as { outputs: Record<string, NativeTensor>; latencyMs: number };
  if (!input.typedOutputs) return result as RunResult;

  return {
    latencyMs: result.latencyMs,
    outputs: Object.fromEntries(
      Object.entries(result.outputs).map(([name, tensor]) => [
        name,
        fromNativeTypedTensor(tensor),
      ]),
    ),
  };
}

/**
 * On Android/iOS, typed-array tensors are encoded for the bridge here (and typed outputs
 * decoded), so the native side gets their bytes instead of JSON numbers. The web plugin
 * takes them as they are.
 */
export const CapacitorOnnx: CapacitorOnnxPlugin = {
  loadModel: (input: LoadModelInput) =>
    BridgedCapacitorOnnx.loadModel(
      Capacitor.isNativePlatform() && input.warmupInputs
        ? {
            ...input,
            warmupInputs: toNativeTensors(
              input.warmupInputs,
            ) as LoadModelInput["warmupInputs"],
          }
        : input,
    ),
  run,
  release: (input) => BridgedCapacitorOnnx.release(input),
};
