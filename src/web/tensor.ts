import * as ort from "onnxruntime-web";
import type { RawTensor } from "../definitions";

/**
 * Converts a transport RawTensor into an ort.Tensor. The ort.Tensor constructor
 * does not accept number[] for every dtype: int64 requires BigInt64Array, and
 * bool/uint8 require typed arrays.
 */
export function toOrtTensor(raw: RawTensor): ort.Tensor {
  switch (raw.type) {
    case "int64":
      return new ort.Tensor("int64", BigInt64Array.from(raw.data, BigInt), raw.dims);
    case "uint32":
      return new ort.Tensor("uint32", Uint32Array.from(raw.data), raw.dims);
    case "int32":
      return new ort.Tensor("int32", Int32Array.from(raw.data), raw.dims);
    case "uint8":
      return new ort.Tensor("uint8", Uint8Array.from(raw.data), raw.dims);
    case "bool":
      return new ort.Tensor("bool", Uint8Array.from(raw.data, (v) => (v ? 1 : 0)), raw.dims);
    case "float16":
      return new ort.Tensor("float16", Uint16Array.from(raw.data), raw.dims);
    default:
      return new ort.Tensor("float32", Float32Array.from(raw.data), raw.dims);
  }
}

/** Converts an ort.Tensor result into a transport RawTensor. Number() handles int64 (bigint). */
export function fromOrtTensor(tensor: ort.Tensor): RawTensor {
  return {
    type: tensor.type as RawTensor["type"],
    dims: tensor.dims,
    data: Array.from(tensor.data as ArrayLike<number | bigint>, Number),
  };
}
