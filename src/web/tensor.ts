import * as ort from "onnxruntime-web";
import type { InputTensor, RawTensor, TypedRawTensor, TypedTensorData } from "../definitions";
import { asTypedTensorData } from "../tensor-data";

/**
 * Converts a transport tensor into an ort.Tensor. The ort.Tensor constructor
 * does not accept number[] for every dtype: int64 requires BigInt64Array, and
 * bool/uint8 require typed arrays.
 */
export function toOrtTensor(raw: InputTensor): ort.Tensor {
  if (!Array.isArray(raw.data)) {
    const typed = asTypedTensorData(raw.type, raw.data);
    // Always a copy: the caller's array is never kept, and the proxy worker
    // (ort.env.wasm.proxy) transfers, so detaches, the input buffers.
    return typedOrtTensor(raw, typed === raw.data ? typed.slice() : typed);
  }

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

function typedOrtTensor(raw: InputTensor, data: TypedTensorData): ort.Tensor {
  switch (raw.type) {
    case "int64":
      return new ort.Tensor("int64", data as BigInt64Array, raw.dims);
    case "uint32":
      return new ort.Tensor("uint32", data as Uint32Array, raw.dims);
    case "int32":
      return new ort.Tensor("int32", data as Int32Array, raw.dims);
    case "uint8":
      return new ort.Tensor("uint8", data as Uint8Array, raw.dims);
    case "bool":
      return new ort.Tensor("bool", data as Uint8Array, raw.dims);
    case "float16":
      return new ort.Tensor("float16", data as Uint16Array, raw.dims);
    default:
      return new ort.Tensor("float32", data as Float32Array, raw.dims);
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

/** Converts an ort.Tensor result into a TypedRawTensor, keeping its typed array. */
export function fromOrtTensorTyped(tensor: ort.Tensor): TypedRawTensor {
  const type = tensor.type as TypedRawTensor["type"];
  const data = tensor.data as ArrayLike<number> | ArrayLike<bigint>;
  return {
    type,
    dims: tensor.dims,
    // float16 keeps its raw bits: ORT may hand a Float16Array where it exists.
    data:
      type === "float16" && !(data instanceof Uint16Array)
        ? new Uint16Array(
            (data as Uint16Array).buffer,
            (data as Uint16Array).byteOffset,
            data.length,
          )
        : asTypedTensorData(type, data),
  };
}
