import type { TensorType, TypedTensorData } from "./definitions";

/**
 * Converts tensor data to the typed array of its type (see `TypedTensorData`). Data already
 * of that typed array is returned as is (not copied); bool data is normalized to 0/1.
 */
export function asTypedTensorData(
  type: TensorType,
  data: ArrayLike<number> | ArrayLike<bigint>,
): TypedTensorData {
  switch (type) {
    case "float32":
      return data instanceof Float32Array
        ? data
        : Float32Array.from(data as ArrayLike<number>, Number);
    case "float16":
      return data instanceof Uint16Array
        ? data
        : Uint16Array.from(data as ArrayLike<number>, Number);
    case "int32":
      return data instanceof Int32Array
        ? data
        : Int32Array.from(data as ArrayLike<number>, Number);
    case "int64":
      return data instanceof BigInt64Array
        ? data
        : BigInt64Array.from(data as ArrayLike<number>, (value) => BigInt(value));
    case "uint32":
      return data instanceof Uint32Array
        ? data
        : Uint32Array.from(data as ArrayLike<number>, Number);
    case "uint8":
      return data instanceof Uint8Array
        ? data
        : Uint8Array.from(data as ArrayLike<number>, Number);
    case "bool":
      return Uint8Array.from(data as ArrayLike<number>, (value) => (value ? 1 : 0));
  }
}

const BASE64_CHUNK = 0x8000;

/** The bytes of a typed array (platform byte order, little-endian on every target). */
export function typedArrayToBase64(data: TypedTensorData): string {
  const bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
  let binary = "";
  for (let start = 0; start < bytes.length; start += BASE64_CHUNK) {
    binary += String.fromCharCode.apply(
      null,
      bytes.subarray(start, start + BASE64_CHUNK) as unknown as number[],
    );
  }
  return btoa(binary);
}

/** The typed array of `type` whose bytes `base64` encodes (see `typedArrayToBase64`). */
export function base64ToTypedArray(type: TensorType, base64: string): TypedTensorData {
  const binary = atob(base64);
  const bytes = new Uint8Array(binary.length);
  for (let index = 0; index < binary.length; index++) {
    bytes[index] = binary.charCodeAt(index);
  }

  const { buffer } = bytes;
  const of = <T>(bytesPerElement: number, view: (length: number) => T): T => {
    if (bytes.length % bytesPerElement !== 0) {
      throw new Error(
        `${bytes.length} bytes do not hold whole ${type} elements of ${bytesPerElement} bytes`,
      );
    }
    return view(bytes.length / bytesPerElement);
  };

  switch (type) {
    case "float32":
      return of(4, (length) => new Float32Array(buffer, 0, length));
    case "float16":
      return of(2, (length) => new Uint16Array(buffer, 0, length));
    case "int32":
      return of(4, (length) => new Int32Array(buffer, 0, length));
    case "int64":
      return of(8, (length) => new BigInt64Array(buffer, 0, length));
    case "uint32":
      return of(4, (length) => new Uint32Array(buffer, 0, length));
    case "uint8":
    case "bool":
      return bytes;
  }
}
