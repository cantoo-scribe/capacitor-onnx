import type { RawTensor } from './definitions';

export interface GetInputTensorOptions {
  shape?: number[];
}

function parseInputData(input: ArrayLike<number>): number[] {
  const data = Array.from(input, (value) => Number(value));
  if (data.length === 0) {
    throw new Error('input data must not be empty');
  }
  if (data.some((value) => Number.isNaN(value) || !Number.isFinite(value))) {
    throw new Error('input data must contain only finite numeric values');
  }
  return data;
}

function parseShape(shape: number[] | undefined, fallbackLength: number): number[] {
  const resolvedShape = shape ?? [1, fallbackLength];
  if (resolvedShape.length === 0) {
    throw new Error('input tensor shape must not be empty');
  }
  if (resolvedShape.some((dim) => !Number.isInteger(dim) || dim <= 0)) {
    throw new Error('input tensor shape must contain only positive integers');
  }

  const expectedLength = resolvedShape.reduce((acc, current) => acc * current, 1);
  if (expectedLength !== fallbackLength) {
    throw new Error('input tensor shape product must match input data length');
  }

  return resolvedShape;
}

export function getInputTensor(input: ArrayLike<number>, options?: GetInputTensorOptions): RawTensor {
  const data = parseInputData(input);
  const shape = parseShape(options?.shape, data.length);
  return {
    data,
    shape,
    type: 'float32',
  };
}
