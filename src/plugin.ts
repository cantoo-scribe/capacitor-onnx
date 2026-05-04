import { registerPlugin } from '@capacitor/core';

import type { CapacitorOnnxPlugin } from './definitions';
import { getInputTensor, GetInputTensorOptions } from './runtime';

const CapacitorOnnxNative = registerPlugin<CapacitorOnnxPlugin>('CapacitorOnnx', {
  web: () => import('./web').then((m) => new m.CapacitorOnnxWeb()),
});

export const CapacitorOnnx = {
  native: CapacitorOnnxNative,
  prepareModel: CapacitorOnnxNative.prepareModel.bind(CapacitorOnnxNative),
  warmupModel: CapacitorOnnxNative.warmupModel.bind(CapacitorOnnxNative),
  getInputTensor: (input: ArrayLike<number>, options?: GetInputTensorOptions) => {
    const opt = options ?? {};
    const shape = opt.shape ?? [1, input.length];
    return getInputTensor(input, { ...opt, shape });
  },
  runInference: CapacitorOnnxNative.runInference.bind(CapacitorOnnxNative),
  getModelStatus: CapacitorOnnxNative.getModelStatus.bind(CapacitorOnnxNative),
  clearModel: CapacitorOnnxNative.clearModel.bind(CapacitorOnnxNative),
  clearAllCache: CapacitorOnnxNative.clearAllCache.bind(CapacitorOnnxNative),
  getDiagnostics: CapacitorOnnxNative.getDiagnostics.bind(CapacitorOnnxNative),
  isActive: () => CapacitorOnnxNative.isActive().then(v => v.value).catch(() => false),
};