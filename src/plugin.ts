import { registerPlugin } from '@capacitor/core';

import type { CapacitorOnnxPlugin } from './definitions';

const CapacitorOnnxNative = registerPlugin<CapacitorOnnxPlugin>('CapacitorOnnx', {
  web: () => import('./web').then((m) => new m.CapacitorOnnxWeb()),
});

export const CapacitorOnnx = {
  loadModel: CapacitorOnnxNative.loadModel.bind(CapacitorOnnxNative),
  run: CapacitorOnnxNative.run.bind(CapacitorOnnxNative),
  clear: () => CapacitorOnnxNative.clearModel.bind(CapacitorOnnxNative),
  clearAllCache: () => CapacitorOnnxNative.clearAllCache.bind(CapacitorOnnxNative)
};