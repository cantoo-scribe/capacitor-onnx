import { WebPlugin } from '@capacitor/core';

import type {
  CapacitorOnnxPlugin,
  ClearAllCacheResult,
  ClearModelInput,
  ClearModelResult,
  LoadModelInput,
  LoadModelResult,
  RunInput,
  RunResult
} from './definitions';

export class CapacitorOnnxWeb extends WebPlugin implements CapacitorOnnxPlugin {
  async loadModel(_input: LoadModelInput): Promise<LoadModelResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async run(_input: RunInput): Promise<RunResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async clearModel(_input: ClearModelInput): Promise<ClearModelResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async clearAllCache(): Promise<ClearAllCacheResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }
}
