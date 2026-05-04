import { WebPlugin } from '@capacitor/core';

import type {
  CapacitorOnnxPlugin,
  ClearAllCacheResult,
  ClearModelInput,
  ClearModelResult,
  DiagnosticsResult,
  GetModelStatusInput,
  GetModelStatusResult,
  IsActiveResult,
  PrepareModelInput,
  PrepareModelResult,
  RunInferenceInput,
  RunInferenceResult,
  WarmupModelInput,
  WarmupModelResult,
} from './definitions';

export class CapacitorOnnxWeb extends WebPlugin implements CapacitorOnnxPlugin {
  async isActive(): Promise<IsActiveResult> {
    return { value: false };
  }

  async prepareModel(_input: PrepareModelInput): Promise<PrepareModelResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async warmupModel(_input: WarmupModelInput): Promise<WarmupModelResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async runInference(_input: RunInferenceInput): Promise<RunInferenceResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async getModelStatus(_input: GetModelStatusInput): Promise<GetModelStatusResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async clearModel(_input: ClearModelInput): Promise<ClearModelResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async clearAllCache(): Promise<ClearAllCacheResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }

  async getDiagnostics(): Promise<DiagnosticsResult> {
    throw new Error('CapacitorOnnx is only available on Android in this version.');
  }
}
