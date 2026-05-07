import { WebPlugin } from "@capacitor/core";

import type {
  CapacitorOnnxPlugin,
  ClearAllCacheResult,
  ClearModelInput,
  ClearModelResult,
  LoadModelInput,
  LoadModelResult,
  RunInput,
  RunResult,
} from "../definitions";
import { CapacitorOnnxWeb } from "./index";

export class WrapperCapacitorOnnxWeb extends WebPlugin implements CapacitorOnnxPlugin {
  private readonly engine = new CapacitorOnnxWeb();

  async loadModel(input: LoadModelInput): Promise<LoadModelResult> {
    return await this.engine.loadModel(input);
  }

  async run(input: RunInput): Promise<RunResult> {
    return await this.engine.run(input);
  }

  async release(input: ClearModelInput): Promise<void> {
    await this.engine.release(input);
  }

  async clearModel(input: ClearModelInput): Promise<ClearModelResult> {
    return await this.engine.clearModel(input);
  }

  async clearAllCache(): Promise<ClearAllCacheResult> {
    return await this.engine.clearAllCache();
  }
}
