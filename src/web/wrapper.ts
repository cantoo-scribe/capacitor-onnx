import { WebPlugin } from "@capacitor/core";

import type {
  CapacitorOnnxPlugin,
  LoadModelInput,
  LoadModelResult,
  ReleaseModelInput,
  RunInput,
  RunResult,
  TypedRunResult,
} from "../definitions";
import { CapacitorOnnxWeb } from "./index";

export class WrapperCapacitorOnnxWeb extends WebPlugin implements CapacitorOnnxPlugin {
  private readonly engine = new CapacitorOnnxWeb();

  async loadModel(input: LoadModelInput): Promise<LoadModelResult> {
    return await this.engine.loadModel(input);
  }

  async run(input: RunInput & { typedOutputs: true }): Promise<TypedRunResult>;
  async run(input: RunInput): Promise<RunResult>;
  async run(input: RunInput): Promise<RunResult | TypedRunResult> {
    return await this.engine.run(input);
  }

  async release(input: ReleaseModelInput): Promise<void> {
    await this.engine.release(input);
  }
}
