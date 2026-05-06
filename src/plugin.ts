import { registerPlugin } from "@capacitor/core";

import type { CapacitorOnnxPlugin } from "./definitions";

const CapacitorOnnxNative = registerPlugin<CapacitorOnnxPlugin>("CapacitorOnnx", {
  web: () => import("./web/index").then((m) => new m.CapacitorOnnxWeb()),
});

export const CapacitorOnnx = {
  loadModel: CapacitorOnnxNative.loadModel.bind(CapacitorOnnxNative),
  run: CapacitorOnnxNative.run.bind(CapacitorOnnxNative),
  release: CapacitorOnnxNative.release.bind(CapacitorOnnxNative),
  clearModel: () => CapacitorOnnxNative.clearModel.bind(CapacitorOnnxNative),
  clearAllCache: () => CapacitorOnnxNative.clearAllCache.bind(CapacitorOnnxNative),
};
