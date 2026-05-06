import { registerPlugin } from "@capacitor/core";

import type { CapacitorOnnxPlugin } from "./definitions";

export const CapacitorOnnx = registerPlugin<CapacitorOnnxPlugin>("CapacitorOnnx", {
  web: () => import("./web/index").then((m) => new m.CapacitorOnnxWeb()),
});
