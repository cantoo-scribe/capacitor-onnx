export function nowMs(): number {
  return typeof performance !== "undefined" && typeof performance.now === "function"
    ? performance.now()
    : Date.now();
}

export function elapsedMs(startTime: number): number {
  return Math.max(0, Math.round(nowMs() - startTime));
}
