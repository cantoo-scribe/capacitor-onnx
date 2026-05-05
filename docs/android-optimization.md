# 10. Optimization and Hardware Acceleration (Android)

This document defines a practical strategy to improve Android performance with ONNX Runtime, balancing latency, stability, and per-device compatibility.

## 1) ONNX Runtime SessionOptions

### graphOptimizationLevel
- Recommendation: use a high level in production (ORT_ENABLE_ALL), especially for reused sessions.
- Advantage: operator fusion and graph simplification reduce inference latency.
- Trade-off: session creation may become slower; this is usually acceptable when the session is created in loadModel and reused in later inferences.

### intraOpNumThreads
- Defines parallelism within an operator.
- Initial recommendation:
  - Low-end: 1-2
  - Mid/high-end: 2-4
- Trade-off: more threads can improve throughput, but may worsen tail latency (p95) and increase thermal/battery usage.

### interOpNumThreads
- Defines parallelism between operators.
- Initial recommendation: 1 for most mobile models (smaller graphs).
- Trade-off: higher values may add overhead for small models.

### executionMode
- ORT_SEQUENTIAL: generally better for single-request inference on mobile.
- ORT_PARALLEL: can help for larger models with real graph parallelism.
- Recommendation: start with ORT_SEQUENTIAL and validate with per-device benchmarks.

### memory pattern
- Recommendation: enable for inputs with stable shapes.
- Trade-off: better performance for repeated inferences; less gain with dynamic shapes.

### CPU arena allocator
- Recommendation: keep enabled by default to reduce allocation overhead.
- Trade-off: may increase resident memory usage in some scenarios.

## 2) Execution Providers on Android

### CPUExecutionProvider
- Disponibilidade: sempre presente.
- Availability: always present.
- Role: reliable baseline and universal fallback.
- When to use: safe mode, maximum compatibility, debugging.

### NNAPIExecutionProvider
- Availability: depends on Android version + driver/vendor.
- Role: main hardware acceleration path (NPU/DSP/GPU through NNAPI).
- Trade-off: gains vary widely across manufacturers and supported operations.

### XNNPACKExecutionProvider (if applicable)
### XNNPACKExecutionProvider (if applicable)
- On Android, the most common path is ORT CPU EP with internal optimizations; XNNPACK depends on specific build/distribution.
- Recommendation: treat as optional and validate in the real artifact (AAR) before exposing as a public mode.

### QNNExecutionProvider (if applicable)
### QNNExecutionProvider (if applicable)
- Focused on Qualcomm hardware with a specific stack.
- Recommendation: consider only for dedicated distribution by manufacturer/controlled device fleet.
- Trade-off: higher operational complexity and compatibility matrix overhead.

## 3) GPU / NPU on Android: practical viability

- Direct GPU support on Android through ONNX Runtime is not the most portable path for broad-market apps.
- Main realistic path: NNAPI, which tries to map workloads to device accelerators.
- Important limitation: not every model/op runs on NNAPI.
- Expected behavior: when an op is unsupported, partial or total fallback to CPU occurs.
- Practical impact: some devices may get large gains, while others may get little gain (or regressions).
- Manufacturer compatibility:
  - Recent Qualcomm: trend toward better gains on compatible models.
  - Exynos/MediaTek/variants: more heterogeneous behavior.
  - Older devices: CPU fallback is often dominant.

## 4) Recommended configuration strategy

Define a configurable execution mode in the plugin/app:
- cpu
- nnapi
- auto

### Mode semantics
- cpu: force CPUExecutionProvider.
- nnapi: try NNAPI; if unavailable/invalid, explicit error or flag-controlled fallback.
- auto: try NNAPI first and fall back safely to CPU.

### Recommended parameters
- numThreads (mapped to intraOpNumThreads)
- interOpNumThreads
- graphOptimizationLevel
- executionMode
- enableMemoryPattern
- enableCpuMemArena

### Performance instrumentation (required)
Measure and record separately:
- download
- session creation
- pre-processing
- inference
- post-processing

Also record:
- effective provider used
- whether fallback occurred (yes/no)
- errors by op/provider

## 5) Model optimization

### Quantizacao INT8
- Advantage: major latency and memory reduction on CPU/NNAPI when well supported.
- Trade-off: can degrade accuracy; requires validation with real datasets.

### Float16 / mixed precision
- Advantage: possible gains on compatible accelerators.
- Trade-off: support varies by backend/device; gains are inconsistent on pure CPU.

### Graph optimization
- Apply in the export/conversion pipeline and keep ORT optimized at runtime.

### ORT format
- Converting ONNX to ORT format can reduce initialization cost and improve runtime.
- Trade-off: more complex build/deploy pipeline.

### Removing unnecessary outputs
- Reduces transfer and post-processing.
- Recommended for classification scenarios where only final top-k is needed in the app.

### Input size reduction
- Direct impact on latency and memory.
- Trade-off: risk of accuracy loss; validate the latency vs quality curve.

## 6) Per-device benchmark strategy

### Comparison matrix
Run per device and per model:
- CPU single-thread
- CPU multi-thread (2, 4)
- NNAPI
- XNNPACK (if available in the build)

### Metricas
- average latency
- p95
- memory (peak and approximate resident set)
- failures (crash, session error, unexpected fallback)
- cold start vs warm start

### Minimum protocol
- 5 warmup runs
- 30-100 measured runs per scenario
- battery > 40% and controlled temperature when possible
- no debugger attached

## Safe recommendation for production

1. Production default: auto mode with safe CPU fallback.
2. Session created once in loadModel and reused by modelId+version; never recreate per inference.
3. Execution off the main thread (already aligned with the current plugin).
4. Start with:
   - graphOptimizationLevel alto
   - interOpNumThreads = 1
  - intraOpNumThreads = 2 or 4 (tunable)
5. Required stage telemetry (download/session/pre/inference/post) and effective provider.
6. Progressive NNAPI rollout by device tier/manufacturer (remote feature flag).
7. Continuous benchmarking and regression tracking by model version and app version.

## ONNX Runtime usage (practical checklist)

- Utilizar ONNX Runtime Android.
- Configure multithreading.
- Configure graph optimization.
- Evaluate execution providers available in the Android artifact.
- Consider NNAPI as the main hardware acceleration path.
- Implement safe CPU fallback.
- Create session in loadModel and reuse it for all inferences of the same modelId+version.
- Run inference off the main thread.

## Current plugin state

- Current flow: loadModel already handles file download/cache and initializes the ONNX session.
- Current reuse: classifyImage reuses the same session by modelId+version (without recreating per call).
- Practical implication: initialization cost is concentrated in prepare/warmup, and inference stays focused on session run.
