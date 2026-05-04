# Architectural Decisions

## 2026-05-04 - Initial plugin baseline
- Decision: Start with high-level JS API (`prepareModel`, `classifyImage`, cache and diagnostics methods) instead of generic tensor bridge.
- Rationale: Reduce bridge overhead, avoid large payload transfers, and keep Web API simple.
- Consequence: Lower flexibility in V1, but better performance and operational safety.

## 2026-05-04 - Session concurrency strategy
- Decision: Reuse ONNX sessions and apply per-session serialization (mutex), allowing parallelism across different model sessions.
- Rationale: Improves predictability and thread safety while retaining horizontal concurrency.
- Consequence: Single model version requests are queued; multi-model workloads can scale better.

## 2026-05-04 - Model integrity and cache promotion
- Decision: Download to temporary file, validate SHA-256, then atomically promote to final cache file.
- Rationale: Prevent corrupted or partial artifacts from being used.
- Consequence: Slightly more IO steps but robust consistency.

## 2026-05-04 - Native image inference pipeline in V1
- Decision: Implement image inference in Android (URI/path decode, resize, RGB normalization to NCHW, ONNX run), sem decodificação de logits no plugin.
- Rationale: Keep bridge payloads small and avoid Float32 tensor transfers between JS and Kotlin.
- Consequence: Plugin retorna apenas logits brutos (data/shape/type); labels, topK, argmax e demais regras de negócio ficam na aplicação.

## 2026-05-04 - Raw logits as plugin contract
- Decision: O contrato de inferência exposto ao JS retorna exclusivamente logits puros.
- Rationale: Separar engine de inferência (plugin) da lógica de interpretação de saída (aplicação), reduzindo acoplamento e facilitando evolução por modelo.
- Consequence: O app consumidor deve implementar o pós-processamento adequado para cada modelo.

## 2026-05-04 - Raw tensor input as plugin contract
- Decision: O plugin recebe tensor de entrada já pré-processado (`inputTensor`) em vez de URI de imagem.
- Rationale: Remover pré-processamento específico de domínio do plugin e manter a camada nativa focada em execução ONNX.
- Consequence: A aplicação cliente passa a ser responsável por decode/resize/normalização e por garantir shape/tipo compatíveis com o modelo.

## 2026-05-04 - Structured error envelope for JS
- Decision: Padronizar todas as falhas expostas pela bridge com envelope estruturado (`code`, `message`, `retryable`, `correlationId`, `details`).
- Rationale: Facilitar tratamento consistente de erro no app, observabilidade e políticas de retry por categoria.
- Consequence: Mensagens de erro internas continuam úteis para diagnóstico, mas o contrato oficial de consumo passa a depender prioritariamente de `code`.

## 2026-05-04 - Temporary optional SHA-256 in prepareModel
- Decision: Tornar `sha256` opcional temporariamente em `prepareModel` e pular verificação de integridade quando ausente.
- Rationale: Os modelos atuais ainda não possuem metadata JSON com hash confiável para todos os artefatos.
- Consequence: Mantém velocidade de integração no curto prazo, com menor garantia de integridade até a metadata ser disponibilizada.

## 2026-05-04 - High-level TS helper for tensor ergonomics
- Decision: Manter o contrato nativo baseado em `inputTensor` no plugin e adicionar helper de conveniência no SDK TS (`getInputTensor(...)`) para construir tensor automaticamente a partir de entrada normalizada.
- Rationale: Preservar baixo acoplamento e foco do plugin em execução ONNX, sem perder ergonomia para uso equivalente ao fluxo atual com onnxruntime-web.
- Consequence: O app consumidor pode operar em nível alto (dados normalizados) enquanto o contrato de bridge continua explícito, estável e compatível com diferentes domínios (audio, vision, etc.).

## 2026-05-04 - Android runtime optimization strategy
- Decision: Adotar configuração de execução flexível (`cpu`, `nnapi`, `auto`) com fallback seguro para CPU, sessões reutilizadas e telemetria por etapa (download, criação de sessão, pré, inferência, pós).
- Rationale: Maximizar performance sem comprometer estabilidade diante da heterogeneidade de drivers/providers no Android.
- Consequence: O app ganha aceleração por hardware quando viável (NNAPI), mantendo previsibilidade operacional com fallback controlado e benchmark contínuo por dispositivo.

## 2026-05-04 - Session options exposed in prepareModel
- Decision: Expor `sessionOptions` no `prepareModel` com `executionProvider` (`cpu`/`nnapi`/`auto`) e knobs simples de threads (`intraOpNumThreads`, `interOpNumThreads`).
- Rationale: Permitir ajuste de performance sem aumentar complexidade da API de inferência.
- Consequence: A sessão continua sendo criada uma vez e reutilizada por `modelId+version`, com fallback automático para CPU no modo `auto`.

## 2026-05-05 - Package export strategy as ESM-only
- Decision: Publicar o SDK como ESM-only com `type: module` e `exports` explícito para o entrypoint principal.
- Rationale: O build atual já é ESM, o ecossistema alvo (Capacitor + Vite) é ESM-first e isso reduz complexidade operacional versus dual build (CJS + ESM).
- Consequence: Consumidores devem usar `import` (não `require`), e a superfície pública do pacote passa a ser controlada por `exports`.

## 2026-05-05 - Host/iFrame postMessage bridge in SDK
- Decision: Expor utilitários TS para comunicação entre Host e iFrame (`createHostBridge` e `createIFrameBridge`) com envelope de mensagem padronizado por canal.
- Rationale: Atender integração web com contrato mínimo, tipado e reutilizável, sem acoplamento ao runtime nativo.
- Consequence: Apps Host e iFrame conseguem emitir/reagir a eventos via API comum, com filtros de origem (`targetOrigin`/`allowedOrigins`) para segurança básica.

## 2026-05-05 - Async request protocol for Host/iFrame operations
- Decision: Padronizar as operações de bridge em protocolo assíncrono com `requestId` e ciclo `requested/result/error`, implementado por classes (`OnnxIFrameClient` e `OnnxHostDispatcher`).
- Rationale: Garantir semântica consistente de request/response para métodos críticos (`isActive`, `prepareModel`, `warmupModel`, `classifyImage`) e permitir API baseada em Promise no iFrame.
- Consequence: O iFrame passa a aguardar resultado de forma determinística, e o Host centraliza despacho para nativo com chaves de evento estáveis.

## 2026-05-05 - RunInference-only audio-first contract
- Decision: Remover `classifyImage` da API pública e adotar `runInference` como único método de inferência no SDK, bridge Host/iFrame e plugin Android.
- Rationale: O domínio principal do produto é áudio, e nomenclatura orientada a imagem gerava ambiguidade de contrato.
- Consequence: Consumidores devem migrar chamadas para `runInference`; o contrato fica semântico para áudio e permanece neutro para outros domínios baseados em tensor.
