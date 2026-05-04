# 10. Otimizacao e aceleracao por hardware (Android)

Este documento define uma estrategia pratica para melhorar performance no Android com ONNX Runtime, equilibrando latencia, estabilidade e compatibilidade por dispositivo.

## 1) ONNX Runtime SessionOptions

### graphOptimizationLevel
- Recomendacao: usar nivel alto em producao (ORT_ENABLE_ALL), especialmente para sessao reutilizada.
- Vantagem: fusao de operadores e simplificacao do grafo reduzem latencia de inferencia.
- Trade-off: criacao da sessao pode ficar mais lenta; impacto geralmente aceitavel quando a sessao e criada no prepareModel e reutilizada nas inferencias seguintes.

### intraOpNumThreads
- Define paralelismo dentro de um operador.
- Recomendacao inicial:
  - Low-end: 1-2
  - Mid/high-end: 2-4
- Trade-off: mais threads pode melhorar throughput, mas piorar tail latency (p95) e aumentar consumo termico/bateria.

### interOpNumThreads
- Define paralelismo entre operadores.
- Recomendacao inicial: 1 para a maioria dos modelos mobile (grafos menores).
- Trade-off: valores maiores podem causar overhead em modelos pequenos.

### executionMode
- ORT_SEQUENTIAL: geralmente melhor para inferencia single-request em mobile.
- ORT_PARALLEL: pode ajudar em modelos maiores com grafo paralelo real.
- Recomendacao: comecar com ORT_SEQUENTIAL e validar em benchmark por device.

### memory pattern
- Recomendacao: habilitar para entradas com shape estavel.
- Trade-off: melhor performance em inferencias repetidas; menos ganho com shapes dinamicos.

### CPU arena allocator
- Recomendacao: manter habilitado por padrao para reduzir overhead de alocacao.
- Trade-off: pode aumentar uso de memoria residente em alguns cenarios.

## 2) Execution Providers no Android

### CPUExecutionProvider
- Disponibilidade: sempre presente.
- Papel: baseline confiavel e fallback universal.
- Quando usar: modo seguro, compatibilidade maxima, debug.

### NNAPIExecutionProvider
- Disponibilidade: depende de Android version + driver/vendor.
- Papel: principal caminho de aceleracao por hardware (NPU/DSP/GPU via NNAPI).
- Trade-off: ganhos variam muito entre fabricantes e operacoes suportadas.

### XNNPACKExecutionProvider (se aplicavel)
- Em Android, o caminho mais comum e ORT CPU EP com otimizacoes internas; XNNPACK depende de build/distribuicao especifica.
- Recomendacao: tratar como opcional e validar no artefato real (AAR) antes de expor como modo publico.

### QNNExecutionProvider (se aplicavel)
- Focado em hardware Qualcomm com stack especifica.
- Recomendacao: considerar apenas em distribuicao dedicada por fabricante/parque controlado.
- Trade-off: maior complexidade operacional e matriz de compatibilidade.

## 3) GPU / NPU no Android: viabilidade real

- Suporte GPU direto no Android via ONNX Runtime nao e o caminho mais portavel para app de mercado amplo.
- Caminho principal realista: NNAPI, que tenta mapear para aceleradores do dispositivo.
- Limitacao importante: nem todo modelo/op roda em NNAPI.
- Comportamento esperado: quando op nao suportado, ocorre fallback parcial/total para CPU.
- Impacto pratico: pode haver ganho alto em alguns devices e quase nenhum (ou regressao) em outros.
- Compatibilidade por fabricante:
  - Qualcomm recentes: tendencia a ganhos melhores em modelos compatveis.
  - Exynos/MediaTek/variantes: comportamento mais heterogeneo.
  - Devices antigos: fallback para CPU frequentemente dominante.

## 4) Estrategia recomendada de configuracao

Definir modo de execucao configuravel no plugin/app:
- cpu
- nnapi
- auto

### Semantica dos modos
- cpu: forca CPUExecutionProvider.
- nnapi: tenta NNAPI; se indisponivel/invalido, erro explicito ou fallback controlado por flag.
- auto: tenta NNAPI primeiro e cai para CPU com seguranca.

### Parametros recomendados
- numThreads (mapeado para intraOpNumThreads)
- interOpNumThreads
- graphOptimizationLevel
- executionMode
- enableMemoryPattern
- enableCpuMemArena

### Instrumentacao de performance (obrigatoria)
Medir e registrar separadamente:
- download
- criacao da sessao
- pre-processamento
- inferencia
- pos-processamento

Registrar tambem:
- provider efetivo usado
- fallback ocorrido (sim/nao)
- erro por op/provider

## 5) Otimizacao do modelo

### Quantizacao INT8
- Vantagem: grande reducao de latencia e memoria em CPU/NNAPI quando bem suportado.
- Trade-off: pode degradar acuracia; exige validacao por dataset real.

### Float16 / mixed precision
- Vantagem: possivel ganho em aceleradores compativeis.
- Trade-off: suporte varia por backend/device; ganhos inconsistentes em CPU pura.

### Graph optimization
- Aplicar no pipeline de export/conversao e manter ORT otimizado em runtime.

### ORT format
- Converter ONNX para ORT format pode reduzir custo de inicializacao e melhorar runtime.
- Trade-off: pipeline de build/deploy mais complexo.

### Remocao de outputs desnecessarios
- Reduz transferencia e pos-processamento.
- Recomendado para cenarios de classificacao onde apenas top-k final e necessario no app.

### Reducao de input size
- Impacto direto em latencia e memoria.
- Trade-off: risco de perda de acuracia; validar curva latencia x qualidade.

## 6) Estrategia de benchmark por device

### Matriz de comparacao
Executar por dispositivo e por modelo:
- CPU single-thread
- CPU multi-thread (2, 4)
- NNAPI
- XNNPACK (se disponivel na build)

### Metricas
- latencia media
- p95
- memoria (pico e resident set aproximado)
- falhas (crash, erro de sessao, fallback inesperado)
- cold start vs warm start

### Protocolo minimo
- 5 execucoes de warmup
- 30-100 execucoes medidas por cenario
- bateria > 40% e temperatura controlada quando possivel
- sem debugger anexado

## Recomendacao segura para producao

1. Padrao em producao: modo auto com fallback seguro para CPU.
2. Sessao criada uma vez no prepareModel e reutilizada por modelId+version; nunca recriar por inferencia.
3. Execucao fora da main thread (ja alinhado com o plugin atual).
4. Comecar com:
   - graphOptimizationLevel alto
   - interOpNumThreads = 1
   - intraOpNumThreads = 2 ou 4 (ajustavel)
5. Telemetria obrigatoria de etapa (download/sessao/pre/inferencia/pos) e provider efetivo.
6. Rollout progressivo de NNAPI por device tier/fabricante (feature flag remota).
7. Benchmark continuo e regressao por versao de modelo e versao de app.

## Uso do ONNX Runtime (checklist pratico)

- Utilizar ONNX Runtime Android.
- Configurar multithreading.
- Configurar graph optimization.
- Avaliar execution providers disponiveis no artefato Android.
- Considerar NNAPI como caminho principal de aceleracao por hardware.
- Implementar fallback seguro para CPU.
- Criar sessao no prepareModel e reutilizar em todas as inferencias do mesmo modelId+version.
- Executar inferencia fora da main thread.

## Estado atual do plugin

- Fluxo atual: prepareModel ja resolve download/cache do arquivo e inicializa a sessao ONNX.
- Reuso atual: classifyImage reutiliza a mesma sessao por modelId+version (sem recriar por chamada).
- Implicacao pratica: o custo de inicializacao concentra no prepare/warmup, e a inferencia fica focada no run da sessao.
