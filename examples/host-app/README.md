# Host App (Smoke Test)

Host app minimo para validar integracao do plugin `@cantoo/capacitor-onnx` com Android em ambiente real.

## Pre-requisitos

- Node.js 18+
- pnpm 10+
- JDK 21
- Android SDK configurado

## Fluxo rapido

1. Instalar dependencias:

```bash
pnpm install
```

2. Build web:

```bash
pnpm build
```

3. Adicionar Android (uma vez):

```bash
pnpm cap:add:android
```

4. Sincronizar plugin/assets:

```bash
pnpm cap:sync
```

5. Build Android:

```bash
pnpm android:assemble
```

6. Pipeline completa (apos Android existir):

```bash
pnpm pipeline:validate
```

## Troubleshooting rapido

- Falha de build Android por SDK/NDK ausente:
	- Verifique variaveis de ambiente e configuracao do Android SDK no host.
	- Rode `pnpm cap:sync` novamente apos ajustar o ambiente.

## Smoke test funcional

- Botao **Get diagnostics**: valida bridge com chamada simples.
- Campo **Quick preset** + botao **Apply preset**:
	- Preenche rapidamente `modelId`, `version` e `normalized input`.
	- Mantem `url` e `sha256` (opcional) para voce informar os valores do modelo.
	- Inclui presets curtos e longos de audio para acelerar testes manuais de inferencia.
- Botao **Generate mock audio**:
	- Gera sinal de audio sintetico (senoidal com harmonico) e preenche automaticamente o campo `Normalized input (CSV)`.
	- Permite testar inferencia sem colar manualmente milhares de amostras.
- Botao **Run success E2E**:
	- Usa os campos de configuracao (modelId, version, url, sha256 opcional, normalized input).
	- Converte automaticamente a entrada para tensor via helper `getInputTensor(...)`.
	- Executa sequencia `prepareModel -> warmupModel -> runInference`.
	- Valida assertions minimas de contrato (`sessionReady`, `warmed`, `logits.type`, consistencia shape/data, latencia numerica).
- Botao **Run error E2E**:
	- Forca `runInference` com modelo ausente.
	- Valida contrato de erro estruturado (`code`, `message`, `retryable`, `correlationId`).
- Botao **Clear model cache**:
	- Limpa cache do modelo atual (`modelId` + `version`) no dispositivo.
- Botao **Clear all cache**:
	- Limpa todos os modelos preparados no dispositivo.

## Protecao de chamadas concorrentes

- Enquanto `prepareModel` esta executando, os botoes de acao ficam bloqueados.
- Chamadas concorrentes para o mesmo carregamento de modelo aguardam a primeira (deduplicacao por chave de modelo/versao/url/hash).

## Observacao sobre sucesso E2E

Para o fluxo de sucesso, e necessario informar um `url` real de um modelo ONNX compativel com o tensor de entrada configurado. O `sha256` pode ser omitido temporariamente.
