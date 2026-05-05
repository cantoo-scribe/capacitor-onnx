# Testing Scripts

Quick reference to validate the plugin locally.

## Prerequisites

- Node.js
- pnpm (project uses `pnpm@10.30.3`)
- Android SDK (for Android build steps)

## Plugin package checks (repo root)

```bash
pnpm install
pnpm typecheck
pnpm build
```

## Example app checks

```bash
cd examples/host-app
pnpm install
pnpm build
```

Useful commands:

- `pnpm dev` (run Vite locally)
- `pnpm preview` (serve production build)

## Android validation (example app)

```bash
cd examples/host-app
pnpm cap:sync
pnpm android:assemble
```

## Full validation pipeline

```bash
cd examples/host-app
pnpm pipeline:validate
```

Runs:

1. `pnpm build`
2. `pnpm cap:sync`
3. `pnpm android:assemble`

## Quick troubleshooting

- Build not reflected in app: run `pnpm build` in root, then `pnpm cap:sync` in `examples/host-app`.
- Android/Gradle errors: verify Android SDK and build tools setup.
