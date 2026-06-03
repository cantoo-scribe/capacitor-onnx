#!/usr/bin/env node
// cli.mjs — cross-platform entrypoint for the reduced-ONNX installer.
//
// Exposed as the `cantoo-onnx-reduce` bin. It just locates the bash installer
// shipped next to it and runs it (the heavy lifting — toolchain checks, the ORT
// source build, gradle patching — needs a Unix shell). On Windows, run under WSL2.
import { execFileSync } from 'node:child_process'
import { dirname, join } from 'node:path'
import { fileURLToPath } from 'node:url'

const here = dirname(fileURLToPath(import.meta.url))
const installer = join(here, 'install.sh')

if (process.platform === 'win32') {
  console.error(
    'cantoo-onnx-reduce: this installer needs a Unix shell. On Windows, run it inside WSL2:\n' +
      `  wsl bash "${installer}"`,
  )
  process.exit(1)
}

try {
  execFileSync('bash', [installer, ...process.argv.slice(2)], { stdio: 'inherit' })
} catch (err) {
  process.exit(typeof err?.status === 'number' ? err.status : 1)
}
