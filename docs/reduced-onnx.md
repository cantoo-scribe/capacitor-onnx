# Reduced ONNX Runtime build (Android, opt-in)

By default this plugin ships the **full** Microsoft `onnxruntime-android` AAR and your
app loads a standard `.onnx` model — works on **Android, iOS and Web**, no extra tooling.

This document covers an **opt-in, Android-only** alternative that shrinks the native
runtime: it compiles a `libonnxruntime.so` containing **only the operators your model
uses** and pairs it with a pre-optimized **`.ort`** model. On a typical transformer the
arm64 `.so` drops from **~9.9 MB → ~4.2 MB compressed** (~27 MB → ~11 MB uncompressed),
~**57–59% smaller**, with no loss of functionality for that model.

It is fully **opt-out**: if you don't run the command (or clear `onnxModel`), the app
keeps the full AAR and a plain `.onnx` — nothing breaks.

---

## The two modes

| | **Default — `.onnx`** | **Reduced — `.ort`** |
| --- | --- | --- |
| Platforms | Android, iOS, Web | **Android only** |
| Native runtime | full `onnxruntime-android` (~1500 ops) | reduced `.so` (your model's ops) |
| Model file the app loads | `.onnx` | **`.ort`** (Runtime-style) |
| Extra tooling | none | Python + NDK + CMake/Ninja (first build) |
| Setup | nothing | run `cantoo-onnx-reduce` once |

> **Why `.ort` and not `.onnx`?** A reduced `.so` only has kernels for the operators in
> the op-config. If the app loads a raw `.onnx`, ORT **re-optimizes the graph at runtime
> per platform** — on a real arm device that can introduce operators the host-side config
> never saw (e.g. `ReduceL2` → `ReduceSum`), which the reduced `.so` then lacks
> (`ORT_NOT_IMPLEMENTED`). A **Runtime-style `.ort`** has its optimizations baked/serialized
> and replayed at load, so the runtime op-set is fixed by the file and matches the `.so`.

## Prerequisites (build machine)

Only needed to **produce** the reduced artifacts (typically once, or on CI):

- **bash** — macOS/Linux directly; on Windows use **WSL2**.
- **Python ≥ 3.10** with `onnxruntime==<plugin's ORT version>` and `onnx`
  (the command can create a `.venv` for you).
- **Android NDK** `28.0.13004108` (LTS, pinned for ORT 1.25.x) + **Android SDK**.
- **CMake** and **Ninja**.
- **JDK 21** (to run the consuming app's Gradle build).

The ORT version is **derived** from the plugin's own `android/build.gradle`
(`onnxruntime-android:<version>`), so it always matches — no manual pinning.

## Usage

From the consumer app (the package that depends on `@cantoo/capacitor-onnx`, so the
bin is already in `node_modules/.bin`):

```bash
pnpm exec cantoo-onnx-reduce     # or: npx cantoo-onnx-reduce
```

(Don't use `pnpm dlx cantoo-onnx-reduce` — `dlx` reads the argument as a *package name*,
and the package is `@cantoo/capacitor-onnx`, not `cantoo-onnx-reduce`. To run it ad-hoc
without installing: `pnpm dlx --package @cantoo/capacitor-onnx cantoo-onnx-reduce`.)

It is interactive and prompts for:

- **Android project directory** (autodetects `./android`).
- **Model URL or local path** (`.onnx`) — the build input. A URL is downloaded+cached;
  a local path is used directly (absolute is safest).
- **ORT version** (defaults to the plugin's).
- **Target ABIs** (default `arm64-v8a,armeabi-v7a`).
- **Python interpreter** (offers to create a `.venv`).
- **Remote AAR cache URL** (optional, read-only GET).
- **Remote AAR upload target** (optional rsync/ssh).
- **Remote `.ort` upload target** (optional rsync/ssh file).

### What it changes in your app

1. Copies the mechanism into `android/onnx/`:
   - `onnx-reduce.gradle` — resolves/produces the reduced AAR at Gradle configuration time.
   - `build-reduced-onnx.sh` — compiles the reduced ORT from source on a full cache miss.
2. Writes a managed block to `android/gradle.properties` (between
   `# >>> cantoo onnx-reduce >>>` markers).
3. Patches `app/build.gradle`: `apply from: '../onnx/onnx-reduce.gradle'`, excludes the
   full `onnxruntime-android`, links the reduced AAR, and injects `ndk { abiFilters … }`.

### `gradle.properties` keys

| Key | Meaning |
| --- | --- |
| `onnxModel` | `.onnx` URL or local path (the build input). Empty = opt-out (full AAR). |
| `onnxOrtVersion` | ORT version to build/convert with (must match the plugin's dep). |
| `onnxCacheUrl` | Read-only remote AAR cache (GET by content hash). Empty = disabled. |
| `onnxCacheUploadUrl` | rsync/ssh target to publish the built AAR to the cache. Empty = no publish. |
| `onnxOrtUploadUrl` | rsync/ssh **file** target to publish the `.ort` model. Empty = no publish. |
| `onnxConfigUrl` | Read-only GET URL for the op-config. Set it so **consumers skip Python** on a cache hit (see below). Empty = always regenerate locally. |
| `onnxConfigUploadUrl` | rsync/ssh **file** target to publish the op-config (the generator publishes it for consumers). Empty = no publish. |
| `onnxPython` | Python interpreter with `onnxruntime`+`onnx` (e.g. a `.venv`). Only the **generator** needs it. |

## How the build resolves the AAR

The AAR cache hash is `op-config ∪ ORT version ∪ ABIs ∪ nnapi`, so the **op-config** is the
key everything hinges on. At Gradle configuration time `resolveReducedAar()`:

**First, get the op-config (this is what decides whether you need Python):**
1. **Local** — `android/.cache/onnx-reduced/configs/<modelKey>.config` (reused if present).
2. **Remote (consumer fast-path)** — `GET onnxConfigUrl` (if set). Downloading it means the
   build can compute the hash **without Python and without downloading the ~122 MB model**.
3. **Generate (generator path)** — only if neither above: download the model and run the ORT
   Python tooling. This is the **only** step that needs `onnxPython`. The generator then
   publishes the op-config to `onnxConfigUploadUrl` and the `.ort` to `onnxOrtUploadUrl`
   (best-effort) so the next person hits step 2.

**Then, resolve the AAR by hash:**
1. **Local cache** — `android/.cache/onnx-reduced/<hash>/onnxruntime-android.aar`.
2. **Remote cache** — `GET <onnxCacheUrl>/<hash>/onnxruntime-android.aar` (if set).
3. **Build from source** — checks out ORT at the pinned tag and compiles the reduced `.so`
   (~30–60 min, **once** per hash; needs only the op-config, not the model), then auto-uploads
   to `onnxCacheUploadUrl` if set.

> **Generator vs consumer.** The *generator* (whoever has Python + upload creds) runs a build
> once to populate the op-config, `.ort` and AAR on the server. Everyone else is a *consumer*:
> with `onnxConfigUrl` + `onnxCacheUrl` set, their build downloads the op-config, computes the
> hash, and pulls the prebuilt AAR — **no Python, no NDK, no model download**. Leave
> `onnxConfigUrl` empty to keep the old behavior (always regenerate the config via Python).

NNAPI is always compiled into the `.so` (the official JNI glue requires the
`OrtSessionOptionsAppendExecutionProvider_Nnapi` symbol); it is only *used* if a session
opts in. All caches live under `android/.cache/` (git-ignored, survives `gradlew clean`).

## App-side wiring (consumer responsibility)

The command sets up the **native build**; loading the `.ort` is your app's code:

- Host/serve the produced `.ort` (the build can publish it via `onnxOrtUploadUrl`), and
  download it on device. **The local filename must end in `.ort`** — native ORT detects
  the format by extension on a path load; a `.onnx` suffix makes it try the protobuf parser.
- Pass that path to `loadModel`. Keep the default optimization level (the plugin uses
  `ALL_OPT`, which is correct for a Runtime-style `.ort`).
- Bump your model `version` when switching `.onnx` → `.ort` so devices re-download and the
  session cache (`modelId+version`) is invalidated.

The `.ort` is **coupled to the ORT version**: regenerate it whenever the model or the
plugin's ORT version changes.

## Keeping it up to date

The mechanism files copied into `android/onnx/` are versioned with the plugin. After
`pnpm add @cantoo/capacitor-onnx@latest`, re-run `pnpm exec cantoo-onnx-reduce` to
refresh them.

## Opting out

Clear `onnxModel` (or never run the command) → the app links the full
`onnxruntime-android` and loads a plain `.onnx`. No code changes required.
