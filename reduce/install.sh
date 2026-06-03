#!/usr/bin/env bash
#
# install.sh — set up the opt-in, model-driven reduced ONNX Runtime build in a
# Capacitor Android app that depends on @cantoo/capacitor-onnx.
#
# This is the canonical installer SHIPPED INSIDE the plugin package (run it via the
# `cantoo-onnx-reduce` bin / `npx`). Unlike the old self-contained script, the two
# mechanism files live next to this one as REAL files and are COPIED into the app:
#   reduce/onnx-reduce.gradle      → <android>/onnx/onnx-reduce.gradle
#   reduce/build-reduced-onnx.sh   → <android>/onnx/build-reduced-onnx.sh
# So the plugin is the single source of truth — no embedded heredocs to keep in sync.
#
# Background: @cantoo/capacitor-onnx ships the full Microsoft onnxruntime-android
# AAR (~9.4 MB compressed per arm64), whose .so embeds kernels for ~1500 operators.
# A given model uses only a handful. This wires up a build that compiles a reduced
# libonnxruntime.so containing ONLY your model's operators (arm64 ~9.4 MB → ~4 MB),
# driven by YOUR model at build time. Opting out (not running this, or clearing
# onnxModel) keeps the full AAR — nothing breaks if you skip it. Android-only.
#
# What it does, interactively:
#   1. preflight-checks the build toolchain (Python, NDK, cmake, ninja, …)
#   2. optionally creates a Python venv with onnxruntime + onnx
#   3. prompts for parameters (model URL/path, ORT version, ABIs, …)
#   4. copies android/onnx/{onnx-reduce.gradle,build-reduced-onnx.sh}
#   5. writes a managed block to android/gradle.properties (your values)
#   6. patches app/build.gradle to swap the full ORT AAR for the reduced one
#   7. optionally triggers the first build
#
# Platforms: macOS and Linux (x86_64 host) run this directly. On Windows, run it
# inside WSL2 (Ubuntu) — the bash script and the ONNX source build need a Unix
# toolchain.
#
set -euo pipefail

# Resolve our own location so we can copy the mechanism files and read the plugin's
# declared ORT version (android/build.gradle ships one level up in the package).
SELF_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

NDK_VERSION="28.0.13004108"   # LTS pinned by ORT 1.25.x

# Derive the ORT version from the plugin's own android/build.gradle so it always
# matches the onnxruntime-android the plugin depends on (no more "MUST match" drift).
PLUGIN_BUILD_GRADLE="$SELF_DIR/../android/build.gradle"
DERIVED_ORT=""
if [[ -f "$PLUGIN_BUILD_GRADLE" ]]; then
  DERIVED_ORT="$(grep -oE 'onnxruntime-android:[0-9]+\.[0-9]+\.[0-9]+' "$PLUGIN_BUILD_GRADLE" | head -1 | cut -d: -f2 || true)"
fi
ORT_DEFAULT="${DERIVED_ORT:-1.25.1}"

# ----------------------------------------------------------------------------
# Small UI helpers
# ----------------------------------------------------------------------------
bold() { printf '\033[1m%s\033[0m\n' "$1"; }
info() { printf '    %s\n' "$1"; }
ok()   { printf '    \033[32m[ok]\033[0m %s\n' "$1"; }
warn() { printf '    \033[33m[warn]\033[0m %s\n' "$1"; }
miss() { printf '    \033[31m[missing]\033[0m %s\n' "$1"; }

# ask VAR "prompt" "default"  — reads a line, falls back to default if empty.
ask() {
  local __var="$1" __prompt="$2" __default="${3:-}" __reply
  if [[ -n "$__default" ]]; then
    read -r -p "$__prompt [$__default]: " __reply || true
    __reply="${__reply:-$__default}"
  else
    read -r -p "$__prompt: " __reply || true
  fi
  printf -v "$__var" '%s' "$__reply"
}

# ask_yesno "prompt" "y|n"  — returns 0 (yes) / 1 (no).
ask_yesno() {
  local __prompt="$1" __default="${2:-n}" __hint __reply
  [[ "$__default" == "y" ]] && __hint="Y/n" || __hint="y/N"
  read -r -p "$__prompt [$__hint]: " __reply || true
  __reply="${__reply:-$__default}"
  [[ "$__reply" =~ ^[Yy] ]]
}

# ----------------------------------------------------------------------------
# 1. Preflight (doctor)
# ----------------------------------------------------------------------------
doctor() {
  bold "==> Preflight (build toolchain)"
  local have_python=0

  local uname_s; uname_s="$(uname -s 2>/dev/null || echo unknown)"
  case "$uname_s" in
    Darwin|Linux) ok "host: $uname_s" ;;
    MINGW*|MSYS*|CYGWIN*)
      warn "host: $uname_s (Windows shell) — the source build won't work here."
      info "    Run this inside WSL2 (Ubuntu) for the full flow." ;;
    *) warn "host: $uname_s (untested)" ;;
  esac
  if [[ "$(uname -m 2>/dev/null || echo)" != "x86_64" && "$uname_s" == "Linux" ]]; then
    warn "non-x86_64 Linux host — the Android NDK ships only an x86_64 host toolchain."
  fi

  if command -v python3 >/dev/null 2>&1; then
    local pyok
    pyok="$(python3 -c 'import sys; print(1 if sys.version_info[:2] >= (3,10) else 0)' 2>/dev/null || echo 0)"
    if [[ "$pyok" == "1" ]]; then
      ok "python3 $(python3 -c 'import platform; print(platform.python_version())')"
      have_python=1
    else
      warn "python3 is < 3.10 ($(python3 --version 2>&1)). onnxruntime needs Python ≥3.10 (e.g. pyenv / brew install python@3.12)."
    fi
  else
    miss "python3 — needed to generate the operator config (install Python ≥3.10)."
  fi

  local c
  for c in git curl unzip zip java; do
    command -v "$c" >/dev/null 2>&1 && ok "$c" || miss "$c"
  done
  for c in cmake ninja; do
    command -v "$c" >/dev/null 2>&1 && ok "$c ($("$c" --version 2>/dev/null | head -1))" \
      || warn "$c not found — needed only for the first source build (brew install $c)."
  done

  if [[ -n "${ANDROID_SDK_ROOT:-}" ]]; then
    ok "ANDROID_SDK_ROOT=$ANDROID_SDK_ROOT"
    if [[ -d "$ANDROID_SDK_ROOT/ndk/$NDK_VERSION" ]]; then
      ok "NDK $NDK_VERSION"
    else
      warn "NDK $NDK_VERSION not installed — needed only for the first source build:"
      info "    sdkmanager \"ndk;$NDK_VERSION\"  &&  export ANDROID_NDK_HOME=\"\$ANDROID_SDK_ROOT/ndk/$NDK_VERSION\""
    fi
  else
    warn "ANDROID_SDK_ROOT unset — set it before the first source build."
  fi

  PY_OK="$have_python"
}

# ----------------------------------------------------------------------------
# 2. Optional Python venv with onnxruntime + onnx
# ----------------------------------------------------------------------------
maybe_make_venv() {
  PYTHON_DEFAULT="python3"
  local existing="$REPO_ROOT/.venv/bin/python"
  if [[ -x "$existing" ]]; then
    ok ".venv already present at $existing"
    PYTHON_DEFAULT="$existing"
    return
  fi
  [[ "${PY_OK:-0}" == "1" ]] || return
  if ask_yesno "Create a Python venv (.venv) and install onnxruntime==$ORT_VERSION + onnx?" "y"; then
    python3 -m venv "$REPO_ROOT/.venv"
    "$REPO_ROOT/.venv/bin/pip" install --quiet --upgrade pip
    "$REPO_ROOT/.venv/bin/pip" install "onnxruntime==$ORT_VERSION" onnx
    ok "venv ready: $existing"
    PYTHON_DEFAULT="$existing"
  fi
}

# ----------------------------------------------------------------------------
# 4. Copy the mechanism files (shipped next to this script) into <android>/onnx/
# ----------------------------------------------------------------------------
write_mechanism() {
  local f
  for f in onnx-reduce.gradle build-reduced-onnx.sh; do
    [[ -f "$SELF_DIR/$f" ]] || { echo "error: missing $SELF_DIR/$f — broken @cantoo/capacitor-onnx install?" >&2; exit 1; }
  done
  mkdir -p "$ANDROID_DIR/onnx"
  cp "$SELF_DIR/onnx-reduce.gradle"    "$ANDROID_DIR/onnx/onnx-reduce.gradle"
  cp "$SELF_DIR/build-reduced-onnx.sh" "$ANDROID_DIR/onnx/build-reduced-onnx.sh"
  chmod +x "$ANDROID_DIR/onnx/build-reduced-onnx.sh"
  ok "android/onnx/onnx-reduce.gradle"
  ok "android/onnx/build-reduced-onnx.sh"
}

# ----------------------------------------------------------------------------
# 5. Managed block in <android>/gradle.properties (never touches other lines)
# ----------------------------------------------------------------------------
write_gradle_properties() {
  local gp="$ANDROID_DIR/gradle.properties"
  local begin="# >>> cantoo onnx-reduce >>>"
  local end="# <<< cantoo onnx-reduce <<<"
  local block
  block="$(cat <<EOF
$begin
# Managed by cantoo-onnx-reduce (@cantoo/capacitor-onnx) — edit values here or override with -P… at build time.
onnxModel=$MODEL_URL
onnxOrtVersion=$ORT_VERSION
onnxCacheUrl=$CACHE_URL
onnxCacheUploadUrl=$CACHE_UPLOAD_URL
onnxOrtUploadUrl=$ORT_UPLOAD_URL
onnxConfigUrl=$CONFIG_URL
onnxConfigUploadUrl=$CONFIG_UPLOAD_URL
onnxPython=$PYTHON_INTERP
$end
EOF
)"
  if [[ -f "$gp" ]]; then
    cp "$gp" "$gp.bak"
    if grep -qF "$begin" "$gp"; then
      local l inblock=0
      while IFS= read -r l || [[ -n "$l" ]]; do
        if [[ "$l" == "$begin" ]]; then inblock=1; printf '%s\n' "$block"; continue; fi
        if [[ "$inblock" == 1 && "$l" == "$end" ]]; then inblock=0; continue; fi
        [[ "$inblock" == 1 ]] && continue
        printf '%s\n' "$l"
      done < "$gp" > "$gp.tmp"
      mv "$gp.tmp" "$gp"
      ok "updated managed block in android/gradle.properties (backup: gradle.properties.bak)"
    else
      printf '\n%s\n' "$block" >> "$gp"
      ok "appended managed block to android/gradle.properties (backup: gradle.properties.bak)"
    fi
  else
    printf '%s\n' "$block" > "$gp"
    ok "created android/gradle.properties with managed block"
  fi
}

# ----------------------------------------------------------------------------
# 6. Patch app/build.gradle (idempotent, with .bak; falls back to printing)
# ----------------------------------------------------------------------------
SWAP_BLOCK='
// Opt-in reduced ONNX Runtime: swap the full onnxruntime-android (pulled
// transitively by @cantoo/capacitor-onnx) for an AAR with only this model'\''s ops.
if (project.ext.onnx.enabled) {
    def reducedAar = project.ext.onnx.resolveReducedAar()
    configurations.all {
        exclude group: '\''com.microsoft.onnxruntime'\'', module: '\''onnxruntime-android'\''
    }
    dependencies {
        implementation files(reducedAar)
    }
}
'

print_manual_instructions() {
  bold "==> Manual step needed for app/build.gradle"
  info "Add near the top (after apply plugin: 'com.android.application'):"
  info "    apply from: '../onnx/onnx-reduce.gradle'"
  info "And add this block (before 'apply from: capacitor.build.gradle'):"
  printf '%s\n' "$SWAP_BLOCK"
}

patch_build_gradle() {
  local bg="$ANDROID_DIR/app/build.gradle"
  if [[ ! -f "$bg" ]]; then
    warn "$bg not found."
    print_manual_instructions
    return
  fi

  local has_apply has_swap
  grep -q "onnx/onnx-reduce.gradle" "$bg" && has_apply=1 || has_apply=0
  grep -q "resolveReducedAar" "$bg" && has_swap=1 || has_swap=0
  if [[ "$has_apply" == "1" && "$has_swap" == "1" ]]; then
    ok "app/build.gradle already patched (idempotent — nothing to do)"
    return
  fi

  local apply_anchor cap_anchor
  apply_anchor="$(grep -n "apply plugin: 'com.android.application'" "$bg" | head -1 | cut -d: -f1)"
  cap_anchor="$(grep -n "apply from: 'capacitor.build.gradle'" "$bg" | head -1 | cut -d: -f1)"
  if [[ -z "$apply_anchor" || -z "$cap_anchor" ]]; then
    warn "could not find the expected anchors in app/build.gradle — not modifying it."
    print_manual_instructions
    return
  fi

  [[ -f "$bg.bak" ]] || cp "$bg" "$bg.bak"
  local l lineno=0
  while IFS= read -r l || [[ -n "$l" ]]; do
    lineno=$((lineno + 1))
    if [[ "$lineno" == "$cap_anchor" && "$has_swap" == "0" ]]; then
      printf '%s\n' "$SWAP_BLOCK"
    fi
    printf '%s\n' "$l"
    if [[ "$lineno" == "$apply_anchor" && "$has_apply" == "0" ]]; then
      printf '%s\n' "apply from: '../onnx/onnx-reduce.gradle'"
    fi
  done < "$bg" > "$bg.tmp"
  mv "$bg.tmp" "$bg"
  ok "patched app/build.gradle (backup: app/build.gradle.bak)"
}

# Ensure app/build.gradle's abiFilters matches the chosen ABIs:
#   • absent   → inject `ndk { abiFilters … }` into defaultConfig
#   • present & equal   → ok
#   • present & different → ERROR (exit) — the reduced AAR ships only these ABIs.
ensure_abifilters() {
  local bg="$ANDROID_DIR/app/build.gradle"
  [[ -f "$bg" ]] || return 0

  local want
  want="$(printf '%s\n' ${ABIS//,/ } | sort -u | paste -sd, -)"

  if grep -q "abiFilters" "$bg"; then
    local have
    have="$(grep "abiFilters" "$bg" | grep -oE "['\"][A-Za-z0-9_-]+['\"]" | tr -d "'\"" | sort -u | paste -sd, -)"
    if [[ "$have" == "$want" ]]; then
      ok "abiFilters already set to: $have"
    else
      printf '    \033[31m[error]\033[0m abiFilters in app/build.gradle does not match the chosen ABIs.\n' >&2
      info "    app/build.gradle has: ${have:-<unparseable>}"
      info "    installer was given : $want"
      info "    Reconcile before continuing: re-run with ABIs matching the app, or edit app/build.gradle."
      exit 1
    fi
  else
    local dcln
    dcln="$(grep -nE '^[[:space:]]*defaultConfig[[:space:]]*\{' "$bg" | head -1 | cut -d: -f1)"
    if [[ -z "$dcln" ]]; then
      warn "no defaultConfig { } block found — add abiFilters manually:"
      info "    android { defaultConfig { ndk { abiFilters $(printf "'%s' " ${want//,/ }) } } }"
      return 0
    fi
    local quoted
    quoted="$(printf "'%s', " ${want//,/ })"; quoted="${quoted%, }"
    [[ -f "$bg.bak" ]] || cp "$bg" "$bg.bak"
    local l lineno=0
    while IFS= read -r l || [[ -n "$l" ]]; do
      lineno=$((lineno + 1))
      printf '%s\n' "$l"
      if [[ "$lineno" == "$dcln" ]]; then
        printf '        ndk {\n            abiFilters %s\n        }\n' "$quoted"
      fi
    done < "$bg" > "$bg.tmp"
    mv "$bg.tmp" "$bg"
    ok "added abiFilters ($want) to app/build.gradle defaultConfig (backup: app/build.gradle.bak)"
  fi

  if ! printf '%s' "$want" | grep -qE 'x86'; then
    warn "Builds now ship ONLY [$want] — x86/x86_64 EMULATORS can't run them (no matching .so)."
    info "    Use an arm64 device/emulator, or add an x86 ABI."
  fi
}

# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------
bold "Cantoo — reduced ONNX Runtime installer"
echo

doctor
echo

bold "==> Parameters"
if [[ -d "./android/app" ]]; then DEF_ANDROID="./android"; else DEF_ANDROID="."; fi
ask ANDROID_DIR "Android project directory (contains app/)" "$DEF_ANDROID"
ANDROID_DIR="${ANDROID_DIR%/}"
[[ -d "$ANDROID_DIR" ]] || { echo "error: '$ANDROID_DIR' not found" >&2; exit 1; }
REPO_ROOT="$(cd "$ANDROID_DIR/.." && pwd)"

# Model reference (required; loop until non-empty). URL or local path to the .onnx.
MODEL_URL=""
while [[ -z "$MODEL_URL" ]]; do
  ask MODEL_URL "Model URL or local path (.onnx; downloaded/used on first build; e.g. https://host/model.onnx)" ""
  [[ -z "$MODEL_URL" ]] && warn "the model URL or path is required."
done

ask ORT_VERSION "ONNX Runtime version (matches @cantoo/capacitor-onnx's onnxruntime-android)" "$ORT_DEFAULT"
ask ABIS "Target ABIs (comma-separated)" "arm64-v8a,armeabi-v7a"

echo
maybe_make_venv
ask PYTHON_INTERP "Python interpreter (with onnxruntime + onnx)" "${PYTHON_DEFAULT:-python3}"
ask CACHE_URL "Remote AAR cache base URL (optional, read-only GET by hash)" ""
ask CACHE_UPLOAD_URL "Remote AAR UPLOAD target (rsync/ssh, optional; e.g. user@host:/path/android)" ""
ask ORT_UPLOAD_URL "Remote .ort model UPLOAD target (rsync/ssh file, optional; empty = don't publish)" ""
ask CONFIG_URL "Remote op-config GET URL (optional; lets consumers skip Python on cache hit)" ""
ask CONFIG_UPLOAD_URL "Remote op-config UPLOAD target (rsync/ssh file, optional; empty = don't publish)" ""

echo
bold "==> Installing"
ensure_abifilters      # validate/inject ABIs first — a mismatch aborts before any other change
write_mechanism
write_gradle_properties
patch_build_gradle

echo
bold "==> Done"
info "Build the reduced AAR (first run downloads the model + compiles ~8–40 min):"
echo
info "    ANDROID_NDK_HOME=\"\$ANDROID_SDK_ROOT/ndk/$NDK_VERSION\" \\"
info "    \"$ANDROID_DIR/gradlew\" -p \"$ANDROID_DIR\" :app:help --console=plain"
echo
info "(Values were written to android/gradle.properties, so no -P flags are needed.)"
echo

if ask_yesno "Run the build now?" "n"; then
  if [[ -z "${ANDROID_NDK_HOME:-}" && -n "${ANDROID_SDK_ROOT:-}" ]]; then
    export ANDROID_NDK_HOME="$ANDROID_SDK_ROOT/ndk/$NDK_VERSION"
  fi
  "$ANDROID_DIR/gradlew" -p "$ANDROID_DIR" :app:help --console=plain
fi
