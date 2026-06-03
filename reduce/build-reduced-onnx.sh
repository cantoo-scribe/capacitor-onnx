#!/usr/bin/env bash
#
# build-reduced-onnx.sh — compile a reduced ONNX Runtime Android AAR.
#
# Invoked by onnx-reduce.gradle on a full cache miss. Checks out the ONNX
# Runtime source at the pinned tag and builds an AAR that includes ONLY the
# operators listed in the provided reduced-build config — dropping the ~27 MB
# arm64 libonnxruntime.so (official Maven full build) to ~11 MB.
#
# This is the heavy path (30–60 min) and requires the ORT build toolchain:
# Android NDK + SDK, CMake, Ninja, Python 3, and `pip install onnxruntime`
# (for the tooling). It runs only when neither the local nor the remote AAR
# cache has the requested hash, so in practice it runs once per
# (model op-set + ORT version + abis + nnapi) combination — typically on CI.
#
# Inputs (environment variables, set by the Gradle task):
#   ORT_VERSION   ONNX Runtime version/tag to build (e.g. 1.25.1). Required.
#   ORT_ABIS      Comma-separated Android ABIs (e.g. arm64-v8a,armeabi-v7a). Required.
#   ORT_OP_CONFIG Path to the reduced-build operator config. Required.
#   ORT_USE_NNAPI 1 to include the NNAPI EP, 0 for CPU-only. Default 0.
#   ORT_OUT_AAR   Destination path for the produced .aar. Required.
#   ANDROID_NDK_HOME / ANDROID_SDK_ROOT  Toolchain locations. Required.
#   ORT_SRC_DIR   Where to check out / reuse the ORT source. Default: .cache/ort-src.
#
set -euo pipefail

: "${ORT_VERSION:?ORT_VERSION is required}"
: "${ORT_ABIS:?ORT_ABIS is required}"
: "${ORT_OP_CONFIG:?ORT_OP_CONFIG is required}"
: "${ORT_OUT_AAR:?ORT_OUT_AAR is required}"
ORT_USE_NNAPI="${ORT_USE_NNAPI:-0}"
PYTHON="${PYTHON:-python3}"   # interpreter to drive build.py (e.g. a .venv)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORT_SRC_DIR="${ORT_SRC_DIR:-${SCRIPT_DIR}/../android/.cache/ort-src}"

# ORT 1.25.x pins NDK 28.0.13004108 (LTS). Install: sdkmanager "ndk;28.0.13004108"
# then export ANDROID_NDK_HOME="$ANDROID_SDK_ROOT/ndk/28.0.13004108".
: "${ANDROID_NDK_HOME:?Set ANDROID_NDK_HOME to your Android NDK path}"
: "${ANDROID_SDK_ROOT:?Set ANDROID_SDK_ROOT to your Android SDK path}"

if [[ ! -f "$ORT_OP_CONFIG" ]]; then
  echo "build-reduced-onnx: operator config not found: $ORT_OP_CONFIG" >&2
  exit 1
fi

echo "==> Reduced ONNX Runtime build"
echo "    version : $ORT_VERSION"
echo "    abis    : $ORT_ABIS"
echo "    nnapi   : $ORT_USE_NNAPI"
echo "    config  : $ORT_OP_CONFIG"
echo "    out     : $ORT_OUT_AAR"

# 1. Check out (or reuse) the ORT source at the matching tag.
if [[ ! -d "$ORT_SRC_DIR/.git" ]]; then
  echo "==> Cloning onnxruntime @ v${ORT_VERSION}"
  git clone --depth 1 --branch "v${ORT_VERSION}" --recursive \
    https://github.com/microsoft/onnxruntime.git "$ORT_SRC_DIR"
else
  echo "==> Reusing ORT source at $ORT_SRC_DIR"
  git -C "$ORT_SRC_DIR" fetch --depth 1 origin "v${ORT_VERSION}"
  git -C "$ORT_SRC_DIR" checkout "v${ORT_VERSION}"
  git -C "$ORT_SRC_DIR" submodule update --init --recursive --depth 1
fi

# 2. Build the reduced *native* libonnxruntime.so for each ABI (shared lib only,
#    no --build_java). We intentionally skip ORT's Java/AAR tooling: its internal
#    Android Gradle build trips on jlink/JdkImageTransform under JDK 21, and we
#    don't need it. The Java API (classes.jar) and the small JNI glue
#    (libonnxruntime4j_jni.so) are op-set-independent, so step 3 grafts our
#    reduced .so into the official Maven AAR, keeping those as-is.
#
#    Reduced ops (no --minimal_build → keeps the .onnx parser so the runtime
#    model can stay .onnx). No --use_xnnpack/--use_webgpu (lean, CPU-only);
#    --use_nnapi is opt-in. --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF
#    avoids compiling the test targets (which, under REDUCED_OPS_BUILD, trip
#    -Werror on unused test-only constants).
BUILD_DIR="$ORT_SRC_DIR/build/android-reduced"
mkdir -p "$BUILD_DIR"

# NDK strip tool (NDK ships a darwin-x86_64 prebuilt even on Apple Silicon).
HOST_TAG="$(uname -s | tr '[:upper:]' '[:lower:]')-x86_64"
STRIP="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG/bin/llvm-strip"

declare -a BUILT_ABIS=()
IFS=',' read -ra _abis <<< "$ORT_ABIS"
for _abi in "${_abis[@]}"; do
  echo "==> Building reduced native lib for ${_abi} (this can take ~15–25 min)…"
  NNAPI_ARG=()
  [[ "$ORT_USE_NNAPI" == "1" ]] && NNAPI_ARG=(--use_nnapi)
  "$PYTHON" "$ORT_SRC_DIR/tools/ci_build/build.py" \
    --build_dir "$BUILD_DIR/native/$_abi" \
    --config Release \
    --android \
    --android_sdk_path "$ANDROID_SDK_ROOT" \
    --android_ndk_path "$ANDROID_NDK_HOME" \
    --android_abi "$_abi" \
    --android_api 26 \
    --include_ops_by_config "$ORT_OP_CONFIG" \
    --skip_tests \
    --build_shared_lib \
    --compile_no_warning_as_error \
    --enable_lto \
    --parallel \
    --cmake_generator=Ninja \
    --cmake_extra_defines onnxruntime_BUILD_UNIT_TESTS=OFF \
    ${NNAPI_ARG[@]+"${NNAPI_ARG[@]}"}
  SO="$BUILD_DIR/native/$_abi/Release/libonnxruntime.so"
  [[ -f "$SO" ]] || { echo "build-reduced-onnx: missing $SO" >&2; exit 1; }
  [[ -x "$STRIP" ]] && "$STRIP" --strip-unneeded "$SO" || true
  echo "    $_abi: $(ls -l "$SO" | awk '{print $5}') bytes (reduced, stripped)"
  BUILT_ABIS+=("$_abi")
done

# 3. Graft into the official AAR: download it (cached), swap each built ABI's
#    libonnxruntime.so for our reduced one, drop ABIs we didn't build (the app's
#    abiFilters is arm-only anyway), and repackage. Keeps classes.jar and
#    libonnxruntime4j_jni.so from the official AAR — both op-set-independent.
OFFICIAL_AAR="$ORT_SRC_DIR/../onnx-official/onnxruntime-android-${ORT_VERSION}.aar"
mkdir -p "$(dirname "$OFFICIAL_AAR")"
if [[ ! -f "$OFFICIAL_AAR" ]]; then
  echo "==> Downloading official AAR ${ORT_VERSION} (for Java + JNI glue)"
  curl -fsSL -o "$OFFICIAL_AAR" \
    "https://repo1.maven.org/maven2/com/microsoft/onnxruntime/onnxruntime-android/${ORT_VERSION}/onnxruntime-android-${ORT_VERSION}.aar"
fi

WORK="$BUILD_DIR/aar_graft"
rm -rf "$WORK"; mkdir -p "$WORK"
( cd "$WORK" && unzip -q "$OFFICIAL_AAR" )

echo "==> Grafting reduced .so into AAR (abis: ${BUILT_ABIS[*]})"
for _abi in "${BUILT_ABIS[@]}"; do
  cp "$BUILD_DIR/native/$_abi/Release/libonnxruntime.so" "$WORK/jni/$_abi/libonnxruntime.so"
done
# Drop non-built ABIs (e.g. x86, x86_64) to keep the AAR lean.
for d in "$WORK"/jni/*/; do
  abi="$(basename "$d")"
  keep=0
  for b in "${BUILT_ABIS[@]}"; do [[ "$b" == "$abi" ]] && keep=1; done
  [[ "$keep" == "0" ]] && { echo "    dropping ABI $abi"; rm -rf "$d"; }
done

mkdir -p "$(dirname "$ORT_OUT_AAR")"
rm -f "$ORT_OUT_AAR"
( cd "$WORK" && zip -qr -X "$ORT_OUT_AAR" . )
echo "==> Done: $ORT_OUT_AAR ($(ls -l "$ORT_OUT_AAR" | awk '{print $5}') bytes)"
