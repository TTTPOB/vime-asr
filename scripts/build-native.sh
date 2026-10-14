#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

if [[ "$(uname -s)" != Linux || "$(uname -m)" != x86_64 ]]; then
    echo "This build targets Linux x86_64 with AVX2 and FMA." >&2
    exit 1
fi

CC="${CC:-gcc}"
BLASPREFIX="${BLASPREFIX:-/usr}"
UPSTREAM_COMMIT=924694251d9e0f18e5d86bbd06aa3ab5f870002d
UPSTREAM="$ROOT/build/qwen-asr"
PATCHED="$ROOT/build/patched"
LIB="$ROOT/build/lib"
OUTPUT="$ROOT/build/native/libvime_qwen.so"
ARCHIVE="$ROOT/build/transcribe-native-0.4.0-linux-x86_64-cpu-vulkan.tar.gz"
ARCHIVE_PREFIX=transcribe-native-linux-x86_64-cpu-vulkan

mkdir -p "$UPSTREAM" "$PATCHED" "$LIB" "$(dirname -- "$OUTPUT")"
if [[ ! -d "$UPSTREAM/.git" ]]; then
    git -C "$UPSTREAM" init -q
fi
git -C "$UPSTREAM" fetch --depth 1 https://github.com/antirez/qwen-asr.git "$UPSTREAM_COMMIT"
git -C "$UPSTREAM" checkout --detach "$UPSTREAM_COMMIT"

# Patch disposable source copies, never the upstream checkout.
cp "$UPSTREAM/qwen_asr.c" "$UPSTREAM/qwen_asr_kernels.c" "$PATCHED/"
patch --batch -d "$PATCHED" -p1 < native/upstream.patch

if [[ ! -f "$LIB/libggml-base.so" || ! -f "$LIB/libggml-cpu-haswell.so" ]]; then
    if [[ ! -f "$ARCHIVE" ]]; then
        curl -fL --retry 2 \
            https://github.com/handy-computer/transcribe.cpp/releases/download/v0.4.0/transcribe-native-0.4.0-linux-x86_64-cpu-vulkan.tar.gz \
            -o "$ARCHIVE.new"
        mv "$ARCHIVE.new" "$ARCHIVE"
    fi
    tar -xzf "$ARCHIVE" -C "$LIB" --strip-components=1 \
        "$ARCHIVE_PREFIX/libggml-base.so" \
        "$ARCHIVE_PREFIX/libggml-cpu-haswell.so" \
        "$ARCHIVE_PREFIX/licenses/LICENSE" \
        "$ARCHIVE_PREFIX/licenses/ggml/LICENSE"
fi

sources=(
    "$PATCHED/qwen_asr.c"
    "$PATCHED/qwen_asr_kernels.c"
    "$UPSTREAM/qwen_asr_kernels_generic.c"
    "$UPSTREAM/qwen_asr_kernels_neon.c"
    "$UPSTREAM/qwen_asr_kernels_avx.c"
    "$UPSTREAM/qwen_asr_audio.c"
    "$UPSTREAM/qwen_asr_encoder.c"
    "$UPSTREAM/qwen_asr_decoder.c"
    "$UPSTREAM/qwen_asr_tokenizer.c"
    "$UPSTREAM/qwen_asr_safetensors.c"
    native/vime_native.c
    native/vime_ar_weights.c
)

# Rename the new library so existing mmap users keep their old inode intact.
"$CC" -shared -fPIC -O3 -ffast-math -mavx2 -mfma \
    -DUSE_BLAS -DUSE_OPENBLAS \
    -I"$UPSTREAM" -Inative -Inative/vendor/ggml \
    -I"$BLASPREFIX/include" -I"$BLASPREFIX/include/openblas" \
    "${sources[@]}" \
    -L"$LIB" -L"$BLASPREFIX/lib" -L"$BLASPREFIX/lib64" \
    -Wl,--enable-new-dtags '-Wl,-rpath,$ORIGIN/../lib' \
    -lggml-cpu-haswell -lggml-base -lstdc++ -lopenblas -lpthread -lm \
    -o "$OUTPUT.new"
mv "$OUTPUT.new" "$OUTPUT"
printf 'Built %s\n' "$OUTPUT"
