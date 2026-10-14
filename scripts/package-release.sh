#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

STAGING="$ROOT/build/release/vime-asr"
OUTPUT="$ROOT/dist/vime-asr-linux-x86_64.tar.gz"
rm -rf "$STAGING"
mkdir -p "$STAGING/licenses" "$(dirname -- "$OUTPUT")"

# Explicit inputs keep models, recordings, credentials, and build trees out.
files=(
    vime_asr/*.py
    pyproject.toml
    uv.lock
    README.md
    LICENSE
    native/vendor/ggml/*.h
    native/vendor/ggml/LICENSE.*
    build/native/libvime_qwen.so
    build/lib/libggml-base.so
    build/lib/libggml-cpu-haswell.so
)
cp --parents "${files[@]}" "$STAGING/"
cp build/qwen-asr/LICENSE "$STAGING/licenses/LICENSE.qwen-asr"
printf '%s\n' "${GITHUB_REF_NAME:-development}" > "$STAGING/VERSION"

# The prebuilt runtime requires Linux x86_64 AVX2/FMA, glibc >= 2.39,
# system OpenBLAS (Ubuntu: libopenblas0), and uv; no rebuild is needed.
tar -czf "$OUTPUT" -C "$(dirname -- "$STAGING")" vime-asr
printf 'Packaged %s\n' "$OUTPUT"
