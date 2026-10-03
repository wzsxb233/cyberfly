#!/usr/bin/env bash
set -euo pipefail
CYBERFLY_RUNTIME_ROOT="${CYBERFLY_MINICPM_ROOT:-/root/.cache/cyberfly/minicpm-runtime}"
CYBERFLY_SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CYBERFLY_SOURCE_REV=64d092c60db4b4ee45768476bd752f03fdcc98ea
mkdir -p "$CYBERFLY_RUNTIME_ROOT"
if [[ ! -x "$CYBERFLY_RUNTIME_ROOT/venv/bin/python" ]]; then
  UV_CACHE_DIR="${CYBERFLY_RUNTIME_ROOT}-uv" uv venv "$CYBERFLY_RUNTIME_ROOT/venv"
fi
UV_CACHE_DIR="${CYBERFLY_RUNTIME_ROOT}-uv" uv pip install --quiet --python "$CYBERFLY_RUNTIME_ROOT/venv/bin/python" \
  cmake==4.4.3 ninja==1.13.2 websockets==15.0.1 fastapi==0.141.1 uvicorn==0.52.4 pillow==12.3.0 numpy==2.4.6 python-multipart==0.0.20
if [[ ! -d "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni/.git" ]]; then
  git clone --no-checkout https://github.com/tc-mb/llama.cpp-omni.git "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni"
  git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" checkout --detach "$CYBERFLY_SOURCE_REV"
fi
if [[ "$(git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" rev-parse HEAD)" != "$CYBERFLY_SOURCE_REV" ]]; then
  echo 'Unexpected source revision; refusing to overwrite an existing checkout.' >&2
  exit 1
fi
if git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" apply --check "$CYBERFLY_SCRIPT_DIR/localhost-bind.patch" 2>/dev/null; then
  git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" apply "$CYBERFLY_SCRIPT_DIR/localhost-bind.patch"
elif ! git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" apply --reverse --check "$CYBERFLY_SCRIPT_DIR/localhost-bind.patch" 2>/dev/null; then
  echo 'Cannot apply the loopback binding patch cleanly.' >&2
  exit 1
fi
if git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" apply --check "$CYBERFLY_SCRIPT_DIR/multimodal-text.patch" 2>/dev/null; then
  git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" apply "$CYBERFLY_SCRIPT_DIR/multimodal-text.patch"
elif ! git -C "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" apply --reverse --check "$CYBERFLY_SCRIPT_DIR/multimodal-text.patch" 2>/dev/null; then
  echo 'Cannot apply the multimodal text-preservation patch cleanly.' >&2
  exit 1
fi
export PATH="$CYBERFLY_RUNTIME_ROOT/venv/bin:/usr/local/cuda-12.9/bin:$PATH"
export CUDA_VISIBLE_DEVICES=2
cmake -S "$CYBERFLY_RUNTIME_ROOT/llama.cpp-omni" -B "$CYBERFLY_RUNTIME_ROOT/build" -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF \
  -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DCMAKE_EXE_LINKER_FLAGS=-Wl,--allow-shlib-undefined
cmake --build "$CYBERFLY_RUNTIME_ROOT/build" --target llama-omni-server llama-omni-cli -j "${CYBERFLY_BUILD_JOBS:-8}"
