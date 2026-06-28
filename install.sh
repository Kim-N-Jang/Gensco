#!/usr/bin/env sh
# GenSCO environment setup (uv-based, Python 3.10).
# Reproduces the working environment on host `inu`. Run from the repo root:
#     sh install.sh
# Idempotent: safe to re-run.
set -e

UV="${UV:-$HOME/.local/bin/uv}"
VENV=.venv
PY="$VENV/bin/python"

# Large CUDA wheels (cudnn ~700MB, cublas ~580MB) time out on flaky links with
# uv's defaults; longer timeout + sequential downloads make the install reliable.
export UV_HTTP_TIMEOUT="${UV_HTTP_TIMEOUT:-900}"
export UV_CONCURRENT_DOWNLOADS="${UV_CONCURRENT_DOWNLOADS:-1}"

# 1) uv virtualenv on Python 3.10 (created only if missing).
#    We intentionally stay on 3.10 (the dependency stack is built for it).
if [ ! -d "$VENV" ]; then
    "$UV" venv --python 3.10 "$VENV"
fi

# 2) Python dependencies.
#    jax[cuda12] pulls the full nvidia CUDA stack (cuDNN/cuBLAS/...) — without it
#    GPU ops fail with "DNN library initialization failed".
#    typing_extensions: needed by the 3.10 patch in step 3.
#    pybind11: needed to build the C++ extension in step 4.
"$UV" pip install --python "$PY" \
    "jax[cuda12]==0.5.0" flax==0.10.4 optax==0.2.4 triton==3.1.0 \
    tensorflow_cpu==2.19.0 tensorboardx tqdm \
    typing_extensions pybind11
"$UV" pip install --python "$PY" torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
"$UV" pip install --python "$PY" numpy==1.26.4

# 3) Make AlphaFold3-derived common/array_view.py importable on Python 3.10.
#    It uses two 3.11-only features (typing.Self, star-unpack in a subscript).
#    Both substitutions are no-ops if the file is already patched.
sed -i \
    -e 's/^from typing import Any, Self, TypeAlias, TypeVar$/from typing import Any, TypeAlias, TypeVar\nfrom typing_extensions import Self/' \
    -e 's/self\[\*slice_prefix, slice(lo, hi)\]/self[(*slice_prefix, slice(lo, hi))]/' \
    common/array_view.py

# 4) Build the C++ heuristics extension (2-opt etc.). The Makefile calls
#    `python3 -m pybind11` and `python3-config`, so build with the venv active.
. "$VENV/bin/activate"
( cd lib && make )

echo "Setup complete. Activate the env with:  source $VENV/bin/activate"
