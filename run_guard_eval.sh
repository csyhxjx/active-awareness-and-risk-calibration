#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTHONPATH="${PROJECT_ROOT}/guard${PYTHONPATH:+:${PYTHONPATH}}"
export HF_HOME="${HF_HOME:-${PROJECT_ROOT}/hf-cache}"
export HF_HUB_CACHE="${HF_HUB_CACHE:-${PROJECT_ROOT}/hf-cache}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"

exec conda run --no-capture-output -n project \
  python "${PROJECT_ROOT}/guard/run_libero_eval_guard.py" "$@"
