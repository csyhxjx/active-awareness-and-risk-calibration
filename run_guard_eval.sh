#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck disable=SC1091
source "${PROJECT_ROOT}/env.sh" >/dev/null
export PYTHONPATH="${PROJECT_ROOT}/guard${PYTHONPATH:+:${PYTHONPATH}}"

cd "${PROJECT_ROOT}"
exec python "${PROJECT_ROOT}/guard/run_libero_eval_guard.py" "$@"
