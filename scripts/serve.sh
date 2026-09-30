#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
[[ -f .env ]] || { printf 'Run scripts/setup.sh first.\n' >&2; exit 1; }
set -a
source .env
set +a
[[ -n "${LAYA_API_KEY:-}" && "$LAYA_API_KEY" != replace-with-a-random-local-key ]] || { printf 'Set a fresh LAYA_API_KEY in .env.\n' >&2; exit 1; }
.venv/bin/python - <<'PY'
import os, torch
if os.environ.get('LAYA_DEVICE') == 'cuda':
    assert torch.cuda.is_available(), 'Refusing to start a requested CUDA service on CPU'
    torch.ones(1, device='cuda').cpu()
PY
exec .venv/bin/laya-serve
