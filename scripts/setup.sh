#!/usr/bin/env bash
set -euo pipefail
ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT_DIR"
DEVICE="${1:-cuda}"
case "$DEVICE" in
  cuda) TORCH_INDEX=https://download.pytorch.org/whl/cu128 ;;
  cpu) TORCH_INDEX=https://download.pytorch.org/whl/cpu ;;
  *) printf 'Usage: scripts/setup.sh [cuda|cpu]\n' >&2; exit 2 ;;
esac
command -v uv >/dev/null || { printf 'Install uv first: https://docs.astral.sh/uv/getting-started/installation/\n' >&2; exit 1; }
if [[ ! -x .venv/bin/python ]]; then
  uv venv --python "${LAYA_PYTHON:-3.13}" .venv
fi
uv pip install --python .venv/bin/python 'torch==2.8.0' --index-url "$TORCH_INDEX"
uv pip install --python .venv/bin/python -r requirements.txt
uv pip freeze --python .venv/bin/python > .venv/requirements.lock.txt
.venv/bin/python - "$DEVICE" <<'PY'
import os, secrets, sys
from pathlib import Path
import torch
device = sys.argv[1]
if device == 'cuda':
    assert torch.cuda.is_available(), 'CUDA is unavailable. Check the NVIDIA driver and PyTorch wheel.'
    torch.ones(1, device='cuda').cpu()
    print('CUDA verified:', torch.cuda.get_device_name(0), 'torch', torch.__version__)
else:
    print('CPU environment verified:', torch.__version__)
env = Path('.env')
if not env.exists():
    text = Path('.env.example').read_text().replace('replace-with-a-random-local-key', secrets.token_urlsafe(36))
    text = text.replace('LAYA_DEVICE=cuda', 'LAYA_DEVICE=' + device)
    fd = os.open(env, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(fd, 'w') as f:
        f.write(text)
    print('Created a private .env with a fresh API key; key not printed.')
else:
    print('Existing .env preserved. Check LAYA_DEVICE before serving.')
PY
