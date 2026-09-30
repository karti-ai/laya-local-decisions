"""Small stdlib client shared by the synthetic smoke check and offline evaluator."""
import json
import os
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def settings():
    values = {}
    env = ROOT / '.env'
    if env.exists():
        for line in env.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                values[key.strip()] = value.strip().strip('"').strip("'")
    values.update(os.environ)
    host = values.get('LAYA_HOST', '127.0.0.1')
    if host == '0.0.0.0':
        host = '127.0.0.1'
    base = values.get('LAYA_BASE_URL', f"http://{host}:{values.get('LAYA_PORT', '8000')}")
    return base.rstrip('/'), values.get('LAYA_API_KEY', '')


def request(path, body=None, *, authenticated=True):
    base, key = settings()
    headers = {}
    if authenticated:
        if not key:
            raise ValueError('Set LAYA_API_KEY in your environment or local .env')
        headers['Authorization'] = 'Bearer ' + key
    if body is not None:
        headers['Content-Type'] = 'application/json'
    req = urllib.request.Request(base + path, headers=headers, data=None if body is None else json.dumps(body).encode())
    with urllib.request.urlopen(req, timeout=180) as response:
        return json.load(response)


def check_health(expected_device=None):
    health = request('/health', authenticated=False)
    devices = health.get('checkpoint_devices', {})
    if not devices or health.get('device_is_preference', True):
        raise RuntimeError('No checkpoint is resident; health reports only a device preference')
    if expected_device and any(not x.startswith(expected_device) for x in devices.values()):
        raise RuntimeError('A checkpoint is running on an unexpected device')
    if any(x.get('count', 0) for x in health.get('cpu_fallbacks', {}).values()):
        raise RuntimeError('CPU fallback occurred; inspect the local server logs')
    return health
