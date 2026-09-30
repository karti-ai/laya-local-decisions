#!/usr/bin/env python3
"""Compare CPU and CUDA on identical synthetic inputs, without HTTP/network time."""
import argparse
import gc
import json
import math
import os
import statistics
import time
from pathlib import Path

os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('LAYA_REVISION', 'reviewed')
os.environ.setdefault('LAYA_CUDA_AMP', 'fp16')
os.environ.setdefault('TOKENIZERS_PARALLELISM', 'false')

import torch
from laya import Router

ROOT = Path(__file__).resolve().parents[1]
parser = argparse.ArgumentParser()
parser.add_argument('--model', default='typed-decisions', choices=['english', 'multilingual', 'typed-decisions'])
parser.add_argument('--devices', default='cpu,cuda')
parser.add_argument('--iterations', type=int, default=12)
parser.add_argument('--threads', type=int, default=4)
parser.add_argument('--output', type=Path, default=ROOT / 'runs/benchmark.json')
args = parser.parse_args()
devices = args.devices.split(',')
if args.iterations < 3 or args.threads < 1 or any(d not in ('cpu', 'cuda') for d in devices):
    parser.error('Use cpu,cuda devices, at least 3 iterations, and at least 1 thread')
torch.set_num_threads(args.threads)
questions = json.loads((ROOT / 'examples/questions.json').read_text())
rows = [json.loads(line) for line in (ROOT / 'examples/emails.jsonl').read_text().splitlines() if line.strip()]
states = [r['state'] for r in rows]
results = []


def measure(device, fn):
    if device == 'cuda':
        torch.cuda.synchronize()
    start = time.perf_counter()
    value = fn()
    if device == 'cuda':
        torch.cuda.synchronize()
    return value, (time.perf_counter() - start) * 1000


for device in devices:
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; no CPU substitution in this benchmark')
    router = Router(device=device, max_loaded=1)
    router.preload([args.model])
    agent = router.load(args.model)
    if agent.device.type != device:
        raise RuntimeError(f'Requested {device}, loaded on {agent.device}')
    for _ in range(3):
        agent.predict(states[0], questions)
        agent.predict_batch(states, questions)
    single_times, batch_times = [], []
    for i in range(args.iterations):
        _, elapsed = measure(device, lambda: agent.predict(states[i % len(states)], questions))
        single_times.append(elapsed)
        _, elapsed = measure(device, lambda: agent.predict_batch(states, questions))
        batch_times.append(elapsed)
    if getattr(agent, 'cpu_fallback_count', 0):
        raise RuntimeError('CPU fallback occurred during benchmarking')
    median_batch = statistics.median(batch_times)
    results.append({'device': device, 'model': args.model, 'revisions': router.loaded_revisions, 'single_median_ms': round(statistics.median(single_times), 2), 'single_p95_ms': round(sorted(single_times)[math.ceil(len(single_times) * .95) - 1], 2), 'batch_states': len(states), 'batch_median_ms': round(median_batch, 2), 'batch_states_per_second': round(len(states) * 1000 / median_batch, 2), 'iterations': args.iterations, 'warmup_calls_per_shape': 3, 'cpu_fallbacks': 0})
    router.unload()
    del router, agent
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

report = {'torch': torch.__version__, 'python': __import__('platform').python_version(), 'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, 'cpu_threads': args.threads, 'results': results, 'conditions': 'Four synthetic short English messages; one four-option choice question per state; 3 warmups per shape; CUDA synchronized; no HTTP, provider access, or email changes. Other workloads and machines will differ.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
