# Verification: RTX 2070 Super

Reference run: 2026-09-30. Linux x86-64, RTX 2070 Super with 8 GB VRAM,
Python 3.13.12, PyTorch 2.8.0+cu128, Transformers 4.57.6, Laya 0.3.22.
CPU inference used four PyTorch threads. CUDA inference used the runtime's
FP16 autocast policy for Turing; model weights remain in the upstream loader's
default storage precision. This is inference, not a training run.

## Direct CPU/GPU benchmark

```bash
.venv/bin/python scripts/benchmark.py \
  --model typed-decisions --devices cpu,cuda \
  --iterations 12 --threads 4 --output runs/benchmark.json
```

| Measurement | CPU | CUDA |
| --- | ---: | ---: |
| Single-state median | 143.12 ms | 14.10 ms |
| Single-state p95 | 157.05 ms | 14.26 ms |
| Four-state batch median | 538.64 ms | 25.08 ms |
| Batch states per second | 7.43 | 159.52 |
| CPU fallback count | 0 | 0 |

GPU single-state median was about 10.2 times faster. Batch throughput was about
21.5 times higher. These ratios compare the same questions and texts on this
machine; they do not describe all CPUs, GPUs, or email workloads.

Conditions: the four short, synthetic English messages in
`examples/emails.jsonl`; one four-option `choice` question per state from
`examples/questions.json`; three warmup calls per shape; twelve timed calls
per shape; explicit CUDA synchronization before and after each timed call.
The CPU path used its default FP32 execution. The API service was stopped
while this benchmark loaded an additional checkpoint. Downloads, loading,
HTTP/network latency, and provider API time are excluded.

The checkpoint was loaded from the reviewed bundle revision
`55cf4c4ebb4ebe31b2550e8bdf3bd21b99753851`. The separate standalone repository
has its own reviewed revision; this run used the bundle's `typed-decisions`
subfolder. Raw aggregate output: [reference-benchmark.json](reference-benchmark.json).

## Serving checks

- The setup script installed the pinned runtime in an isolated environment
  and completed a CUDA tensor operation on the 2070 Super.
- The authenticated HTTP API returned typed answers for single and batch
  requests. Inference without the bearer token returned 401.
- All three checkpoints were resident on CUDA when tested together, with
  zero reported CPU fallbacks after requests. Idle loaded GPU memory was
  approximately 5 GB including the display/driver context; runtime allocations
  increased it. This is not a guaranteed peak-memory budget.
- All four synthetic cases received their expected categories from each
  checkpoint. These easy examples are an API check, not a domain evaluation.
- The public evaluator ran against the invented cases and wrote aggregate
  metrics without message text or source IDs.

## Limits and traps observed

At a 0.9 selected-label probability threshold, `typed-decisions` accepted zero
of the four synthetic examples, even though its top labels were correct.
That illustrates why a copied threshold is not a useful production policy.
No real-mail accuracy, calibration, or archive/delete policy was evaluated.

The upstream loader warned that a shipped temperature for `choice:11+` was
below its allowed range and clamped it to 0.5. The four-option example does
not exercise that bucket. Treat affected probabilities as uncalibrated,
and evaluate all probability-based policies on your own held-out examples.

Running the benchmark's extra GPU checkpoint alongside an API preloading
all three checkpoints exhausted GPU memory during inference. The benchmark
correctly failed on the resulting CPU fallbacks. Stopping the API before
the benchmark produced the no-fallback results above. On an 8 GB GPU, start
with one resident model and a small batch, and measure the memory you actually use.

The foreground API and Linux systemd user-service paths are covered by this
recipe. Windows, macOS accelerator paths, optional fast kernels, long-document
aggregation, training, and real mailbox integration were not tested.

This public report contains only generic hardware/software metadata,
synthetic workloads, model revisions, and aggregate timings. It contains no
mailbox data, user names, server addresses, credentials, or private deployment files.
