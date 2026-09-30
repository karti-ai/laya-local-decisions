# Laya on your own hardware

A working deployment recipe for small engineering teams: run [Laya](https://github.com/NandhaKishorM/laya) locally, verify the actual compute device, and measure classification quality before connecting it to an automation.

**Verified on Linux with an RTX 2070 Super (8 GB), PyTorch 2.8.0 + CUDA 12.8, and Laya 0.3.22.** This repository provides setup, single and batch API checks, a CPU/GPU benchmark, and an evaluator for your own labeled examples. All bundled messages are invented. No mailbox integration, credentials, model weights, or automatic mail actions are included.

Read the companion walkthrough at [models.karti.ai/recipes/laya](https://models.karti.ai/recipes/laya).

## What it does

Laya evaluates text or JSON against questions you define. It returns a choice, a score, or a yes/no probability. It produces zero output text tokens. Useful candidates include inbox categories, ticket queues, refund intent, cancellation intent, and agent workflow routing.

An email parser still decodes MIME, HTML, headers, and attachments. Laya consumes the resulting text. Arbitrary field extraction, summaries, replies, and tool execution require other code or a generative model. It does not expose a chat-completions API and this recipe does not use vLLM or Ollama.

## Hardware and checkpoint choice

| Setup | Starting point |
| --- | --- |
| Linux + NVIDIA GPU | This recipe's tested CUDA path. One checkpoint and small batches first. |
| CPU | Same API and helper scripts; measure latency on your CPU. |
| Windows | Use WSL2 with functioning NVIDIA GPU access; this route was not tested here. |
| macOS / other accelerators | Use upstream's platform instructions; this recipe verifies CPU or CUDA only. |

| Checkpoint | Parameters | Starting use |
| --- | --- | --- |
| `english` | 421M | English classification; compare it on your examples. |
| `multilingual` | 322M | Text outside English and mixed-language workloads. |
| `typed-decisions` | 421M | Candidate for support, invoice, and agent-trace decisions. |

The default loads only `typed-decisions`. Loading all three worked on the tested 8 GB GPU; memory depends on input length, batch size, and other GPU processes. Allow several GB of host RAM and roughly 10 GB of free disk for the Python environment, CUDA libraries, and downloads. Upstream's Unsloth interface lists 4–5 GB RAM minimums for individual models. Download size is not a VRAM budget.

## Quick start

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) and a compatible NVIDIA driver. You do not need Docker for this recipe.

```bash
git clone https://github.com/karti-ai/laya-local-decisions
cd laya-local-decisions
git checkout v0.1.0

# Creates an isolated Python environment and a private .env with a random key.
# CUDA is verified before the setup succeeds.
scripts/setup.sh cuda

# Or choose CPU instead, in a separate checkout/environment:
# scripts/setup.sh cpu

# First startup downloads the selected checkpoint and keeps it loaded.
scripts/serve.sh
```

The setup uses Python 3.13 by default and installs the pinned runtime in `requirements.txt`. `uv` can obtain Python if needed. `LAYA_PYTHON` selects another compatible interpreter. The helper scripts use only Python's standard library, apart from the model benchmark.

The default URL is `http://127.0.0.1:8000`. Keep the server terminal open. In another terminal:

```bash
.venv/bin/python scripts/smoke.py --model typed-decisions --device cuda
.venv/bin/python scripts/evaluate.py examples/emails.jsonl \
  --model typed-decisions --threshold 0.9
```

For a CPU install, pass `--device cpu` to both commands. The examples verify plumbing; four easy cases cannot establish useful domain accuracy.

### Make an API request

The helper client reads the key from `.env`; there is no need to paste it into a command or notebook.

```bash
.venv/bin/python - <<'PY'
import json, sys
sys.path.insert(0, 'scripts')
from client import request

result = request('/v1/systemone', {
    'model': 'typed-decisions',
    'state': 'The customer was charged twice and requests a refund.',
    'questions': {
        'team': {
            'type': 'choice',
            'instructions': 'Which team should handle this?',
            'criteria': {
                'billing': 'Payments, invoices, and refunds',
                'technical': 'Bugs and outages',
                'other': 'Everything else'
            }
        }
    }
})
print(json.dumps(result, indent=2))
PY
```

`POST /v1/systemone/batch` accepts a `states` array with the same questions and model. Start with batches of four and measure peak memory before increasing the batch size. One API call containing many states and questions can multiply the work considerably.

## Verify acceleration; do not infer it from a setting

`LAYA_DEVICE=cuda` requests CUDA. Upstream can fall back to CPU after a GPU error. Our smoke and evaluation scripts check `/health` before and after inference: resident checkpoint devices must match your requested device, and CPU fallback counters must stay zero. Unauthenticated inference must return 401.

```bash
curl --fail http://127.0.0.1:8000/health
nvidia-smi
```

The CUDA wheel in this recipe supports Turing (`sm_75`). Newer is not always compatible with older GPUs: verify the installed wheel and driver instead of changing only a CUDA environment variable. The Laya runtime selects FP16 on pre-Ampere NVIDIA cards. BF16 should not be forced on a 2070 Super. Optional TileLang/FlashAttention fast paths are outside this tested recipe.

To preload more checkpoints, edit `.env` and restart:

```dotenv
LAYA_MODELS=english,multilingual,typed-decisions
LAYA_MAX_LOADED=3
```

`LAYA_MODELS` controls preloading, not the set of models clients may request. Specify `model` in requests to compare checkpoints. Otherwise the upstream router may select a different checkpoint based on the language. `LAYA_REVISION=reviewed` opts into upstream's reviewed checkpoint revisions, and health reports the actual loaded SHAs. Exact dependency versions are recorded locally in `.venv/requirements.lock.txt`.

## Measure CPU versus GPU

Stop the API before running this benchmark on an 8 GB GPU. It loads an additional checkpoint on each device in turn; it does not benchmark through the running API. The reference verification records an out-of-memory failure when it was run alongside all three preloaded API models.

```bash
.venv/bin/python scripts/benchmark.py \
  --model typed-decisions --devices cpu,cuda \
  --iterations 12 --threads 4 --output runs/benchmark.json
```

Both devices receive identical invented messages and questions. The benchmark warms both shapes, synchronizes CUDA, and reports median and p95 single-state time, batch time, and batch throughput. It refuses a CUDA-to-CPU substitution. HTTP/network time, Gmail API time, and download/startup time are excluded. See [the verification notes](docs/VERIFICATION.md) for the recorded reference run.

## Evaluate your task before automating it

1. Define a small set of categories with distinct descriptions. Start with 4–10; large option sets share a limited prompt budget.
2. Label 100–200 representative examples, including ambiguous messages, mixed purposes, long threads, and languages you actually receive. Keep development examples separate from a final holdout.
3. Compare checkpoints with the same questions. Inspect each category's errors, coverage at candidate probability thresholds, and calibration.
4. Choose the threshold on development data and measure it once on the holdout. A displayed `0.9` is a model estimate, not proof of 90% correctness.
5. Begin with suggestions and review. Automate a reversible action only when its measured error rate is acceptable for that action.

Store your own examples locally as JSONL, outside version control:

```json
{"id":"local-case-001","state":"A message you have labeled locally","expected":"account"}
```

```bash
mkdir -p data/private
# Put your own labeled holdout in data/private/holdout.jsonl.
.venv/bin/python scripts/evaluate.py data/private/holdout.jsonl \
  --questions examples/questions.json --model typed-decisions \
  --threshold 0.9 --batch-size 4 --output runs/holdout.json
```

`category` is the evaluator's scored question. Its expected values must match your choice keys. It reports accuracy, a confusion table, threshold coverage, accuracy among accepted predictions, and a ten-bin empirical calibration error. It uses the predicted label's probability, rather than the upstream entropy-based `confidence` field. It stores only aggregate metrics, never source text or IDs. It makes no provider calls except to your configured Laya endpoint and never modifies mail.

The initial model training does not guarantee calibrated predictions on your domain. Upstream documents weak zero-shot results on some workflow benchmarks, option-label effects, and temperature-fitting requirements. Specialized weights can help, but checkpoint selection and any fine-tuning must be evaluated on unseen data. See [the model card's limitations](https://huggingface.co/convaiinnovations/laya#honest-limits) and [the calibration interface](https://github.com/NandhaKishorM/laya/blob/main/laya/calibrate.py).

Short context is another constraint: prompts/options consume part of the 512/1024-token default windows. A long email thread may be truncated. Decide which parts of a thread matter, or evaluate the upstream long-document path; do not silently equate a clipped excerpt with the whole conversation.

## Connect it to an engineering workflow

```text
Provider API → MIME / HTML cleanup → reviewed deterministic rules
                                       ↓ unmatched examples
                                 local Laya API
                                       ↓
                         category + probabilities
                                       ↓
                          review / allowed action
```

For mail, use provider APIs or your existing connector to obtain text, deduplicate message IDs, cache classification results, and record the model revision and question schema. Keep provider rate limits and retries separate from inference concurrency. Existing sender rules remain useful; the model handles cases where text matters.

An inbox with 100,000 messages is mostly a data-ingestion and evaluation problem. Benchmark a realistic sample and extrapolate measured batch throughput; do not promise a completion time from a GPU microbenchmark. If you later enable archive or label actions, keep an action log and an undo path. Decide how important/starred messages, receipts, account notices, and attachments should be handled before enabling those actions. Treat message text as data, with a fixed category schema and an independently defined action policy.

## Run as a service

On a Linux host with a systemd user session, a transient background service is enough for a pilot:

```bash
systemd-run --user --unit=laya-local \
  --property=Restart=on-failure "$PWD/scripts/serve.sh"
journalctl --user -u laya-local -f
systemctl --user stop laya-local
```

For startup after reboot, create a persistent user unit pointing to the absolute `scripts/serve.sh` path, enable it, and ensure the user session persists as required by your operating environment. The wrapper resolves its own checkout and loads the local `.env`.

Localhost is the default. For team access, bind to your private VPN address, retain the API key, and use a tunnel or TLS reverse proxy appropriate to your network. Keep `.env`, provider tokens, real messages, and action logs out of Git. This repository ignores them by default.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| CUDA unavailable or unsupported GPU | NVIDIA driver, installed wheel architecture, and a tiny CUDA tensor operation. |
| Server answers but is slow | `/health` actual devices and CPU fallback counts; startup and model eviction differ from warm inference. |
| Import/load hangs | The wrapper sets `USE_TF=0`; use the isolated environment instead of a shared TensorFlow environment. |
| First request is slow | Model downloads and construction happen once. Preload before measuring. |
| Model changes cause repeated loading | Preload the checkpoints you serve and increase `LAYA_MAX_LOADED` to match. |
| Temperature warning | Some shipped temperature entries are clamped. Treat affected probabilities as uncalibrated and evaluate your own thresholds. |
| OOM or CPU fallback | Reduce batch size/questions/context or unload checkpoints; rerun health and smoke checks. |
| Confident errors | Inspect your labeled holdout, label descriptions, context truncation, language routing, and calibration. |

## Sources and license

- [Laya source, benchmarks, and limitations](https://github.com/NandhaKishorM/laya)
- [Laya model card and weights](https://huggingface.co/convaiinnovations/laya)
- [Unsloth GUI alternative](https://unsloth.ai/docs/models/decision-laya)
- [PyTorch installation](https://pytorch.org/get-started/locally/)
- [uv installation](https://docs.astral.sh/uv/getting-started/installation/)

Recipe and helper code: Apache-2.0. Upstream software and weights keep their separate licenses and attribution; see [NOTICE](NOTICE).
