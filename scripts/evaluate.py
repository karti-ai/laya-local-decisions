#!/usr/bin/env python3
"""Score a local, labeled JSONL file and write only aggregate metrics."""
import argparse
import json
from collections import Counter
from pathlib import Path
from client import ROOT, check_health, request

parser = argparse.ArgumentParser()
parser.add_argument('data', type=Path)
parser.add_argument('--questions', type=Path, default=ROOT / 'examples/questions.json')
parser.add_argument('--model', default='typed-decisions', choices=['english', 'multilingual', 'typed-decisions'])
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
parser.add_argument('--threshold', type=float, default=0.9)
parser.add_argument('--batch-size', type=int, default=4)
parser.add_argument('--output', type=Path, default=ROOT / 'runs/evaluation.json')
args = parser.parse_args()
if not 0 <= args.threshold <= 1 or not 1 <= args.batch_size <= 64:
    parser.error('threshold must be 0..1 and batch-size must be 1..64')
questions = json.loads(args.questions.read_text())
criteria = questions['category']['criteria']
rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
if not rows:
    parser.error('data must contain labeled rows')
for row in rows:
    if row.get('expected') not in criteria or 'state' not in row:
        parser.error('every row requires state and an expected label from category.criteria')
check_health(args.device)
confusion = Counter()
accepted = correct = accepted_correct = 0
bins = [dict(n=0, correct=0, probability_sum=0.0) for _ in range(10)]
for offset in range(0, len(rows), args.batch_size):
    chunk = rows[offset:offset + args.batch_size]
    result = request('/v1/systemone/batch', {'model': args.model, 'states': [r['state'] for r in chunk], 'questions': questions})
    if len(result['results']) != len(chunk):
        raise RuntimeError('Batch response length mismatch')
    for row, prediction in zip(chunk, result['results']):
        answer = prediction['answers']['category']
        label = answer['choice']
        probability = answer['probabilities'][label]
        is_correct = label == row['expected']
        confusion[row['expected'], label] += 1
        correct += is_correct
        if probability >= args.threshold:
            accepted += 1
            accepted_correct += is_correct
        b = bins[min(int(probability * 10), 9)]
        b['n'] += 1
        b['correct'] += is_correct
        b['probability_sum'] += probability
check_health(args.device)
ece = sum(abs(b['correct'] - b['probability_sum']) for b in bins) / len(rows)
report = {'model': args.model, 'rows': len(rows), 'accuracy': correct / len(rows), 'threshold': args.threshold, 'coverage': accepted / len(rows), 'accepted': accepted, 'accepted_accuracy': accepted_correct / accepted if accepted else None, 'empirical_calibration_error_10_bins': ece, 'confusion': [{'expected': expected, 'predicted': predicted, 'count': n} for (expected, predicted), n in sorted(confusion.items())], 'note': 'No source text or IDs are saved. The threshold is an experiment, not a validated probability guarantee. This script makes no email changes.'}
args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
