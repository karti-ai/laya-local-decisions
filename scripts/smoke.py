#!/usr/bin/env python3
import argparse
import json
import urllib.error
from client import ROOT, check_health, request

parser = argparse.ArgumentParser(description='Synthetic API check; does not connect to an email provider')
parser.add_argument('--model', default='typed-decisions', choices=['english', 'multilingual', 'typed-decisions'])
parser.add_argument('--device', default='cuda', choices=['cpu', 'cuda'])
args = parser.parse_args()
health = check_health(args.device)
questions = json.loads((ROOT / 'examples/questions.json').read_text())
rows = [json.loads(line) for line in (ROOT / 'examples/emails.jsonl').read_text().splitlines() if line.strip()]
body = {'model': args.model, 'state': rows[0]['state'], 'questions': questions}
try:
    request('/v1/systemone', body, authenticated=False)
except urllib.error.HTTPError as error:
    assert error.code == 401, f'Expected 401, got {error.code}'
else:
    raise AssertionError('Unauthenticated inference was accepted')
single = request('/v1/systemone', body)
assert single['usage']['output_tokens'] == 0
batch = request('/v1/systemone/batch', {'model': args.model, 'states': [row['state'] for row in rows], 'questions': questions})
assert len(batch['results']) == len(rows)
predictions = []
for row, result in zip(rows, batch['results']):
    answer = result['answers']['category']
    assert answer['choice'] in questions['category']['criteria']
    assert abs(sum(answer['probabilities'].values()) - 1) < 0.01
    predictions.append({'id': row['id'], 'expected': row['expected'], 'predicted': answer['choice']})
health = check_health(args.device)
print(json.dumps({'authenticated_api': True, 'single_and_batch': True, 'checkpoint_devices': health['checkpoint_devices'], 'predictions': predictions, 'synthetic_correct': sum(r['expected'] == r['predicted'] for r in predictions), 'synthetic_total': len(rows), 'note': 'Four synthetic cases verify the API plumbing, not domain accuracy.'}, indent=2))
