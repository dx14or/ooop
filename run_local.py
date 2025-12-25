import argparse
import json
import sys
from pathlib import Path
from app.ingest import IngestError, ingest_channel_sync
from app.pipeline import run_from_path

def _print_text(result: dict) -> None:
    warning = result.get('warning')
    if warning:
        print(f'Warning: {warning}')
    labeling = result.get('labeling')
    if labeling:
        for key, stats in labeling.items():
            coverage = stats.get('coverage', 0.0)
            labeled = stats.get('labeled', 0)
            total = stats.get('total', 0)
            print(f'{key.capitalize()} labeling coverage: {coverage:.2%} ({labeled}/{total})')
    predictions = result.get('predictions', [])
    if not predictions:
        print('No predictions.')
        return
    print('Predicted next topics:')
    for idx, item in enumerate(predictions, start=1):
        topic_id = item.get('topic_id')
        prob = item.get('prob', 0.0)
        label = item.get('label') or item.get('terms', '')
        print(f'{idx}. topic={topic_id} prob={prob:.3f} label={label}')

def main() -> int:
    parser = argparse.ArgumentParser(description='Local topic prediction runner.')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--data-path', help='Path to JSONL/JSONL.GZ export.')
    group.add_argument('--channel-url', help='Telegram channel URL to ingest.')
    parser.add_argument('--json', action='store_true', help='Print raw JSON output.')
    args = parser.parse_args()
    if args.data_path:
        path = Path(args.data_path)
        if not path.exists():
            print(f'Data path not found: {path}', file=sys.stderr)
            return 1
    else:
        try:
            path = ingest_channel_sync(args.channel_url, Path(__file__).resolve().parent / 'data')
        except IngestError as exc:
            print(f'Telegram ingest failed: {exc}', file=sys.stderr)
            return 1
    result = run_from_path(path)
    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_text(result)
    return 0
if __name__ == '__main__':
    raise SystemExit(main())
