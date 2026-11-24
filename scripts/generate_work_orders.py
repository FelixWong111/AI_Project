"""Generate input requirements from filenames/splits, never from actual counts."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from src.data.development import ROOT, read_csv
from src.work_orders import COUNT_KEYS, reference_counts, validate_cases, validate_counts


def generate(manifest, split=None, counts=None, purpose='common', mapping_csv=None):
    manifest = Path(manifest)
    if manifest.suffix.lower() == '.csv':
        rows = read_csv(manifest)
    else:
        payload = json.loads(manifest.read_text(encoding='utf-8-sig'))
        rows = payload['records'] if isinstance(payload, dict) else payload
    mapping = {}
    if mapping_csv is not None:
        for row in read_csv(mapping_csv):
            key, name = row['case_id'], row['image_id']
            if key in mapping:
                raise ValueError(f'Duplicate mapping key: {key}')
            mapping[key] = name
    requested = reference_counts() if counts is None else validate_counts(counts)
    if purpose not in ('common', 'functional'):
        raise ValueError('purpose must be common or functional')
    if purpose == 'common' and requested != reference_counts():
        raise ValueError('Common evaluation must use the fixed reference counts')
    cases = []
    for row in rows:
        actual_split = str(row.get('split', '')).lower()
        if actual_split == 'val':
            actual_split = 'validation'
        if split is not None and actual_split != split:
            continue
        # No access to labels, actual counts, boxes, or image pixels.
        source_id = row['image_id']
        iid = mapping[source_id] if source_id in mapping else source_id
        cases.append(dict(image_id=iid, expected_counts=dict(requested)))
    return sorted(validate_cases(cases), key=lambda row: row['image_id'])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'outputs/training/data/manifest.json')
    parser.add_argument('--split', choices=('train', 'validation', 'test'))
    parser.add_argument('--purpose', choices=('common', 'functional'), default='common')
    parser.add_argument('--counts', type=int, nargs=5)
    parser.add_argument('--mapping-csv', type=Path, help='Explicit case_id,image_id CSV for legacy case IDs')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    output = args.output.resolve()
    if not output.is_relative_to(ROOT) or any(p.startswith('P03_') for p in output.relative_to(ROOT).parts):
        parser.error('Write outside P03, inside the project')
    if output.exists():
        parser.error('Use a new output file; existing requirements are not overwritten')
    try:
        counts = None if args.counts is None else dict(zip(COUNT_KEYS, args.counts))
        cases = generate(args.manifest, args.split, counts, args.purpose, args.mapping_csv)
    except (ValueError, KeyError, OSError) as exc:
        parser.error(str(exc))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(cases, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')
    print(f'{len(cases)} cases ({args.purpose}) -> {output.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
