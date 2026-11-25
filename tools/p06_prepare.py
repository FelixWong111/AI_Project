"""P06 data audit/conversion CLI. Never trains or assigns partitions."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data.p06_labelme import DataError, export_dataset, inspect_dataset


def main(argv=None):
    parser = argparse.ArgumentParser(description='Check or convert Labelme rectangle annotations (P06).')
    parser.add_argument('command', choices=['check', 'convert'])
    parser.add_argument('--input', required=True, type=Path, help='Flat directory containing original images and JSON files')
    parser.add_argument('--output', type=Path, help='New labels/xyxy output directory; existing paths are refused')
    parser.add_argument('--report', type=Path, help='New JSON audit report file; existing files are refused')
    parser.add_argument('--manifest', type=Path, help='Optional CSV: image_id,scene_id,split; no automatic splitting')
    parser.add_argument('--confirmed-empty', action='append', default=[], metavar='IMAGE_ID',
                        help='Explicit human confirmation of an empty scene; repeat for multiple images')
    args = parser.parse_args(argv)
    if args.command == 'convert' and args.output is None:
        parser.error('convert requires --output')
    try:
        if args.output is not None and args.output.exists():
            raise FileExistsError(f'Output already exists: {args.output}')
        if args.report is not None:
            if args.report.exists():
                raise FileExistsError(f'Report already exists: {args.report}')
            root = args.input.resolve()
            require_path = args.report.resolve()
            if require_path == root or root in require_path.parents:
                raise ValueError('Report must be outside the source directory.')
        report = inspect_dataset(args.input, args.manifest, args.confirmed_empty)
        if args.report is not None:
            with args.report.open('x', encoding='utf-8') as stream:
                json.dump(report, stream, ensure_ascii=False, indent=2, allow_nan=False)
        print(json.dumps(report, ensure_ascii=True, indent=2, allow_nan=False))
        if not report['ok']:
            return 1
        if args.command == 'convert':
            export_dataset(report, args.output)
        return 0
    except (DataError, OSError, ValueError) as exc:
        print(f'{type(exc).__name__}: {exc}', file=sys.stderr)
        return 2


if __name__ == '__main__':
    raise SystemExit(main())
