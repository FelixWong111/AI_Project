"""Validate only explicitly selected train/validation records, never test content."""
from __future__ import annotations

import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

from PIL import Image, ImageOps
from src.work_orders import reference_counts
from src.adapters.baseline2 import image_filename, validate_counts

ROOT = Path(__file__).resolve().parents[2]
COUNT_FIELDS = ('long_screw_count', 'thick_screw_count', 'thin_screw_count',
                'hex_nut_count', 'flat_washer_count')
SKIP = {'REVIEW', 'UNJUDGEABLE', 'IGNORE', 'IGNORED', 'ERROR'}
VALID = {'REVIEWED', 'OK', 'VALID', 'VERIFIED', 'COMPLETE'}


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    raw = Path(path).read_bytes()
    for encoding in ('utf-8-sig', 'gb18030'):
        try:
            return list(csv.DictReader(raw.decode(encoding).splitlines()))
        except UnicodeDecodeError:
            continue
    raise ValueError(f'Cannot decode CSV without data loss: {path}')


def discover_development():
    # Inspect only first-level handoff folder names; never recurse into test assets.
    packages = list(ROOT.glob('P03_*'))
    found = []
    for package in packages:
        for base in (package, *[p for p in package.iterdir() if p.is_dir()]):
            for part in base.iterdir():
                if part.is_dir() and part.name.startswith('01_') and (part/'ground_truth_images.csv').is_file():
                    found.append(part)
    if len(found) != 1:
        raise ValueError(f'Expected one P03 development package; use --data-dir explicitly. Found {len(found)}')
    return found[0]


def prepare(data_dir, settings):
    """Return audited records and evidence without altering the source package."""
    data_dir = Path(data_dir).resolve()
    if (data_dir/'dataset_v2.json').is_file():
        from src.data.dataset_v2 import prepare_development
        return prepare_development(data_dir, settings)
    fixed = reference_counts()
    if 'expected_counts' in settings and validate_counts(settings['expected_counts']) != fixed:
        raise ValueError('Training settings conflict with the common fixed reference')
    if settings.get('reference_label_field') != 'reference_label_all_one':
        raise ValueError('This P03 label field documents only the all-one reference; confirm another rule explicitly')
    confirmation = settings.get('scene_confirmation', {})
    accept_unknown = confirmation.get('each_image_is_distinct_scene') is True and bool(confirmation.get('source'))
    records, exclusions, ids, scene_splits = [], [], set(), {}
    images_csv = data_dir/'ground_truth_images.csv'
    objects_csv = data_dir/'ground_truth_objects.csv'
    for row in read_csv(images_csv):
        raw_split = row['split'].strip().lower()
        if raw_split not in ('train', 'val', 'validation'):
            continue  # Filter before resolving or opening any image.
        split = 'validation' if raw_split in ('val', 'validation') else 'train'
        image_id = image_filename(row['image_id'])
        if image_id in ids:
            raise ValueError(f'Duplicate image_id: {image_id}')
        ids.add(image_id)
        status = row['annotation_status'].strip().upper()
        if status in SKIP:
            exclusions.append(dict(image_id=image_id, split=split, reason=status)); continue
        if status not in VALID:
            raise ValueError(f'Unrecognized annotation_status: {image_id}: {status}')
        scene = row['scene_id'].strip()
        unknown = scene.upper() in ('', 'TBD', 'UNKNOWN', 'N/A')
        if unknown and not accept_unknown:
            raise ValueError('Scene identity is unconfirmed; supply documented confirmation before training')
        # Preserve source scene_id. The grouping key expresses the explicit user confirmation,
        # and is not written back or presented as a capture-record SceneID.
        key = ('confirmed_image:' + image_id) if unknown else scene
        if scene_splits.setdefault(key, split) != split:
            raise ValueError(f'Confirmed scene crosses partitions: {key}')
        counts = [int(row[field]) for field in COUNT_FIELDS] + [int(row['non_catalogue_count'])]
        if any(n < 0 for n in counts):
            raise ValueError(f'Negative counts: {image_id}')
        label = 'PASS' if counts[:5] == list(fixed.values()) and counts[5] == 0 else 'FAIL'
        if row[settings['reference_label_field']].strip().upper() != label:
            raise ValueError(f'Label/count conflict: {image_id}')
        path = data_dir/'images'/raw_split/image_id
        if not path.is_file():
            raise FileNotFoundError(path)
        with Image.open(path) as source:
            image = ImageOps.exif_transpose(source).convert('RGB')
            if image.size != (int(row['width']), int(row['height'])):
                raise ValueError(f'Image dimensions conflict: {image_id}')
            pixels = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
        records.append(dict(image_id=image_id, scene_id=scene, scene_group=key, split=split,
                            label=label, counts=counts, path=path.relative_to(ROOT).as_posix(),
                            width=int(row['width']), height=int(row['height']),
                            sha256=digest(path), pixel_sha256=pixels))
    by_id = {r['image_id']: r for r in records}
    actual, objects_seen = defaultdict(Counter), set()
    for obj in read_csv(objects_csv):
        iid = obj['image_id']
        if iid not in by_id:
            continue
        cid, ignore = int(obj['class_id']), int(obj['ignore'])
        identity = (iid, obj['object_id'])
        if not obj['object_id'] or identity in objects_seen or cid not in range(6) or ignore not in (0, 1):
            raise ValueError(f'Invalid object: {identity}')
        objects_seen.add(identity)
        x1,y1,x2,y2 = [float(obj[k]) for k in ('xmin','ymin','xmax','ymax')]
        r = by_id[iid]
        if not (0 <= x1 < x2 <= r['width'] and 0 <= y1 < y2 <= r['height']):
            raise ValueError(f'Invalid box: {identity}')
        if not ignore: actual[iid][cid] += 1
    for r in records:
        if [actual[r['image_id']][c] for c in (1,2,3,4,5,0)] != r['counts']:
            raise ValueError(f'Object/image count mismatch: {r["image_id"]}')
    # No automatic relabelling or silent split changes; duplicates fail closed.
    for field in ('sha256', 'pixel_sha256'):
        seen = {}
        for r in records:
            if r[field] in seen:
                raise ValueError(f'Duplicate {field}: {seen[r[field]]} / {r["image_id"]}')
            seen[r[field]] = r['image_id']
    records.sort(key=lambda r:(r['split'], r['image_id']))
    summary = {}
    for split in ('train', 'validation'):
        subset = [r for r in records if r['split']==split]
        counts = Counter(r['label'] for r in subset)
        if set(counts) != {'PASS', 'FAIL'}: raise ValueError(f'{split} needs both labels')
        summary[split] = dict(images=len(subset), scenes=len({r['scene_group'] for r in subset}), **counts)
    evidence = dict(summary=summary, exclusions=exclusions, expected_counts=fixed,
                    scene_confirmation=confirmation, source_scene_ids='preserved, including TBD',
                    official_held_out_test='NOT_RUN', source_files={
                        images_csv.relative_to(ROOT).as_posix():digest(images_csv),
                        objects_csv.relative_to(ROOT).as_posix():digest(objects_csv)})
    return records, evidence
