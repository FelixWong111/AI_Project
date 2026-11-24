"""Strict P06 Labelme rectangle import. No inference, splitting or PASS labels."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile

from PIL import Image, ImageOps

CLASSES = ('non_catalogue', 'long_screw', 'thick_screw', 'thin_screw', 'hex_nut', 'flat_washer')
IMAGE_SUFFIXES = {'.png', '.jpg', '.jpeg', '.bmp'}


class DataError(ValueError):
    def __init__(self, code, message, suggestion='Correct the source annotation and run check again.'):
        super().__init__(message)
        self.code, self.suggestion = code, suggestion


def require(condition, code, message, suggestion='Correct the source annotation and run check again.'):
    if not condition:
        raise DataError(code, message, suggestion)


def _dimensions(width, height):
    require(all(type(v) is int and v > 0 for v in (width, height)),
            'INVALID_DIMENSIONS', 'Image dimensions must be positive integers.')


def _numbers(values, count):
    require(isinstance(values, (tuple, list)) and len(values) == count,
            'INVALID_COORDINATES', f'Expected {count} coordinates.')
    require(all(type(v) in (int, float) for v in values),
            'INVALID_COORDINATES', 'Coordinates must be numeric, excluding bool.')
    require(all(math.isfinite(v) for v in values), 'NONFINITE', 'NaN/Inf coordinates are forbidden.')


def _box(box, width, height):
    _dimensions(width, height)
    _numbers(box, 4)
    x1, y1, x2, y2 = box
    require(x1 < x2 and y1 < y2, 'ZERO_AREA', 'Box must have positive width and height.')
    require(0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height,
            'OUT_OF_BOUNDS', 'Box exceeds original-image continuous boundaries; no automatic clipping.')


def to_yolo(xyxy, width, height):
    _box(xyxy, width, height)
    x1, y1, x2, y2 = xyxy
    return ((x1+x2)/(2*width), (y1+y2)/(2*height), (x2-x1)/width, (y2-y1)/height)


def from_yolo(xywh, width, height, *, decimal_places=None):
    _dimensions(width, height)
    _numbers(xywh, 4)
    require(all(0 <= v <= 1 for v in xywh), 'OUT_OF_BOUNDS', 'YOLO coordinates must be in [0,1].')
    cx, cy, w, h = xywh
    box = [(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height]
    # Explicit decimal serialization permits only its derived sub-micro-pixel error.
    # Default legacy behavior and strict source _box/to_yolo validation are unchanged.
    epsilon = [1e-7]*4
    if decimal_places is not None:
        require(type(decimal_places) is int and 9 <= decimal_places <= 16,
                'INVALID_PRECISION', 'Decimal precision must be explicitly between 9 and 16.')
        for i,limit in enumerate((width,height,width,height)):
            quantization = .75 * 10**(-decimal_places) * limit + 1e-9
            require(quantization <= 1e-6, 'INVALID_PRECISION', 'Serialization uncertainty exceeds 1e-6 pixel.')
            epsilon[i] = max(epsilon[i],quantization)
    for i, limit in enumerate((width, height, width, height)):
        if -epsilon[i] <= box[i] < 0:
            box[i] = 0.0
        elif limit < box[i] <= limit + epsilon[i]:
            box[i] = float(limit)
    _box(box, width, height)
    return box


def _metadata(value):
    require('annotation_status' not in value, 'UNAGREED_STATUS',
            'annotation_status semantics are not agreed.',
            'Ask P03 to confirm the status contract; do not convert REVIEW into an empty scene.')
    require('ignore' not in value, 'UNAGREED_IGNORE', 'ignore semantics are not agreed.',
            'Ask P03/P08 to define ignore handling; do not silently drop objects.')
    require(isinstance(value.get('flags', {}), dict) and not value.get('flags', {}),
            'UNAGREED_FLAGS', 'Nonempty or malformed flags require an agreed annotation policy.')


def _load_labelme_document(path):
    try:
        doc = json.loads(path.read_text(encoding='utf-8-sig'))
    except (ValueError, OSError) as exc:
        raise DataError('INVALID_JSON', str(exc)) from exc
    require(isinstance(doc, dict), 'INVALID_SCHEMA', 'Labelme JSON must be an object.')
    _metadata(doc)
    return doc


def _read_annotation(path, confirmed_empty):
    doc = _load_labelme_document(path)
    name = doc.get('imagePath')
    require(isinstance(name, str) and name and '/' not in name and '\\' not in name
            and ':' not in name and name not in {'.', '..'},
            'UNSAFE_IMAGE_PATH', 'imagePath must be a local basename, without traversal or absolute paths.')
    image = path.parent / name
    require(image.stem == path.stem and image.suffix.lower() in IMAGE_SUFFIXES,
            'IMAGE_PAIR_MISMATCH', 'JSON and image must have the same stem and a supported image extension.')
    require(image.is_file(), 'MISSING_IMAGE', f'Corresponding image is missing: {name}',
            'Provide the original image alongside its JSON; embedded imageData is not a replacement.')
    require(not image.is_symlink() and image.resolve().parent == path.parent.resolve(),
            'UNSAFE_IMAGE_PATH', 'Image must stay inside the input directory.')
    try:
        with Image.open(image) as im:
            require(im.getexif().get(274, 1) == 1, 'EXIF_ORIENTATION',
                    'Nontrivial EXIF orientation makes annotation coordinates ambiguous.',
                    'Have P03 normalize image orientation and recheck boxes before import.')
            im.load()  # Decode, not just header inspection.
            width, height = im.size
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            digest = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
    except (OSError, ValueError, Image.DecompressionBombError) as exc:
        if isinstance(exc, DataError):
            raise
        raise DataError('INVALID_IMAGE', str(exc), 'Replace the corrupt/unsupported image and recheck.') from exc
    _dimensions(doc.get('imageWidth'), doc.get('imageHeight'))
    require((width, height) == (doc['imageWidth'], doc['imageHeight']),
            'DIMENSION_MISMATCH', f'Image is {width}x{height}; JSON dimensions disagree.')
    objects = _parse_shapes(doc, width, height, name, confirmed_empty)
    return dict(image_id=name, source_json=str(path.resolve()), source_image=str(image.resolve()),
                width=width, height=height, pixel_sha256=digest, scene_id=None, split=None, objects=objects)


def _parse_shapes(doc, width, height, name, confirmed_empty):
    """Validate Labelme rectangles in an explicitly established coordinate frame."""
    _dimensions(width, height)
    shapes = doc.get('shapes')
    require(isinstance(shapes, list), 'INVALID_SCHEMA', 'shapes must be a list.')
    require(shapes or name in confirmed_empty, 'UNCONFIRMED_EMPTY', 'Empty annotations are not verified empty scenes.',
            'Ask P03 to verify the empty scene, then explicitly pass --confirmed-empty IMAGE_ID.')
    objects, boxes = [], {}
    for index, shape in enumerate(shapes):
        require(isinstance(shape, dict), 'INVALID_SCHEMA', f'shapes[{index}] must be an object.')
        _metadata(shape)
        require(shape.get('group_id') is None, 'UNAGREED_GROUP', 'Grouped shapes require an agreed object policy.')
        require(shape.get('shape_type') == 'rectangle', 'UNSUPPORTED_SHAPE', f'shapes[{index}] is not rectangle.')
        label = shape.get('label')
        require(isinstance(label, str) and label in CLASSES, 'UNKNOWN_LABEL', f'Unknown label at shapes[{index}]: {label!r}',
                'Use exactly one of the six catalogue names; no aliases or index offsets are inferred.')
        points = shape.get('points')
        require(isinstance(points, list) and len(points) == 2, 'INVALID_COORDINATES', 'Rectangle requires two points.')
        for point in points:
            _numbers(point, 2)
        (xa, ya), (xb, yb) = points
        box = [min(xa, xb), min(ya, yb), max(xa, xb), max(ya, yb)]
        _box(box, width, height)
        key = tuple(box)
        require(key not in boxes, 'DUPLICATE_OBJECT' if boxes.get(key) == label else 'CONFLICTING_OBJECT',
                f'Duplicate or conflicting box at shapes[{index}].', 'Have P03 review overlapping duplicate labels.')
        boxes[key] = label
        objects.append(dict(object_id=index, class_id=CLASSES.index(label), xyxy=box, ignore=0))
    return objects


def _manifest(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        reader = csv.DictReader(stream)
        require(reader.fieldnames is not None and {'image_id', 'scene_id', 'split'} <= set(reader.fieldnames),
                'INVALID_MANIFEST', 'CSV requires image_id,scene_id,split columns.')
        rows = {}
        scenes = {}
        for row in reader:
            image_id, scene, split = (row.get(k) for k in ('image_id', 'scene_id', 'split'))
            require(all(isinstance(v, str) and v and v == v.strip() for v in (image_id, scene, split))
                    and split in {'train', 'val', 'test'}, 'INVALID_PARTITION',
                    f'Incomplete or invalid partition row: {row!r}', 'Ask P03 for an explicit scene_id and train/val/test assignment.')
            require(image_id not in rows, 'DUPLICATE_IMAGE_ID', f'Duplicate manifest image_id: {image_id}')
            require(scene not in scenes or scenes[scene] == split, 'SCENE_LEAKAGE', f'Scene {scene} crosses partitions.')
            rows[image_id] = dict(scene_id=scene, split=split)
            scenes[scene] = split
    return rows


def inspect_dataset(input_dir, manifest=None, confirmed_empty=()):
    """Audit one flat directory; no split is inferred. Errors prevent all export."""
    root = Path(input_dir)
    report = dict(ok=False, input_dir=str(root.resolve()), partition_verified=False,
                  evidence='SYNTHETIC_OR_USER_SUPPLIED_ANNOTATIONS_NOT_MODEL_PREDICTIONS', records=[], errors=[])

    def error(file, exc):
        report['errors'].append(dict(file=str(file), type=exc.code, message=str(exc), suggestion=exc.suggestion))

    if not root.is_dir():
        error(root, DataError('MISSING_INPUT', 'Input directory does not exist.'))
        return report
    paths = sorted(root.iterdir())
    for sub in root.iterdir():
        if sub.is_dir() and any(p.suffix.lower() in IMAGE_SUFFIXES | {'.json'} for p in sub.rglob('*') if p.is_file()):
            error(sub, DataError('NESTED_INPUT', 'Only a flat Labelme directory is supported.', 'Choose the leaf directory explicitly.'))
    annotations = [p for p in paths if p.suffix.lower() == '.json']
    if not annotations:
        error(root, DataError('NO_ANNOTATIONS', 'No Labelme JSON files found.'))
    for path in annotations:
        try:
            require(not path.is_symlink(), 'UNSAFE_JSON_PATH', 'Symlink annotations are not supported.')
            report['records'].append(_read_annotation(path, set(confirmed_empty)))
        except DataError as exc:
            error(path, exc)
    stems = {p.stem for p in annotations}
    image_stems = {}
    for path in paths:
        if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file():
            if path.stem not in stems:
                error(path, DataError('MISSING_JSON', 'Corresponding JSON is missing.', 'Provide complete Labelme annotation; do not treat as an empty scene.'))
            if path.stem.casefold() in image_stems:
                error(path, DataError('AMBIGUOUS_STEM', 'Multiple images share one label filename stem.'))
            image_stems[path.stem.casefold()] = path.name
    if manifest is not None:
        try:
            rows = _manifest(manifest)
            records = report['records']
            for record in records:
                require(record['image_id'] in rows, 'MISSING_MANIFEST_ROW', f"Missing manifest entry: {record['image_id']}")
                record.update(rows[record['image_id']])
            for image_id in rows:
                require((root / image_id).is_file() and image_id in {r['image_id'] for r in records},
                        'MISSING_MANIFEST_IMAGE', f'Manifest image is missing or has invalid annotation: {image_id}')
            hashes = {}
            for record in records:
                digest = record['pixel_sha256']
                require(digest not in hashes or hashes[digest] == record['split'], 'IMAGE_LEAKAGE',
                        f"Identical decoded image crosses partitions: {record['image_id']}")
                hashes[digest] = record['split']
            report['partition_verified'] = not report['errors']
        except (DataError, OSError, UnicodeError, csv.Error) as exc:
            error(manifest, exc if isinstance(exc, DataError) else DataError('INVALID_MANIFEST', str(exc)))
    report['ok'] = not report['errors']
    return report


def export_dataset(report, output_dir):
    """Export labels and xyxy records; test records never enter this output."""
    require(report['ok'], 'INVALID_DATASET', 'Dataset has errors; export is blocked.')
    out = Path(output_dir)
    source = Path(report['input_dir']).resolve()
    target = out.resolve()
    require(source != target and source not in target.parents and target not in source.parents,
            'UNSAFE_OUTPUT', 'Input and output directories must not contain each other.')
    if out.exists():
        raise FileExistsError(f'Output already exists: {out}')
    out.parent.mkdir(parents=True, exist_ok=True)
    # Write on the same volume; Windows rename refuses an existing destination.
    # TemporaryDirectory only cleans its own newly allocated staging directory.
    with tempfile.TemporaryDirectory(prefix='.p06-export-', dir=out.parent) as temporary:
        staged = Path(temporary) / 'complete'
        staged.mkdir()
        _write_export(report, staged)
        if out.exists():
            raise FileExistsError(f'Output already exists: {out}')
        staged.rename(out)


def _write_export(report, out):
    records = [r for r in report['records'] if r['split'] != 'test']
    for record in records:
        directory = out / 'labels' / (record['split'] or 'unpartitioned')
        directory.mkdir(parents=True, exist_ok=True)
        lines = []
        for obj in record['objects']:
            coords = to_yolo(obj['xyxy'], record['width'], record['height'])
            lines.append(str(obj['class_id']) + ' ' + ' '.join(format(v, '.12g') for v in coords))
        (directory / (Path(record['image_id']).stem + '.txt')).write_text(
            '\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    (out / 'objects.json').write_text(json.dumps(records, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    summary = {k: v for k, v in report.items() if k != 'records'}
    summary.update(exported_images=len(records), excluded_test_images=len(report['records'])-len(records),
                   class_names=list(CLASSES), coordinate_convention='continuous original-image xyxy; 0 <= x <= width, 0 <= y <= height',
                   training_ready=False, note='Labels only; no images copied, no split created, no training configuration generated.')
    (out / 'conversion.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
