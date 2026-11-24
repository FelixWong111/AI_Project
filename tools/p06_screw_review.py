"""Build a Train-only comparison and an offline manual line annotation page."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import shutil
from uuid import uuid4

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.p06_geometry import measure_object
from tools.p06_measure_candidate import load_display, sha256
from tools.p06_measure_geometry import SOURCE, catalogue_records, crop_for_object, result_global, save

HISTORY = ROOT / 'output/p06_geometry_stage1'
OUTPUT = ROOT / 'output/p06_screw_review'
KNOWN = ('fail113_006', 'fail116_007')
PHASES = ('before', 'after', 'body_profile')
KEYS = ('underhead_length', 'thread_major_diameter')


def compare_marks(queue, rows, annotations):
    """Compare only explicit human judgements. Missing/incomplete lines stay pending."""
    if (annotations.get('queue_id') != queue['queue_id'] or annotations.get('version') != 1
            or annotations.get('coordinates') != 'original_display_xy'):
        raise ValueError('Annotation file belongs to a different queue or coordinate system')
    tasks = {t['object_id']: t for t in queue['tasks']}
    results = {r['object_id']: r for r in rows}
    marks = annotations['marks']
    if not isinstance(marks, dict) or marks.keys() - tasks.keys():
        raise ValueError('Unknown object in annotation file')
    scale = np.asarray(queue['scale_px_per_mm'])
    comparisons = []
    for oid, task in tasks.items():
        fields = marks.get(oid, {})
        if not isinstance(fields, dict) or fields.keys() - set(KEYS):
            raise ValueError(f'Invalid measurement fields: {oid}')
        for key in KEYS:
            state = 'pending'
            if key in fields:
                if fields[key] is None:
                    state = 'unclear'
                else:
                    pts = np.asarray(fields[key], dtype=float)
                    x,y,r,b = task['crop_xyxy']
                    if (pts.shape not in ((1,2),(2,2)) or not np.isfinite(pts).all()
                            or (pts < [x,y]).any() or (pts >= [r,b]).any()):
                        raise ValueError(f'Invalid endpoints: {oid} {key}')
                    if len(pts) == 2:
                        if np.array_equal(pts[0],pts[1]):
                            raise ValueError(f'Coincident endpoints: {oid} {key}')
                        state = 'marked'
            record = dict(object_id=oid, role=task['role'], dimension=key, human=state)
            phases = (*PHASES,'edge_audit') if 'edge_audit' in results[oid] else PHASES
            for phase in phases:
                predicted = results[oid][phase]['dimensions'].get(key)
                item = dict(candidate=predicted is not None)
                if state == 'marked' and predicted is not None:
                    line = np.asarray(predicted['line_xy'])
                    item['length_difference_px'] = float(np.linalg.norm(line[1]-line[0]) - np.linalg.norm(pts[1]-pts[0]))
                    item['length_difference_candidate_mm'] = float(predicted['mm_estimate'] - np.linalg.norm((pts[1]-pts[0])/scale))
                    if key == 'underhead_length':
                        item['shoulder_offset_px'], item['tip_offset_px'] = np.linalg.norm(line-pts,axis=1).tolist()
                record[phase] = item
            comparisons.append(record)
    return comparisons


def choose(rows):
    """Eight per class, distinct images; balance available and refused cases."""
    cases, images = [], set()
    for cid in (2, 1, 3):
        pool = sorted((r for r in rows if r['class_id'] == cid and r['scale_ok']),
                      key=lambda r: hashlib.sha256(r['object_id'].encode()).digest())
        signatures = Counter()
        for i in range(8):
            eligible = [r for r in pool if r['image_id'] not in images]
            if not eligible:
                raise ValueError('Insufficient distinct Train images')
            def signature(r):
                return tuple(bool(r[p]['dimensions']) for p in PHASES)
            if cid == 2 and i < len(KNOWN):
                row = next(r for r in eligible if r['object_id'] == KNOWN[i])
            else:
                row = min(eligible, key=lambda r: signatures[signature(r)])
            signatures[signature(row)] += 1
            images.add(row['image_id'])
            cases.append(dict(row, role='review' if i >= 6 else 'development'))
    return sorted(cases, key=lambda r: r['role'])


def compact(result):
    return {k: result[k] for k in ('status', 'reason', 'dimensions')}


def build(dest):
    previous = HISTORY / 'train_full_final'
    history = json.loads((previous/'records.json').read_bytes())
    old = {(r['image_id'], r['object_id']): r for r in history if r['class_id'] in (1,2,3)}
    hashes = json.loads((previous/'source_image_sha256.json').read_bytes())
    selection = json.loads((HISTORY/'selection.json').read_bytes())
    index, objects = catalogue_records('train')
    rows = []
    for name in sorted(index):
        path = SOURCE/'images/train'/name
        if sha256(path) != hashes[name]:
            raise ValueError(f'Train image changed: {name}')
        image, _ = load_display(path)
        for obj in objects[name]:
            cid = int(obj['class_id'])
            if cid not in (1,2,3):
                continue
            oid = obj['object_id']
            ref = old.pop((name, oid))
            crop, box = crop_for_object(image, obj)
            if list(box) != ref['crop_xyxy'] or cid != ref['class_id'] or ref['split'] != 'train':
                raise ValueError(f'Train crop or GT changed: {oid}')
            scale_ok = image.size == (3024,4032)
            if scale_ok:
                bgr = cv2.cvtColor(np.asarray(crop), cv2.COLOR_RGB2BGR)
                raw = measure_object(bgr, cid, selection['scale_px_per_mm'], selection['parameters'])
                candidate = result_global(raw, box[:2])
            else:
                candidate = dict(status='UNMEASURABLE', reason='SCALE_TRANSFER_UNCONFIRMED_RESOLUTION', dimensions={})
            rows.append(dict(image_id=name, object_id=oid, class_id=cid, crop_xyxy=box,
                             scale_ok=scale_ok, before=compact(ref['before']), after=compact(ref['after']),
                             body_profile=compact(candidate)))
    if old:
        raise ValueError('Historical screw inventory has unmatched entries')
    cases = choose(rows)
    (dest/'crops').mkdir()
    tasks = []
    for row in cases:
        name, oid = row['image_id'], row['object_id']
        image, _ = load_display(SOURCE/'images/train'/name)
        crop = image.crop(row['crop_xyxy'])
        filename = f'crops/{len(tasks)+1:02d}.png'
        crop.save(dest/filename)
        tasks.append(dict(image_id=name, object_id=oid, class_id=row['class_id'], role=row['role'],
                          crop_xyxy=row['crop_xyxy'], image_sha256=hashes[name], src=filename,
                          crop_sha256=sha256(dest/filename)))
    manifest = dict(split='train', scale_px_per_mm=selection['scale_px_per_mm'], tasks=tasks)
    manifest['queue_id'] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    save(dest/'queue.json', manifest)
    save(dest/'comparison.json', rows)
    template = Path(__file__).with_name('p06_screw_review.html').read_text(encoding='utf-8')
    (dest/'index.html').write_text(template.replace('/* QUEUE */null', json.dumps(manifest).replace('<', r'\u003c')),
                                   encoding='utf-8')
    counts = {}
    for cid in (1,2,3):
        group = [r for r in rows if r['class_id'] == cid]
        counts[cid] = dict(objects=len(group), **{p: sum(bool(r[p]['dimensions']) for r in group) for p in PHASES})
    save(dest/'summary.json', dict(coverage=counts, train_images=len(index),
                                   code_sha256=sha256(ROOT/'src/p06_geometry.py'),
                                   historical_records_sha256=sha256(previous/'records.json'),
                                   scale_px_per_mm=selection['scale_px_per_mm'], parameters=selection['parameters'],
                                   accuracy='awaiting human lines and specimen-matched calipers'))
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--compare-marks', type=Path, metavar='JSON',
                        help='Print per-item evidence for an exported annotation file, without modifying the queue')
    parser.add_argument('--edge-results',type=Path,help='Optional Train edge-audit records to include with --compare-marks')
    args = parser.parse_args()
    dest = args.output.resolve()
    if args.compare_marks:
        rows = json.loads((dest/'comparison.json').read_bytes())
        if args.edge_results:
            extra = {r['object_id']:r for r in json.loads(args.edge_results.read_bytes())}
            for row in rows:
                other = extra[row['object_id']]
                if (other['split']!='train' or any(row[k]!=other[k] for k in ('image_id','crop_xyxy','class_id'))
                        or row['body_profile']!=other['previous']):
                    raise ValueError('Edge audit does not match the reference cohort')
                row['edge_audit'] = other['edge_audit']
        data = compare_marks(json.loads((dest/'queue.json').read_bytes()),rows,json.loads(args.compare_marks.read_bytes()))
        print(json.dumps(data, ensure_ascii=True, indent=2, allow_nan=False))
        return
    if args.edge_results:
        parser.error('--edge-results requires --compare-marks')
    if dest.exists() or not dest.is_relative_to(ROOT/'output'):
        parser.error('Choose a new directory under output/')
    dest.parent.mkdir(exist_ok=True, parents=True)
    # A Windows TemporaryDirectory uses a private ACL. Renaming its child
    # preserves that ACL and prevents the desktop user's browser reading it.
    stage = dest.parent / ('.p06-screw-review-' + uuid4().hex)
    stage.mkdir()
    try:
        counts = build(stage)
        stage.rename(dest)
    finally:
        if stage.exists() and stage.resolve().parent == dest.parent:
            shutil.rmtree(stage)
    print(counts)


if __name__ == '__main__':
    main()
