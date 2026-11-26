"""Independent evaluation of one inspection run; Python standard library only."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

CLASSES = range(6)
NAMES = ('non_catalogue', 'long_screw', 'thick_screw', 'thin_screw', 'hex_nut', 'flat_washer')


def read_csv(path, required):
    with Path(path).open(encoding='utf-8-sig', newline='') as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError(f'{path}: missing or duplicate columns')
        missing = set(required) - set(reader.fieldnames)
        if missing:
            raise ValueError(f'{path}: missing columns {sorted(missing)}')
        rows = list(reader)
        if any(None in row for row in rows):
            raise ValueError(f'{path}: row has more fields than header')
        return rows


def integer(value, context, minimum=0):
    try:
        s = str(value).strip()
        if not s.isdigit():
            raise ValueError()
        n = int(s)
        if n < minimum:
            raise ValueError()
        return n
    except (TypeError, ValueError):
        raise ValueError(f'{context}: expected integer >= {minimum}, got {value!r}') from None


def number(value, context):
    try:
        n = float(value)
        if not math.isfinite(n):
            raise ValueError()
        return n
    except (TypeError, ValueError):
        raise ValueError(f'{context}: expected finite number, got {value!r}') from None


def box(row, width, height, context):
    b = tuple(number(row[k], f'{context} {k}') for k in ('xmin', 'ymin', 'xmax', 'ymax'))
    if not (0 <= b[0] < b[2] <= width and 0 <= b[1] < b[3] <= height):
        raise ValueError(f'{context}: invalid or out-of-bounds box {b}')
    return b


def iou(a, b):
    inter = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(0, min(a[3], b[3]) - max(a[1], b[1]))
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter
    return inter / union if union else 0.0


def ratio(a, b):
    return a / b if b else None


def rates(tp, fp, fn):
    p, r = ratio(tp, tp + fp), ratio(tp, tp + fn)
    return {'precision': p, 'recall': r, 'f1': ratio(2 * tp, 2 * tp + fp + fn)}


def unique(rows, field, source):
    result = {}
    for row in rows:
        key = row[field].strip()
        if not key or key in result:
            raise ValueError(f'{source}: empty or duplicate {field}: {key!r}')
        result[key] = row
    return result


def image_level_details(rows):
    """Add explicit denominators; preserve the original scored-subset metrics."""
    eligible=[r for r in rows if r['ground_truth_label'] in ('PASS','FAIL')]
    complete=[r for r in eligible if r['prediction_status'] in ('PASS','FAIL')]
    positive=sum(r['ground_truth_label']=='FAIL' for r in eligible)
    negative=len(eligible)-positive
    tp=sum(r['ground_truth_label']=='FAIL' and r['predicted_label']=='FAIL' for r in complete)
    fp=sum(r['ground_truth_label']=='PASS' and r['predicted_label']=='FAIL' for r in complete)
    fn=sum(r['ground_truth_label']=='FAIL' and r['predicted_label']=='PASS' for r in complete)
    tn=sum(r['ground_truth_label']=='PASS' and r['predicted_label']=='PASS' for r in complete)
    fail=rates(tp,fp,fn);passed=rates(tn,fn,fp)
    recall_fail,recall_pass=ratio(tp,tp+fn),ratio(tn,tn+fp)
    balanced=(recall_fail+recall_pass)/2 if recall_fail is not None and recall_pass is not None else None
    statuses=Counter(r['prediction_status'] for r in rows)
    return dict(input_cases=len(rows),ground_truth_eligible=len(eligible),completed_eligible=len(complete),
                correct=tp+tn,wrong_on_completed=fp+fn,unresolved_eligible=len(eligible)-len(complete),
                all_input_correct_fraction=ratio(tp+tn,len(rows)),all_eligible_correct_fraction=ratio(tp+tn,len(eligible)),
                completed_accuracy=ratio(tp+tn,len(complete)),completed_balanced_accuracy=balanced,
                prediction_coverage=ratio(statuses['PASS']+statuses['FAIL'],len(rows)),
                eligible_coverage=ratio(len(complete),len(eligible)),status_counts=dict(statuses),
                confusion_matrix=dict(positive_class='FAIL',tp=tp,fp=fp,fn=fn,tn=tn,rows_truth_PASS_FAIL=[[tn,fp],[fn,tp]]),
                fail=dict(**fail,support_all=positive,support_scored=tp+fn,recall_all_eligible=ratio(tp,positive)),
                passed=dict(**passed,support_all=negative,support_scored=tn+fp,recall_all_eligible=ratio(tn,negative)),
                false_pass=dict(count=fn,rate=ratio(fn,positive),denominator=positive,denominator_definition='All eligible true FAIL cases, including unresolved predictions'),
                false_fail=dict(count=fp,rate=ratio(fp,negative),denominator=negative,denominator_definition='All eligible true PASS cases, including unresolved predictions'),
                note='Completed accuracy/precision/recall exclude unresolved predictions as in the existing evaluator; all-input correct fraction and coverage retain them explicitly. No REVIEW/ERROR remapping.')


def evaluate(images_path, objects_path, expected_path, predictions_dir, output_dir):
    pred_dir, out = Path(predictions_dir), Path(output_dir)
    if out.exists():
        raise ValueError(f'output directory already exists: {out}')
    manifest = unique(read_csv(images_path, ['image_id','scene_id','split','width','height','annotation_status']), 'image_id', images_path)
    with Path(expected_path).open(encoding='utf-8-sig') as f:
        cases = json.load(f)
    if not isinstance(cases, list) or not cases:
        raise ValueError('expected counts must be a nonempty list')
    expected = {}
    for case in cases:
        if not isinstance(case, dict) or set(case) != {'image_id','expected_counts'}:
            raise ValueError('each case must contain only image_id and expected_counts')
        image_id, counts = case['image_id'], case['expected_counts']
        if not isinstance(image_id, str) or not image_id.strip() or image_id in expected:
            raise ValueError(f'empty or duplicate image_id: {image_id!r}')
        if not isinstance(counts, dict) or set(counts) != set('12345'):
            raise ValueError(f'{image_id}: expected_counts must contain keys 1 through 5')
        if any(type(v) is not int or v < 0 for v in counts.values()):
            raise ValueError(f'{image_id}: expected counts must be nonnegative integers')
        expected[image_id] = {int(k): v for k, v in counts.items()}
    if set(expected) - set(manifest):
        raise ValueError(f'images missing from manifest: {sorted(set(expected)-set(manifest))}')
    predictions = unique(read_csv(pred_dir/'images.csv', ['image_id','model_id','status','predicted_label','prediction_score','score_type','non_catalogue_count']), 'image_id', 'images.csv')
    if set(predictions) != set(expected):
        raise ValueError(f'images.csv image IDs do not match expected_counts: missing={sorted(set(expected)-set(predictions))}, extra={sorted(set(predictions)-set(expected))}')
    model_ids = {r['model_id'] for r in predictions.values()}
    if len(model_ids) != 1 or next(iter(model_ids)) not in {'baseline1','baseline2','baseline3','final'}:
        raise ValueError('one valid model_id is required per run')
    model = next(iter(model_ids))
    capable = model in {'baseline3','final'}
    gt = defaultdict(list)
    keys = set()
    for row in read_csv(objects_path, ['image_id','object_id','class_id','xmin','ymin','xmax','ymax','ignore']):
        image_id = row['image_id']
        if image_id not in manifest:
            raise ValueError(f'ground-truth object has unknown image_id {image_id}')
        key = (image_id, row['object_id'])
        if not row['object_id'].strip() or key in keys:
            raise ValueError(f'duplicate or empty object_id: {key}')
        keys.add(key)
        cls = integer(row['class_id'], 'ground truth class_id')
        ignore = integer(row['ignore'], 'ground truth ignore')
        if cls not in CLASSES or ignore not in (0, 1):
            raise ValueError(f'{key}: invalid class or ignore flag')
        m = manifest[image_id]
        gt[image_id].append({'id': row['object_id'], 'class': cls, 'ignore': ignore,
                             'box': box(row, integer(m['width'], 'width', 1), integer(m['height'], 'height', 1), str(key))})
    detected = defaultdict(list)
    reported_counts = defaultdict(dict)
    if capable:
        detkeys = set()
        for row in read_csv(pred_dir/'objects.csv', ['image_id','model_id','detection_id','class_id','xmin','ymin','xmax','ymax','score']):
            image_id = row['image_id']
            if image_id not in expected or row['model_id'] != model:
                raise ValueError(f'objects.csv: unknown image or model: {image_id}')
            did = integer(row['detection_id'], 'detection_id')
            if (image_id, did) in detkeys:
                raise ValueError(f'duplicate detection_id {image_id}/{did}')
            detkeys.add((image_id, did))
            cls = integer(row['class_id'], 'prediction class_id')
            if cls not in CLASSES:
                raise ValueError('predicted class_id must be 0 through 5')
            m = manifest[image_id]
            detected[image_id].append({'id': did, 'class': cls, 'score': number(row['score'], 'detection score'),
                                       'box': box(row, integer(m['width'], 'width', 1), integer(m['height'], 'height', 1), image_id)})
        for row in read_csv(pred_dir/'counts.csv', ['image_id','class_id','observed_count','expected_count','difference']):
            image_id = row['image_id']
            if image_id not in expected:
                raise ValueError(f'counts.csv: unknown image_id {image_id}')
            c = integer(row['class_id'], 'counts class_id')
            if c not in CLASSES or c in reported_counts[image_id]:
                raise ValueError(f'{image_id}: invalid or duplicate count class {c}')
            observed = integer(row['observed_count'], 'observed_count')
            wanted = integer(row['expected_count'], 'expected_count')
            try:
                difference = int(row['difference'])
            except ValueError:
                raise ValueError(f'{image_id}: invalid difference') from None
            if wanted != (0 if c == 0 else expected[image_id][c]) or difference != observed - wanted:
                raise ValueError(f'{image_id}: counts.csv expected count or difference disagrees')
            reported_counts[image_id][c] = observed
    cm = Counter()
    status_counts = Counter()
    class_stats = {c: Counter() for c in CLASSES}
    error_counts = Counter()
    count_errors = {c: [] for c in CLASSES}
    details = []
    valid_vectors = []
    counted_scenes = set()
    for image_id in sorted(expected):
        m, pred, exp = manifest[image_id], predictions[image_id], expected[image_id]
        if not m['scene_id'].strip() or not m['split'].strip():
            raise ValueError(f'{image_id}: scene_id and split required')
        integer(m['width'], 'width', 1); integer(m['height'], 'height', 1)
        annotation = m['annotation_status'].strip().upper()
        if annotation not in {'COMPLETE','VERIFIED','OK','REVIEWED','REVIEW'}:
            raise ValueError(f'{image_id}: annotation_status must be COMPLETE, VERIFIED, OK or REVIEW; map P03 values explicitly')
        status, label = pred['status'].strip().upper(), pred['predicted_label'].strip().upper()
        if status not in {'PASS','FAIL','REVIEW','UNSUPPORTED_COUNTS','ERROR'}:
            raise ValueError(f'{image_id}: invalid prediction status {status}')
        if (status in {'PASS','FAIL'} and label != status) or (status not in {'PASS','FAIL'} and label):
            raise ValueError(f'{image_id}: status and predicted_label disagree')
        status_counts[status] += 1
        truth = [o for o in gt[image_id] if not o['ignore']]
        ignored = [o for o in gt[image_id] if o['ignore']]
        tc = Counter(o['class'] for o in truth)
        # Validate conventional manifest count columns if present; never use them as inference input.
        for c, name in enumerate(NAMES):
            keys_to_check = ['non_catalogue_count'] if c == 0 else [name + '_count', f'count_{c}']
            for key in keys_to_check:
                if key in m and m[key].strip() and integer(m[key], f'{image_id} {key}') != tc[c]:
                    raise ValueError(f'{image_id}: {key} does not match non-ignored objects')
        truth_label = 'FAIL' if tc[0] or any(tc[c] != exp[c] for c in range(1,6)) else 'PASS'
        for label_field in ('ground_truth_label', 'label', 'reference_label_all_one'):
            stored = m.get(label_field, '').strip().upper()
            # The P03 column reference_label_all_one applies only to that fixed work order.
            applicable = label_field != 'reference_label_all_one' or all(exp[c] == 1 for c in range(1, 6))
            if applicable and stored and annotation != 'REVIEW' and not ignored and stored != truth_label:
                raise ValueError(f'{image_id}: stored {label_field} disagrees with derived label')
        evaluable = annotation != 'REVIEW' and not ignored
        reason = 'unjudgeable_annotation' if annotation == 'REVIEW' else ('ignored_objects' if ignored else '')
        if evaluable and status in {'PASS','FAIL'}:
            counted_scenes.add(m['scene_id'])
            cm[('T' if truth_label == 'FAIL' else 'F') + ('P' if label == 'FAIL' else 'N')] += 1
            if truth_label == 'FAIL' and label == 'PASS' and tc[0]:
                error_counts['false_pass_with_non_catalogue'] += 1
        d = {'image_id': image_id, 'scene_id': m['scene_id'], 'split': m['split'], 'annotation_status': annotation,
             'prediction_status': status, 'ground_truth_label': truth_label if evaluable else '',
             'predicted_label': label, 'excluded_from_image_and_count_metrics': reason or ('' if status in {'PASS','FAIL'} else status)}
        if capable and annotation != 'REVIEW' and status in {'PASS','FAIL'}:
            ds = detected[image_id]
            pc = Counter(o['class'] for o in ds)
            if set(reported_counts[image_id]) != set(CLASSES) or any(reported_counts[image_id][c] != pc[c] for c in CLASSES):
                raise ValueError(f'{image_id}: counts.csv must match objects.csv for classes 0 through 5')
            if pred['non_catalogue_count'].strip() == '' or integer(pred['non_catalogue_count'], image_id+' non_catalogue_count') != pc[0]:
                raise ValueError(f'{image_id}: predicted non_catalogue_count disagrees with objects')
            rule_label = 'FAIL' if pc[0] or any(pc[c] != exp[c] for c in range(1,6)) else 'PASS'
            # A size-verification rule can reject an apparently correct count.
            # PASS is still invalid if the observed counts or class-0 rule fail.
            if label == 'PASS' and rule_label != 'PASS':
                raise ValueError(f'{image_id}: predicted PASS disagrees with detected counts')
            # Matched eligible targets take priority; ignored same-class regions neutralize remaining predictions.
            for c in CLASSES:
                targets = [o for o in truth if o['class'] == c]
                ignored_c = [o for o in ignored if o['class'] == c]
                matched = set()
                class_stats[c]['support'] += len(targets)
                for det in sorted((o for o in ds if o['class'] == c), key=lambda x: (-x['score'], x['id'])):
                    candidates = [(iou(det['box'], obj['box']), idx) for idx, obj in enumerate(targets) if idx not in matched]
                    candidates.sort(key=lambda x: (-x[0], str(targets[x[1]]['id'])))
                    if candidates and candidates[0][0] >= .5:
                        matched.add(candidates[0][1]); class_stats[c]['tp'] += 1
                    elif any(iou(det['box'], obj['box']) >= .5 for obj in ignored_c):
                        class_stats[c]['ignored_predictions'] += 1
                    else:
                        class_stats[c]['fp'] += 1
                class_stats[c]['fn'] += len(targets) - len(matched)
                if evaluable:
                    count_errors[c].append(abs(pc[c] - tc[c]))
                    d[f'actual_{c}'] = tc[c]; d[f'predicted_{c}'] = pc[c]
                    if c and (pc[c] - exp[c]) != (tc[c] - exp[c]):
                        error_counts[f'count_difference_error_class_{c}'] += 1
            if evaluable:
                valid_vectors.append(all(pc[c] == tc[c] for c in range(1,6)))
            # This is a spatial diagnostic, not a unique assignment or a proven causal classification.
            error_counts['class0_predictions_overlapping_valid_target'] += sum(
                any(o['class'] > 0 and iou(det['box'], o['box']) >= .5 for o in truth)
                for det in ds if det['class'] == 0)
        details.append(d)
    per_class = {}
    if capable:
        for c in CLASSES:
            s = class_stats[c]
            per_class[str(c)] = {'name': NAMES[c], **{k: s[k] for k in ('support','tp','fp','fn','ignored_predictions')},
                                 **rates(s['tp'],s['fp'],s['fn']), 'count_mae': ratio(sum(count_errors[c]),len(count_errors[c])),
                                 'count_image_support': len(count_errors[c])}
    summary = {'model_id': model, 'capability': 'object_level' if capable else 'image_level',
               'input_cases': len(expected), 'input_scenes': len({manifest[x]['scene_id'] for x in expected}),
               'scored_image_cases': sum(cm.values()), 'scored_scenes': len(counted_scenes),
               'prediction_status_counts': dict(status_counts),
               'image_level': {'positive_class': 'FAIL', 'tp': cm['TP'], 'fp': cm['FP'], 'fn': cm['TN'], 'tn': cm['FN'],
                               'fail_support': cm['TP']+cm['TN'], 'pass_support': cm['FP']+cm['FN'],
                               **rates(cm['TP'],cm['FP'],cm['TN']), 'false_pass': cm['TN'], 'false_fail': cm['FP']},
               'object_classes': per_class, 'count_vector_exact_rate': ratio(sum(valid_vectors),len(valid_vectors)) if capable else None,
               'count_vector_support': len(valid_vectors), 'critical_errors': dict(error_counts),
               'matching': 'Same class, score descending, detection_id numeric ascending; highest IoU >= 0.50, object_id lexical tie-break. Unmatched predictions overlapping same-class ignored ground truth at IoU >= 0.50 are neutral.',
               'exclusions': 'REVIEW ground truth and images containing ignored objects are excluded from image and count metrics; REVIEW, ERROR, UNSUPPORTED_COUNTS predictions excluded. Object metrics retain judgeable non-ignored objects in images with ignored objects.',
               'inputs': {'ground_truth_images': str(images_path), 'ground_truth_objects': str(objects_path), 'expected_counts': str(expected_path), 'predictions_dir': str(predictions_dir)},
               'note': 'Latency and independent test validity require separate evidence; zero-denominator metrics are null.'}
    summary['image_level_details']=image_level_details(details)
    out.mkdir(parents=True)
    (out/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    columns = ['image_id','scene_id','split','annotation_status','prediction_status','ground_truth_label','predicted_label','excluded_from_image_and_count_metrics'] + [f'{p}_{c}' for c in CLASSES for p in ('actual','predicted')]
    with (out/'per_image.csv').open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=columns); writer.writeheader(); writer.writerows(details)
    return summary


def main():
    parser = argparse.ArgumentParser(description='Evaluate one run against separate ground truth')
    parser.add_argument('--ground-truth-images', required=True)
    parser.add_argument('--ground-truth-objects', required=True)
    parser.add_argument('--expected-counts', required=True)
    parser.add_argument('--predictions-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    try:
        result = evaluate(args.ground_truth_images, args.ground_truth_objects, args.expected_counts, args.predictions_dir, args.output_dir)
    except (ValueError, OSError, json.JSONDecodeError, KeyError) as exc:
        parser.exit(2, f'evaluation failed: {exc}\n')
    print(f"Evaluated {result['scored_image_cases']} image cases for {result['model_id']}; results in {args.output_dir}")


if __name__ == '__main__':
    main()
