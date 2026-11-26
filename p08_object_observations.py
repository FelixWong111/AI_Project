"""Score saved detections, including observations on images marked REVIEW.

This measures emitted observations, not final PASS/FAIL decisions or size accuracy.
"""
import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from src.evaluator import iou, rates

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dataset', required=True, type=Path)
parser.add_argument('--predictions-dir', required=True, type=Path)
parser.add_argument('--output', required=True, type=Path)
parser.add_argument('--details-output',type=Path,help='Additional saved-prediction errors/count differences; does not run inference')
args = parser.parse_args()
data = args.dataset
out = args.predictions_dir

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))

manifest = {r['image_id']: r for r in read(data/'ground_truth_images.csv')}
images = {r['image_id']: r for r in read(out/'images.csv')}
if set(images) != set(manifest):
    raise ValueError('Image lists do not match')
truth = defaultdict(list)
pred = defaultdict(list)
for r in read(data/'ground_truth_objects.csv'):
    if r['ignore'] != '0':
        raise ValueError('Ignored objects require an explicit matching policy')
    truth[r['image_id']].append(r)
for r in read(out/'objects.csv'):
    pred[r['image_id']].append(r)
counts = defaultdict(dict)
for r in read(out/'counts.csv'):
    image, cls = r['image_id'], int(r['class_id'])
    if cls in counts[image]:
        raise ValueError('Duplicate counts')
    counts[image][cls] = int(r['observed_count'])
stats = {c: Counter() for c in range(6)}
mae = {c: [] for c in range(6)}
vector_ok = []
status_counts = Counter()
errors=[];count_differences=[]
for image in sorted(manifest):
    if manifest[image]['annotation_status'].upper() == 'REVIEW':
        continue
    status = images[image]['status']
    if status not in ('PASS', 'FAIL', 'REVIEW'):
        raise ValueError(f'Unavailable prediction {image}: {status}')
    if set(counts[image]) != set(range(6)):
        raise ValueError(f'Missing observed count: {image}')
    status_counts[status] += 1
    gt_counts = Counter(int(r['class_id']) for r in truth[image])
    pr_counts = Counter(int(r['class_id']) for r in pred[image])
    if any(counts[image][c] != pr_counts[c] for c in range(6)):
        raise ValueError(f'Objects and counts disagree: {image}')
    vector_ok.append(all(gt_counts[c] == pr_counts[c] for c in range(1,6)))
    for c in range(6):count_differences.append(dict(image_id=image,class_id=c,truth_count=gt_counts[c],observed_count=pr_counts[c],difference=pr_counts[c]-gt_counts[c],status=status))
    for c in range(6):
        gt = [r for r in truth[image] if int(r['class_id']) == c]
        ds = sorted((r for r in pred[image] if int(r['class_id']) == c),
                    key=lambda r: (-float(r['score']), int(r['detection_id'])))
        matched = set()
        stats[c]['support'] += len(gt)
        for det in ds:
            b = tuple(float(det[k]) for k in ('xmin','ymin','xmax','ymax'))
            candidates = sorted(((iou(b, tuple(float(r[k]) for k in ('xmin','ymin','xmax','ymax'))), n)
                                 for n,r in enumerate(gt) if n not in matched),
                                key=lambda item: (-item[0], str(gt[item[1]]['object_id'])))
            if candidates and candidates[0][0] >= .5:
                matched.add(candidates[0][1]); stats[c]['tp'] += 1
            else:
                stats[c]['fp'] += 1
                overlaps=[(iou(b,tuple(float(g[k]) for k in ('xmin','ymin','xmax','ymax'))),g) for g in truth[image]]
                best=max(overlaps,key=lambda x:x[0]) if overlaps else (0,None)
                errors.append(dict(image_id=image,event='FP',class_id=c,detection_id=det['detection_id'],score=float(det['score']),
                    overlap_true_class=int(best[1]['class_id']) if best[0]>=.5 else None,overlap_iou=best[0]))
        stats[c]['fn'] += len(gt) - len(matched)
        errors.extend(dict(image_id=image,event='FN',class_id=c,object_id=g['object_id']) for n,g in enumerate(gt) if n not in matched)
        mae[c].append(abs(gt_counts[c]-pr_counts[c]))
report = {'scope': 'Saved predicted boxes/counts, including REVIEW; does not treat REVIEW as PASS/FAIL or verify physical sizes',
          'image_count': sum(status_counts.values()), 'prediction_status_counts': dict(status_counts),
          'five_class_vector_exact': sum(vector_ok)/len(vector_ok), 'classes': {}}
for c, s in stats.items():
    report['classes'][str(c)] = {k:s[k] for k in ('support','tp','fp','fn')}
    report['classes'][str(c)].update(rates(s['tp'],s['fp'],s['fn']))
    report['classes'][str(c)]['count_mae'] = sum(mae[c])/len(mae[c])
if args.output.exists():
    if not args.details_output or json.loads(args.output.read_text(encoding='utf8'))!=report:
        raise ValueError('Output already exists; use a new path')
else:args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
if args.details_output:
    if args.details_output.exists():raise ValueError('Details already exist; preserve previous evidence')
    args.details_output.write_text(json.dumps(dict(scope='Postprocess saved predictions only; existing summary unchanged',
        errors=errors,count_differences=count_differences,
        permitted_rejected_as_id0=[e for e in errors if e['event']=='FP' and e['class_id']==0 and e['overlap_true_class'] in (1,2,3,4,5)],
        non_catalogue_misses=[e for e in errors if e['event']=='FN' and e['class_id']==0]),ensure_ascii=False,indent=2)+'\n',encoding='utf8')
print(json.dumps(report, ensure_ascii=False, indent=2))
