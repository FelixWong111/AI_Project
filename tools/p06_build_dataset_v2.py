"""Build v2 only from the explicitly approved P06 development manifest; no training."""
import argparse
import csv
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.p06_labelme import require
from tools.p06_preview_annotations import digest
from tools.p06_build_dataset import create_plan, _validate_structure, build_dataset, summarize

SOURCE=ROOT/'data/p03/development_supplement_v1/development_v1'
CANDIDATE=ROOT/'output/p06/yolo_dataset_v2_candidate/split_manifest.csv'
OUTPUT=ROOT/'output/p06/yolo_dataset_v2'
TOTALS=[159,305,306,310,308,305]


def create_v2_plan(source,manifest,old_plan,confirmed_empty):
    source,manifest=Path(source).resolve(),Path(manifest).resolve()
    with manifest.open(encoding='utf-8-sig',newline='') as stream:
        rows=list(csv.DictReader(stream))
    required={'image_id','scene_id','split','provenance','scene_authority','scene_basis','reason',
              'sha256','pixel_sha256',*(f'count_{c}' for c in range(6))}
    require(rows and all(required<=set(r) for r in rows),'INVALID_MANIFEST','Incomplete approved manifest.')
    require(len({r['image_id'] for r in rows})==len(rows),'DUPLICATE_IMAGE','Manifest repeats image IDs.')
    for row in rows:
        require(row['image_id']==Path(row['image_id']).name and '/' not in row['image_id'] and '\\' not in row['image_id'],
                'UNSAFE_INPUT','Manifest image IDs must be local basenames.')
        require(row['split'] in {'train','val'} and bool(row['scene_id']),'INVALID_SPLIT','Explicit train/val and scene IDs required.')
        require(row['provenance'] in {'old51','new275'},'INVALID_PROVENANCE','Unexpected provenance.')
    val={r['image_id'] for r in rows if r['split']=='val'}
    plan=create_plan(source,val,confirmed_empty,version=2)
    require({r['image_id'] for r in plan['records']}=={r['image_id'] for r in rows},'MANIFEST_IMAGES','Manifest/source sets differ.')
    approved={r['image_id']:r for r in rows}
    prior={r['image_id']:dict(split=r['split'],scene_id=r['scene_id']) for r in old_plan['records']}
    for record in plan['records']:
        row=approved[record['image_id']]
        record.update({k:row[k] for k in ('scene_id','split','scene_authority','scene_basis','reason','provenance')})
    plan.update(version=2,approved_manifest=rows,approved_manifest_sha256=digest(manifest),
                approved_manifest_source=str(manifest),old_assignments=prior,
                method='Approved manifest; seeded annotation-count stratification; preserve original partitions')
    plan['summary']=summarize(plan['records'])
    validate_v2_plan(plan)
    return plan


def validate_v2_plan(plan):
    require(plan.get('version')==2,'INVALID_VERSION','Use the explicit v2 workflow.')
    require(digest(Path(plan['approved_manifest_source']))==plan['approved_manifest_sha256'],
            'MANIFEST_CHANGED','Approved manifest file changed after planning.')
    if 'package_snapshot' in plan:
        package=Path(plan['source']).parent
        require({p.relative_to(package).as_posix():digest(p) for p in package.rglob('*') if p.is_file()}==plan['package_snapshot'],
                'SOURCE_CHANGED','Audited package changed after planning.')
    _validate_structure(plan,allow_id0_val=True)
    approved=plan['approved_manifest']
    require(len({r['image_id'] for r in approved})==len(approved),'DUPLICATE_IMAGE','Repeated approved image ID.')
    by_id={r['image_id']:r for r in approved}
    require(set(by_id)=={r['image_id'] for r in plan['records']},'MANIFEST_IMAGES','Approved list changed.')
    for row in plan['records']:
        a=by_id[row['image_id']]
        require(all(row[k]==a[k] for k in ('split','scene_id','provenance','scene_authority','scene_basis','reason')),
                'MANIFEST_CHANGED','Record differs from approved manifest.')
        require(row['counts']==[int(a[f'count_{c}']) for c in range(6)],'MANIFEST_COUNTS','Approved class counts differ.')
        require(row['display_pixel_sha256']==a['pixel_sha256'] and plan['source_sha256'][row['image_id']]==a['sha256'],
                'MANIFEST_HASH','Approved pixel/file SHA differs.')
        old=plan['old_assignments'].get(row['image_id'])
        require((old is not None)==(row['provenance']=='old51'),'INVALID_PROVENANCE','Old/new membership differs.')
        if old:
            require(row['split']==old['split'] and row['scene_id']==old['scene_id'],
                    'OLD_PARTITION_CHANGED','Original train/validation assignment or scene changed.')
    require(set(plan['old_assignments'])<={r['image_id'] for r in plan['records']},'MISSING_OLD_IMAGE','Original image missing.')


def validate_project_v2(plan):
    validate_v2_plan(plan)
    s=plan['summary']
    require((s['train']['images'],s['val']['images'])==(261,65),'PROJECT_SPLIT','Expected 261/65.')
    require([sum(r['counts'][c] for r in plan['records']) for c in range(6)]==TOTALS,'PROJECT_COUNTS','Expected 1693 instances.')
    require(s['train']['classes']==[127,248,244,249,248,245] and s['val']['classes']==[32,57,62,61,60,60],
            'PROJECT_COUNTS','Split class totals differ from approved candidate.')
    require(set(s['train']['empty'])=={'fail124.jpg'} and set(s['val']['empty'])=={'fail13.jpg','fail123.jpg'},
            'EMPTY_PLACEMENT','Empty backgrounds differ from approval.')


def report_v2(plan,checks):
    s=plan['summary']
    return ['# P06 YOLO dataset v2','',
        'Derived ground-truth dataset, not model predictions. No training or held-out test reads.',
        'Approved manifest preserved. Independent arrangements confirmed by user; scene IDs remain P06 provisional pending P03/P08 review.',
        f'Train/val: {s["train"]["images"]}/{s["val"]["images"]}; instances {checks["instances"]}; totals {checks["classes"]}.',
        f'Train classes: {s["train"]["classes"]}; validation classes: {s["val"]["classes"]}.',
        'Old 41 train / 10 validation membership is unchanged; new data adds 220 train / 55 validation.',
        'Validation now includes 32 ID 0 instances in 20 images. This enables validation of the sampled non-catalogue examples, not universal rejection claims.',
        'Empty backgrounds: train fail124; validation fail13/fail123. Empty is not ID 0.',
        'PNG pixels equal ImageOps.exif_transpose source RGB at full resolution. JSON coordinates are applied directly, never rotated/scaled/clipped.',
        f'Maximum serialized TXT roundtrip error: {checks["max_roundtrip_error_pixels"]:.12g} pixels; limit 1e-6.',
        'Source SHA values and geometry are checked. See source_plan.json, split_manifest.csv and validation/checks.json.',
        f'Official Ultralytics loading: {checks["ultralytics"]["status"]}. Temporary label caches removed before publication.',
        'dataset.yaml uses path: .; run with cwd at dataset root and absolute model/output paths. No test/download entry.',
        'Old 10 validation images have been repeatedly used in development; report them separately from the new 55. Neither group is a held-out final test.',
        'PASS/FAIL filename prefixes are metadata; no expected-count image-level evaluation is implied.',
        'Physical subtype coverage, calibration and size tolerance are not established by this dataset.',
        'Baseline prediction and all future training require independent output directories. Original v1 is preserved.','']


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=OUTPUT)
    args=parser.parse_args()
    require(args.output.resolve()==OUTPUT,'UNSAFE_OUTPUT','This CLI publishes only output/p06/yolo_dataset_v2.')
    if OUTPUT.exists(): raise FileExistsError(f'Refusing existing dataset: {OUTPUT}')
    audit=ROOT/'output/p06/supplement_audit_v1/source_sha256.json'
    hashes=json.loads(audit.read_text(encoding='utf-8'))
    package=SOURCE.parent
    def check_package():
        require({p.relative_to(package).as_posix():digest(p) for p in package.rglob('*') if p.is_file()}==hashes,
                'SOURCE_CHANGED','Package differs from audited source.')
    check_package()
    manifest_hash=digest(CANDIDATE)
    old=json.loads((ROOT/'output/p06/yolo_dataset/source_plan.json').read_text(encoding='utf-8'))
    plan=create_v2_plan(SOURCE,CANDIDATE,old,{'fail13.jpg','fail123.jpg','fail124.jpg'})
    plan['package_snapshot']=hashes
    validate_project_v2(plan)
    check_package()
    require(digest(CANDIDATE)==manifest_hash,'MANIFEST_CHANGED','Approved manifest changed during planning.')
    print(json.dumps(plan['summary']),flush=True)
    result=build_dataset(plan,OUTPUT,version=2)
    print(json.dumps(result,indent=2),flush=True)


if __name__=='__main__': main()
