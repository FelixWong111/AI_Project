"""Audit and prepare the new collection without training or modifying source data."""
from __future__ import annotations
import argparse
import json
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.data.dataset_v2 import (IMAGE_SUFFIXES, annotate, assign_splits, auxiliary, capture_records,
    digest, duplicates, export, old_inventory, pixel_info, previews, register_heif, relative,
    save_csv, save_json, statistics, verify,write_report,audit_inventory)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=ROOT/'数据集')
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/dataset_preparation_candidate',help='Candidate output only; never overwrite the frozen curated Stage2 tree')
    parser.add_argument('--verify-only',action='store_true')
    parser.add_argument('--refresh-candidate',action='store_true',help='Regenerate derived files only while explicitly NOT_READY; originals rechecked first')
    parser.add_argument('--resolutions',type=Path,default=ROOT/'results/data_manifests/mechanical_components_v2/annotation_resolutions.json')
    args=parser.parse_args();source=args.source.resolve();out=args.output.resolve()
    resolutions=json.loads(args.resolutions.read_text(encoding='utf-8')) if args.resolutions.exists() else {}
    if not out.is_relative_to(ROOT/'data') or out==ROOT/'data' or out.is_relative_to(source):
        parser.error('Output must be a new directory below project/data, outside original sources')
    if args.verify_only:
        print(json.dumps(verify(out,json.loads((out/'audit/source_files.json').read_text(encoding='utf-8'))),indent=2));return
    if args.refresh_candidate:
        status=json.loads((out/'dataset_status.json').read_text(encoding='utf-8'))
        if status.get('frozen') is not False or status.get('status')!='CANDIDATE_NOT_READY':parser.error('Only an unfrozen candidate may be refreshed')
        original=json.loads((out/'audit/source_files.json').read_text(encoding='utf-8'))
        if any(digest(ROOT/r['path'])!=r['sha256'] for r in original):parser.error('Original source changed; prepare a new version')
        register_heif()
        infos={r['path']:r for r in json.loads((out/'audit/image_inventory.json').read_text(encoding='utf-8'))}
        capture,_=capture_records(source);previous=json.loads((out/'manifests/records.json').read_text(encoding='utf-8'))['records']
        by_id={r['image_id']:r for r in previous};records=annotate(source,infos,capture,resolutions)
        for r in records:r['scene_group_id']=by_id[r['image_id']]['scene_group_id']
        save_json(out/'audit/split_strategy.json',assign_splits(records))
        export(records,out);previews(records,out)
        # Remove only byte-verified stale derived placements, never source images or other files.
        for r in records:
            old=ROOT/by_id[r['image_id']]['image_path']
            if old!=ROOT/r['image_path'] and old.exists():
                if not old.resolve().is_relative_to(out/'yolo/images') or digest(old)!=r['sha256']:raise ValueError('Unexpected stale derived file')
                old.unlink();label=out/'yolo/labels'/old.parent.name/(old.stem+'.txt')
                if label.exists():label.unlink()
        stats=json.loads((out/'manifests/dataset_statistics.json').read_text(encoding='utf-8'))
        stats['public']=statistics(records);stats['annotation_issues']={r['image_id']:r['annotation_issues'] for r in records if r['annotation_issues']}
        save_json(out/'manifests/dataset_statistics.json',stats)
        state=json.loads((out/'dataset_status.json').read_text(encoding='utf-8'))
        state['blockers']=['Human visual annotation sign-off scope absent','Review residual near-duplicate candidates before sealing','Confirm no prior unrecorded training/tuning use before sealing Stage 3']
        if stats['annotation_issues']:state['blockers'].append('Unresolved annotation/capture conflicts')
        save_json(out/'dataset_status.json',state)
        verify(out,original);write_report(out)
        print(json.dumps(stats,ensure_ascii=False,indent=2));return
    if out.exists():parser.error('Output exists: use --verify-only or a new version; no overwrite')
    register_heif();out.mkdir(parents=True)
    files=sorted(p for p in source.rglob('*') if p.is_file())
    original=[dict(path=relative(p),size_bytes=p.stat().st_size,sha256=digest(p)) for p in files]
    save_json(out/'audit/source_files.json',original)
    save_json(out/'audit/file_pairing.json',audit_inventory(original))
    images=[p for p in files if p.suffix.lower() in IMAGE_SUFFIXES]
    print(f'Scanning {len(images)} source images',flush=True)
    with ThreadPoolExecutor(max_workers=4) as pool:
        infos={r['path']:r for r in pool.map(pixel_info,images)}
    save_json(out/'audit/image_inventory.json',[{k:v for k,v in r.items() if k!='thumbnail'} for r in infos.values()])
    capture,evidence=capture_records(source);save_json(out/'audit/capture_records.json',evidence)
    records=annotate(source,infos,capture,resolutions)
    duplicate=duplicates(records,infos);save_json(out/'audit/duplicates.json',duplicate)
    print(f'Public images {len(records)}; issues {[(r["image_id"],r["annotation_issues"]) for r in records if r["annotation_issues"]]}',flush=True)
    print('Hashing old dataset roots for cross-version provenance audit (no models)',flush=True)
    old=old_inventory(source,out,infos);save_json(out/'audit/old_data_comparison.json',old)
    for match in old['matches']:
        for r in records:
            if r['source_image']==match['new_path']:
                r['annotation_issues'].append('OLD_DATA_USAGE_CONFLICT');r['annotation_status']='REVIEW';r['label']=''
    allocation=assign_splits(records);save_json(out/'audit/split_strategy.json',allocation)
    export(records,out);previews(records,out)
    print('Decoding HEIC ruler/dimension assets into lossless PNGs',flush=True)
    specialized=auxiliary(source,infos,out)
    stats=dict(status='CANDIDATE_NOT_READY',received=dict(images=len(images),json=sum(p.suffix.lower()=='.json' for p in files),
               workbooks=sum(p.suffix.lower()=='.xlsx' for p in files),bytes=sum(r['size_bytes'] for r in original)),
               public=statistics(records),specialized=specialized,
               annotation_issues={r['image_id']:r['annotation_issues'] for r in records if r['annotation_issues']},
               duplicate_groups=len(duplicate['exact_groups']),near_duplicate_candidates=len(duplicate['near_candidates']),
               old_data_exact_matches=len(old['matches']),seed=42,formal_test='NOT_RUN')
    save_json(out/'manifests/dataset_statistics.json',stats)
    save_json(out/'dataset_status.json',dict(status='CANDIDATE_NOT_READY',frozen=False,formal_training_allowed=False,
              blockers=['Human visual annotation sign-off scope absent','Review residual near-duplicate candidates before sealing',
                        'Confirm no prior unrecorded training/tuning use before sealing Stage 3'],formal_test='NOT_RUN'))
    checks=verify(out,original)
    write_report(out)
    print(json.dumps(stats,ensure_ascii=False,indent=2),flush=True)
    print(f'Output: {relative(out)}; automatic checks={all(checks.values())}; NOT formally frozen',flush=True)


if __name__=='__main__':main()
