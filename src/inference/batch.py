"""Contract batch CLI; uses exactly the same registered backends as app.py."""
from __future__ import annotations
import argparse
import csv
import json
import platform
import sys
import time
from pathlib import Path

from src.adapters.baseline2 import IMAGE_FIELDS
from src.work_orders import load_cases
from src.models.baseline1.features import load_image_bgr
from src.p07_bridge import MODELS, UnavailableBackend, load_backend, validate_counts, validate_result
from src.p07_output import write_diagnostics
from src.model_registry import portable




def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input_dir',required=True,type=Path)
    parser.add_argument('--model',required=True,choices=MODELS)
    parser.add_argument('--output_dir',required=True,type=Path)
    parser.add_argument('--counts_file',required=True,type=Path)
    parser.add_argument('--config',type=Path,default=Path(__file__).resolve().parents[2]/'config/p07.json')
    parser.add_argument('--bundle',type=str,help='Override model bundle path')
    parser.add_argument('--record-timing',action='store_true',help='Offline adapter processing time; excludes decoding/loading and is not live latency')
    args=parser.parse_args(argv)
    try:
        cases=load_cases(args.counts_file)
    except (ValueError, OSError) as exc:
        parser.error('Invalid counts_file: ' + str(exc))
    if args.output_dir.exists():raise FileExistsError('Use a new output directory; preserve previous runs')
    config=json.loads(args.config.read_text(encoding='utf-8-sig'))
    if args.bundle:config.setdefault('model_config',{})[args.model+'_bundle']=args.bundle
    load_error=None
    try:backend=load_backend('real',args.model,config)
    except Exception as exc:
        load_error=str(exc);backend=UnavailableBackend(args.model,exc)
    rows=[];objects=[];counts=[];diagnostics=[];timing=[]
    for case in cases:
        started=None
        try:
            if load_error:raise RuntimeError(load_error)
            frame=load_image_bgr(str(args.input_dir/case['image_id']))
            if args.record_timing:
                backend.synchronize();started=time.perf_counter()
            result=validate_result(backend.infer(frame,case['expected_counts']),args.model,frame.shape)
            if args.record_timing:backend.synchronize()
        except Exception as exc:
            result=UnavailableBackend(args.model,exc).infer()
            print(f'{case["image_id"]}: {exc}',file=sys.stderr)
        if args.record_timing:
            timing.append(dict(image_id=case['image_id'],status=result['status'],processing_ms=None if started is None else (time.perf_counter()-started)*1000))
        row=dict(result,image_id=case['image_id'],model_id=args.model)
        rows.append({k:row.get(k) for k in IMAGE_FIELDS})
        if args.model == 'baseline3':
            diagnostics.append((case['image_id'], case['expected_counts'], result))
        if args.model in ('baseline3','final') and result.get('counts') is not None:
            fields = ('detection_id','class_id','xmin','ymin','xmax','ymax','score')
            objects.extend(dict(image_id=case['image_id'],model_id=args.model,
                                **{k:o[k] for k in fields}) for o in result['objects'])
            for c in '012345':
                expected=0 if c=='0' else case['expected_counts'][c]
                observed=result['counts'][c]
                counts.append(dict(image_id=case['image_id'],class_id=int(c),observed_count=observed,
                                   expected_count=expected,difference=observed-expected))
    args.output_dir.mkdir(parents=True)
    def write(name,fields,values):
        with (args.output_dir/name).open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(values)
    write('images.csv',IMAGE_FIELDS,rows)
    if args.record_timing:write('timing.csv',['image_id','status','processing_ms'],timing)
    if args.model in ('baseline3','final') and not load_error:
        write('objects.csv',['image_id','model_id','detection_id','class_id','xmin','ymin','xmax','ymax','score'],objects)
        write('counts.csv',['image_id','class_id','observed_count','expected_count','difference'],counts)
    metadata=portable(dict(backend.metadata))
    metadata.setdefault('hardware',dict(platform=platform.platform(),python=sys.version))
    metadata.update(per_image_expected_counts={c['image_id']:c['expected_counts'] for c in cases},
                    load_error=load_error,evidence_mode='batch_inference')
    if args.record_timing:metadata['offline_timing']='perf_counter: adapter preprocess/features/classifier/postprocess/result validation + GPU completion sync; excludes image decoding, weight load and export; not live latency'
    (args.output_dir/'run_config.json').write_text(json.dumps(metadata,indent=2,ensure_ascii=False),encoding='utf-8')
    if args.model == 'baseline3':
        write_diagnostics(args.output_dir, diagnostics, metadata)
    print(f'{args.model}: {len(rows)} rows -> {args.output_dir}')
    return 1 if any(r['status']=='ERROR' for r in rows) else 0
