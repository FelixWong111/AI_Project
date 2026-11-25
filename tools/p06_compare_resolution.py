"""Compare fixed best.pt at 640/960 on accepted Validation only; never train."""
import argparse
from contextlib import chdir, redirect_stdout, redirect_stderr
import gc
import json
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.p06_train_baseline import (inspect_environment,configure_runtime,validate_best,predict_val,
                                     require,save_json,Tee,snapshot)
from tools.p06_accepted_data import accepted_plan,check_sources,prepare_runtime
from tools.p06_preview_annotations import digest
from src.data.p06_labelme import CLASSES

SOURCE=ROOT/'data/p03/development_v2'
ACCEPTANCE=ROOT/'output/p06_gt_correction_01/acceptance'
DATA=ROOT/'output/p06_runtime_p03_gtfix01'
WEIGHT=ROOT/'output/p06_runs/p03_baseline3_640_seed42/weights/best.pt'
WEIGHT_HASH='4c5a1f759623a88ed2d21d639a4f3254e9da92d89107e4cdbfd6bddcf4e76b3d'
OUT=ROOT/'output/p06_runs/p03_baseline3_resolution_640_960'
CREATED=False

def summarize(rows):
    return dict(images=len(rows),detections=sum(r['detections'] for r in rows),
        class_counts=[sum(r['class_counts'][i] for r in rows) for i in range(6)],
        class_confusions=sum(len(r['confusions_gt_to_pred']) for r in rows),
        unmatched_gt=sum(len(r['unmatched_gt']) for r in rows),
        extra_predictions=sum(len(r['unmatched_predictions']) for r in rows),
        possible_duplicates=sum(p['possible_duplicate'] for r in rows for p in r['unmatched_predictions']),
        flagged_images=[r['image_id'] for r in rows if r['confusions_gt_to_pred'] or r['unmatched_gt'] or r['unmatched_predictions']])

def changes(a,b):
    require(len(a)==len(b) and [r['image_id'] for r in a]==[r['image_id'] for r in b],'Comparison membership/order differs')
    keys=['detections','class_counts','confusions_gt_to_pred','unmatched_gt','unmatched_predictions']
    return [dict(image_id=x['image_id'],**{str(size):{k:r[k] for k in keys} for size,r in [(640,x),(960,y)]})
            for x,y in zip(a,b) if any(x[k]!=y[k] for k in keys)]

def benchmark(model,records,imgsz,torch):
    import numpy as np
    # Same real Validation images, batch one (camera-style), no drawing in timed region.
    args=dict(imgsz=imgsz,conf=.25,iou=.7,device=0,batch=1,agnostic_nms=False,verbose=False)
    for row in records[:3]: model.predict(str(DATA/'images/val'/row['image_id']),**args)
    torch.cuda.synchronize();gc.collect();torch.cuda.empty_cache();torch.cuda.reset_peak_memory_stats(0)
    times=[]
    for repeat in range(2):
        for row in records:
            torch.cuda.synchronize();start=perf_counter()
            result,=model.predict(str(DATA/'images/val'/row['image_id']),**args)
            torch.cuda.synchronize();elapsed=(perf_counter()-start)*1000
            require(result.boxes.data.is_cuda,'Prediction moved off CUDA')
            require(bool(torch.isfinite(result.boxes.data).all()),'Nonfinite prediction')
            times.append(dict(repeat=repeat+1,image_id=row['image_id'],call_ms=elapsed,**result.speed))
    result=dict(samples=len(times),warmup_images=3,repeats=2,batch=1,
        call_includes='JPEG reading/decoding, preprocessing, inference, postprocessing and Python overhead; excludes rendering',
        gpu_inference_mean_ms=float(np.mean([r['inference'] for r in times])),
        gpu_inference_median_ms=float(np.median([r['inference'] for r in times])),
        gpu_inference_p95_ms=float(np.percentile([r['inference'] for r in times],95)),
        call_mean_ms=float(np.mean([r['call_ms'] for r in times])),
        call_median_ms=float(np.median([r['call_ms'] for r in times])),
        call_p95_ms=float(np.percentile([r['call_ms'] for r in times],95)),
        peak_allocated_bytes=int(torch.cuda.max_memory_allocated(0)),
        model_fp16=bool(model.predictor.model.fp16),raw_samples=times)
    return result

def main():
    global CREATED
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    require(not OUT.exists(),'Refuse to overwrite resolution comparison')
    require(subprocess.check_output(['git','branch','--show-current'],cwd=ROOT,text=True).strip()=='feat/p06-baseline3','Unexpected Git branch')
    import torch
    environment=inspect_environment(torch)
    require(digest(WEIGHT)==WEIGHT_HASH,'Unexpected best.pt hash')
    plan=accepted_plan(SOURCE,ACCEPTANCE)
    records=[r for r in plan['records'] if r['split']=='val']
    require(len(records)==46 and plan['summary']['val']['classes']==[14,46,45,44,45,45],'Unexpected corrected Validation')
    config=dict(sizes=[640,960],weight=str(WEIGHT),weight_sha256=WEIGHT_HASH,environment=environment,
        validation=dict(batch=8,device=0,workers=4,conf=.001,iou=.7,agnostic_nms=False,plots=True),
        prediction=dict(batch=1,device=0,conf=.25,iou=.7,agnostic_nms=False),
        source=str(SOURCE),acceptance=str(ACCEPTANCE),training_started=False,test_accessed=False)
    if args.dry_run: print(json.dumps(config,indent=2));return 0
    configure_runtime();official=prepare_runtime(plan,DATA)
    before=snapshot(DATA)
    OUT.mkdir();CREATED=True;save_json(OUT/'config.json',dict(config,official_loader=official))
    from ultralytics import YOLO
    results={}; predictions={}
    with (OUT/'console.log').open('x',encoding='utf-8') as stream:
        with redirect_stdout(Tee(sys.stdout,stream)),redirect_stderr(Tee(sys.stderr,stream)),chdir(DATA):
            for size in [640,960]:
                run=OUT/str(size);run.mkdir();torch.cuda.empty_cache()
                model=YOLO(str(WEIGHT));require(list(model.names.values())==list(CLASSES),'Checkpoint class order differs')
                observed={}
                model.add_callback('on_val_end',lambda validator: observed.update(conf=validator.args.conf,iou=validator.args.iou,imgsz=validator.args.imgsz))
                torch.cuda.reset_peak_memory_stats(0)
                metrics=validate_best(model,DATA,run,plan,imgsz=size)
                val_peak=int(torch.cuda.max_memory_allocated(0))
                require(observed==dict(conf=.001,iou=.7,imgsz=size),'Unexpected validation thresholds/size')
                predictions[size]=predict_val(model,DATA,run,plan,imgsz=size)
                require(model.predictor.args.conf==.25 and model.predictor.args.iou==.7 and model.predictor.args.batch==1,'Prediction thresholds/batch changed')
                timing=benchmark(model,records,size,torch)
                save_json(run/'timing.json',timing)
                results[str(size)]=dict(metrics=metrics,diagnostics=summarize(predictions[size]),
                    timing={k:v for k,v in timing.items() if k!='raw_samples'},validation_peak_allocated_bytes=val_peak)
                del model;gc.collect();torch.cuda.empty_cache()
    baseline=json.loads((ROOT/'output/p06_runs/p03_baseline3_gtfix01_revalidation/val_predictions/prediction_summary.json').read_text(encoding='utf-8'))
    baseline_same=all(x['image_id']==y['image_id'] and x['predictions']==y['predictions'] for x,y in zip(baseline,predictions[640])) and len(baseline)==len(predictions[640])
    check_sources(plan);require(snapshot(DATA)==before,'Runtime input changed')
    require(digest(WEIGHT)==WEIGHT_HASH,'Weight changed')
    save_json(OUT/'comparison.json',dict(status='PASS',results=results,changes=changes(predictions[640],predictions[960]),
        reproduced_previous_640_predictions=baseline_same,source_and_runtime_unchanged=True,weight_unchanged=True,
        training_started=False,test_accessed=False,
        pass59={str(size):next(r for r in predictions[size] if r['image_id']=='pass59.jpg') for size in [640,960]},
        limitations=['Validation comparison only; no final Test conclusion','Sequential single-session timing, not a controlled deployment benchmark',
            'Greedy IoU .5 diagnostics are not official AP or human review; previous four human-approved images are not reclassified automatically']))
    print(json.dumps(results,indent=2));return 0

if __name__=='__main__':
    try: raise SystemExit(main())
    except Exception as exc:
        if CREATED:
            import traceback
            (OUT/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
        raise
