"""Frozen v1 best.pt predictions on approved v2 validation; never trains."""
import csv
import argparse
import json
from pathlib import Path
import sys
from contextlib import chdir, redirect_stdout, redirect_stderr

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from tools.p06_train_baseline import CLASSES, Tee, compare_predictions, configure_runtime, inspect_environment, save_json, snapshot
from tools.p06_build_dataset import check_sources
from tools.p06_build_dataset_v2 import validate_project_v2
from tools.p06_preview_annotations import digest, font
from src.data.p06_labelme import require

DATA=ROOT/'output/p06/yolo_dataset_v2'
OUT=ROOT/'output/p06/runs/baseline1_best_on_v2_val'
WEIGHT=ROOT/'output/p06/runs/baseline1_yolo11n_640_seed42_run02/weights/best.pt'
WEIGHT_HASH='830f54ba591a561dc7afe57c600796c6ca1d00a24b5351acebf66c6329d85b17'
PREDICT=dict(imgsz=640,conf=.25,iou=.7,device=0,agnostic_nms=True,verbose=False)


def verify_support(metrics,supports,seen,expected_images):
    require(seen==expected_images and list(map(int,metrics.nt_per_class))==supports,
            'LOADER_SUPPORT','Official validator image/GT counts differ from the frozen group')


def group_records(records):
    val=[r for r in records if r['split']=='val']
    return dict(all65=val,old10=[r for r in val if r['provenance']=='old51'],new55=[r for r in val if r['provenance']=='new275'])


def count_summary(rows):
    n=len(rows)
    return dict(images=n,gt_instances=sum(sum(r['gt_counts']) for r in rows),
        predictions=sum(sum(r['class_counts']) for r in rows),
        five_counts_exact=sum(r['gt_counts'][1:]==r['class_counts'][1:] for r in rows),
        six_counts_exact=sum(r['gt_counts']==r['class_counts'] for r in rows),
        count_mae=[sum(abs(r['gt_counts'][c]-r['class_counts'][c]) for r in rows)/n if n else None for c in range(6)],
        gt_classes=[sum(r['gt_counts'][c] for r in rows) for c in range(6)],
        predicted_classes=[sum(r['class_counts'][c] for r in rows) for c in range(6)],
        id0_false_positive_instances_on_id0_free_images=sum(r['class_counts'][0] for r in rows if r['gt_counts'][0]==0),
        id0_positive_images=sum(r['gt_counts'][0]>0 for r in rows),
        empty_predictions=[dict(image_id=r['image_id'],detections=sum(r['class_counts'])) for r in rows if not sum(r['gt_counts'])])


def write_sheets(rows,out):
    from PIL import Image,ImageDraw,ImageOps
    for start in range(0,len(rows),16):
        sheet=Image.new('RGB',(2000,2520),'#eef1f5'); draw=ImageDraw.Draw(sheet)
        draw.text((20,15),f'OLD best.pt MODEL PREDICTIONS | page {start//16+1} | conf .25 | agnostic NMS',font=font(25),fill='black')
        for index,row in enumerate(rows[start:start+16]):
            x,y=index%4*500,index//4*600+90
            with Image.open(out/Path(row['image_id']).with_suffix('.png')) as image:
                thumb=ImageOps.contain(image,(480,490)); sheet.paste(thumb,(x+(500-thumb.width)//2,y+50))
            draw.text((x+8,y),row['image_id']+' | '+row['provenance'],font=font(21),fill='black')
            draw.text((x+8,y+550),str(row['class_counts']),font=font(21),fill='black')
        sheet.save(out/f'contact_{start//16+1:02}.jpg',quality=93)


def run():
    import torch
    from PIL import Image
    environment=inspect_environment(torch)
    require(digest(WEIGHT)==WEIGHT_HASH,'WEIGHT_CHANGED','Unexpected frozen best.pt')
    plan=json.loads((DATA/'source_plan.json').read_text(encoding='utf-8'))
    require(Path(plan['source']).resolve()==ROOT/'data/p03/development_supplement_v1/development_v1','UNSAFE_INPUT','Unexpected source')
    validate_project_v2(plan); check_sources(plan)
    before=snapshot(DATA)
    old_data_before=snapshot(ROOT/'output/p06/yolo_dataset')
    groups=group_records(plan['records'])
    require([len(groups[g]) for g in ('all65','old10','new55')]==[65,10,55],'GROUP_COUNTS','Wrong validation groups')
    from ultralytics import YOLO
    model=YOLO(str(WEIGHT))
    require(list(model.names.values())==list(CLASSES),'CLASS_MAPPING','Checkpoint class order differs')
    images=OUT/'predictions'; images.mkdir()
    rows=[]
    for index,source in enumerate(groups['all65'],1):
        result,=model.predict(str(DATA/'images/val'/source['png_name']),**PREDICT)
        require(result.boxes.data.is_cuda and next(model.model.parameters()).is_cuda,'NOT_CUDA','Prediction must use CUDA')
        require(bool(torch.isfinite(result.boxes.data).all()),'NONFINITE','Invalid model outputs')
        require(result.orig_shape==(source['height'],source['width']),'GEOMETRY','Wrong original-image shape')
        predictions=[dict(xyxy=b[:4],confidence=b[4],class_id=int(b[5])) for b in result.boxes.data.cpu().tolist()]
        counts=[sum(p['class_id']==c for p in predictions) for c in range(6)]
        with Image.fromarray(result.plot()[:,:,::-1]) as image:
            image.save(images/source['png_name'],compress_level=3)
        rows.append(dict(image_id=source['image_id'],provenance=source['provenance'],
            gt_counts=source['counts'],class_counts=counts,predictions=predictions,
            **compare_predictions(predictions,source['objects'])))
        if index%10==0: print(f'Predicted {index}/65',flush=True)
    save_json(OUT/'predictions.json',rows)
    with (OUT/'prediction_summary.csv').open('x',encoding='utf-8-sig',newline='') as stream:
        writer=csv.writer(stream); writer.writerow(['image_id','provenance','gt_0_5','pred_0_5','confusions','unmatched_gt','extra_predictions'])
        for r in rows: writer.writerow([r['image_id'],r['provenance'],r['gt_counts'],r['class_counts'],r['confusions_gt_to_pred'],r['unmatched_gt'],r['unmatched_predictions']])
    write_sheets(rows,images)
    history=json.loads((ROOT/'output/p06/runs/nms_ablation_yolo11n_640_seed42/B_agnostic/predictions.json').read_text(encoding='utf-8'))
    prior={r['image_id']:r for r in history}; max_delta=0.
    for row in rows:
        if row['provenance']!='old51': continue
        old=prior[row['image_id']]
        require(row['class_counts']==old['class_counts'],'OLD_PREDICTIONS_CHANGED','Old10 counts no longer reproduce')
        for x,y in zip(row['predictions'],old['predictions']):
            require(x['class_id']==y['class_id'],'OLD_PREDICTIONS_CHANGED','Old class changed')
            max_delta=max(max_delta,*(abs(a-b) for a,b in zip(x['xyxy']+[x['confidence']],y['xyxy']+[y['confidence']])))
    require(max_delta<.01,'OLD_PREDICTIONS_CHANGED','Unexpected historical coordinate/confidence drift')
    metrics={}; counts={}
    model=YOLO(str(WEIGHT))
    model.add_callback('on_val_start',lambda validator: require(validator.device.type=='cuda','NOT_CUDA','Validation must use CUDA'))
    observed={}
    model.add_callback('on_val_end',lambda validator: observed.update(seen=int(validator.seen)))
    import yaml
    configs=OUT/'group_configs'; configs.mkdir()
    for group,records in groups.items():
        txt=configs/(group+'.txt')
        txt.write_text(''.join(str(DATA/'images/val'/r['png_name'])+'\n' for r in records),encoding='utf-8')
        cfg=configs/(group+'.yaml')
        cfg.write_text(yaml.safe_dump(dict(path=str(DATA),train='images/train',val=str(txt),nc=6,names=dict(enumerate(CLASSES))),sort_keys=False),encoding='utf-8')
        supports=[sum(r['counts'][c] for r in records) for c in range(6)]
        with chdir(DATA):
            m=model.val(data=str(cfg),split='val',imgsz=640,batch=8,device=0,workers=4,agnostic_nms=True,
                        conf=.001,iou=.7,plots=True,project=str(OUT),name=group)
        verify_support(m,supports,observed.get('seen'),len(records))
        indices={int(c):i for i,c in enumerate(m.box.ap_class_index)}
        classes=[]
        for c,name in enumerate(CLASSES):
            item=dict(class_id=c,name=name,gt=supports[c],precision=None,recall=None,ap50=None,ap50_95=None)
            if supports[c] and c in indices:
                item.update(zip(('precision','recall','ap50','ap50_95'),map(float,m.box.class_result(indices[c]))))
            classes.append(item)
        metrics[group]=dict(zip(('precision','recall','map50','map50_95'),map(float,m.box.mean_results())))
        metrics[group].update(classes=classes,matrix=m.confusion_matrix.matrix.astype(int).tolist(),
                             matrix_axis='rows predicted, columns GT; last background')
        save_json(OUT/group/'metrics.json',metrics[group])
        selected=[r for r in rows if r['image_id'] in {s['image_id'] for s in records}]
        counts[group]=count_summary(selected)
        require(sum(supports)==counts[group]['gt_instances'],'GT_SUPPORT','Group support mismatch')
    require(snapshot(DATA)==before and snapshot(ROOT/'output/p06/yolo_dataset')==old_data_before,
            'DATA_CHANGED','Published dataset changed (label caches excepted)')
    check_sources(plan); validate_project_v2(plan)
    require(digest(WEIGHT)==WEIGHT_HASH,'WEIGHT_CHANGED','Frozen weight changed')
    save_json(OUT/'completion.json',dict(status='PASS',training_started=False,environment=environment,
        weight=str(WEIGHT),weight_sha256=WEIGHT_HASH,prediction_parameters=PREDICT,
        official_validation_parameters=dict(imgsz=640,batch=8,conf=.001,iou=.7,agnostic_nms=True,device=0,workers=4),
        metrics=metrics,count_summaries=counts,old10_max_coordinate_or_conf_delta=max_delta,
        original_sources_and_v1_v2_unchanged=True,
        limitations=['Development validation only; not held-out test','Old10 previously used for selection','Fixed .25 counts and .001 PR/mAP evaluation use different thresholds','No supplied expected_counts, no image-level PASS/FAIL evaluation','ID0 physical subtype and calibration coverage unknown']))


def main():
    global OUT
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=OUT)
    args=parser.parse_args()
    OUT=args.output_dir.resolve()
    require(OUT.is_relative_to(ROOT/'output/p06/runs'),'UNSAFE_OUTPUT','Reference outputs must stay under output/p06/runs')
    if OUT.exists(): raise FileExistsError(f'Refusing existing reference run: {OUT}')
    configure_runtime(); OUT.mkdir(parents=True,exist_ok=False)
    with (OUT/'run.log').open('x',encoding='utf-8') as log:
        with redirect_stdout(Tee(sys.stdout,log)),redirect_stderr(Tee(sys.stderr,log)):
            try: run()
            except BaseException:
                import traceback
                (OUT/'failure.log').write_text(traceback.format_exc(),encoding='utf-8')
                raise


if __name__=='__main__': main()
