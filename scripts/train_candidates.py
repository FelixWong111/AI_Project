"""Audited real-data training. Does not enumerate or open held-out test assets."""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata
import json
import platform
import shutil
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from src.work_orders import reference_counts
from src.data.development import ROOT, digest, discover_development, prepare, write_json
from src.adapters.baseline2 import (ResNet18Features, read_rgb, WEIGHT_FILE, WEIGHT_SHA256,
                                    PREPROCESSING, model_version)
from src.models.baseline1 import features, model


def canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def environment():
    names = ('numpy','scipy','scikit-learn','scikit-image','Pillow','torch','torchvision','joblib','opencv-python')
    import torch
    return dict(python=sys.version, executable=sys.executable, platform=platform.platform(),
                processor=platform.processor(), device='cpu', torch_threads=4,
                dependencies={n:importlib.metadata.version(n) for n in names},
                cuda_available=torch.cuda.is_available())


def metrics(records, truth, predicted, probabilities):
    tn,fp,fn,tp = confusion_matrix(truth,predicted,labels=['PASS','FAIL']).ravel()
    p,r,f,_ = precision_recall_fscore_support(truth,predicted,pos_label='FAIL',average='binary',zero_division=0)
    return dict(positive_class='FAIL', accuracy=float(accuracy_score(truth,predicted)),
                precision=float(p),recall=float(r),f1=float(f),TP=int(tp),FP=int(fp),FN=int(fn),TN=int(tn),
                false_PASS=int(fn),false_FAIL=int(fp),confusion_matrix=[[int(tn),int(fp)],[int(fn),int(tp)]],
                matrix_order=['PASS','FAIL'],support_PASS=int(sum(truth=='PASS')),support_FAIL=int(sum(truth=='FAIL')),
                misclassified=[dict(image_id=a['image_id'],true_label=str(t),predicted_label=str(v),p_fail=float(s))
                               for a,t,v,s in zip(records,truth,predicted,probabilities) if t!=v])


def prepare_once(data_dir, settings, directory):
    directory.mkdir(parents=True,exist_ok=True)
    records,evidence=prepare(data_dir,settings)
    evidence['manifest_sha256']=canonical(records)
    evidence['settings_sha256']=canonical(settings)
    write_json(directory/'manifest.json',dict(records=records,evidence=evidence))
    for split in ('train','validation'):
        write_json(directory/(split+'_counts.json'),[
            dict(image_id=r['image_id'],expected_counts=settings['expected_counts']) for r in records if r['split']==split])
    return records,evidence


def audited_manifest(data_dir, settings, directory):
    path=directory/'manifest.json'
    if not path.exists():return prepare_once(data_dir,settings,directory)
    data=json.loads(path.read_text(encoding='utf-8'));records=data['records'];evidence=data['evidence']
    if evidence['settings_sha256']!=canonical(settings) or evidence['manifest_sha256']!=canonical(records):
        raise ValueError('Prepared manifest/settings changed; prepare a new audited manifest')
    for name,sha in evidence['source_files'].items():
        if digest(ROOT/name)!=sha:raise ValueError('Source annotations changed; re-audit')
    for r in records:
        p=(ROOT/r['path']).resolve()
        from src.data.dataset_v2 import approved_development_path
        if r['split'] not in ('train','validation') or not approved_development_path(data_dir, r, p):
            raise ValueError('Prepared manifest escaped approved development partition')
        if digest(p)!=r['sha256']:raise ValueError(f'Image changed: {r["image_id"]}')
    return records,evidence


def train(args):
    started=time.perf_counter()
    import torch
    torch.set_num_threads(4);torch.manual_seed(42)
    settings_path=getattr(args,'dataset_config',None) or ROOT/'datasets/mechanical_components_v2/configs/training.json'
    settings=json.loads(Path(settings_path).read_text(encoding='utf-8'))
    data_description=settings.get('data_description','P03 real mechanical component images')
    settings['expected_counts']=reference_counts()
    data_dir=args.data_dir or (ROOT/settings['data_dir'] if 'data_dir' in settings else discover_development())
    if getattr(args,'audit_candidate',False):
        if not args.prepare_only:raise ValueError('--audit-candidate is data validation only; requires --prepare-only')
        settings['allow_candidate_annotations']=True
    if (Path(data_dir)/'dataset_v2.json').exists() and not args.prepare_only:
        marker=json.loads((Path(data_dir)/'dataset_v2.json').read_text(encoding='utf-8'))
        state=json.loads((ROOT/marker['status_file']).read_text(encoding='utf-8'))
        if state.get('status')!='READY' or state.get('frozen') is not True:
            raise ValueError('Formal training blocked: new dataset is not READY and frozen')
    prepared=ROOT/settings.get('prepared_dir','outputs/training/data')
    if getattr(args,'dataset_config',None):prepared=prepared/canonical(settings)[:12]
    records,evidence=audited_manifest(data_dir,settings,prepared)
    if args.prepare_only:
        print(json.dumps(evidence,ensure_ascii=False,indent=2));return
    name=args.model
    if getattr(args,'dataset_config',None) and args.output_dir is None:
        raise ValueError('A new dataset requires an explicit NEW --output-dir; frozen model paths are protected')
    dest=Path(args.output_dir or ROOT/'models'/name).resolve()
    if dest.exists() and any(dest.iterdir()):raise FileExistsError(f'Use an empty candidate directory: {dest}')
    report=ROOT/settings.get('report_root','outputs/training')/name
    if getattr(args,'dataset_config',None):report=report/dest.name
    report.mkdir(parents=True,exist_ok=True)
    config=json.loads((ROOT/'config/baseline1.json').read_text(encoding='utf-8'))
    cache_spec=dict(manifest=evidence['manifest_sha256'],model=name,
                    preprocessing=PREPROCESSING if name=='baseline2' else dict(size=config['image_size'],hog=config['hog']),
                    backbone=WEIGHT_SHA256 if name=='baseline2' else None,
                    versions=environment()['dependencies'])
    cache=prepared.parent/'features'/f'{name}-{canonical(cache_spec)[:16]}.npz'
    cache.parent.mkdir(parents=True,exist_ok=True)
    if cache.exists():
        with np.load(cache,allow_pickle=False) as f:X=f['X']
    elif name=='baseline1':
        X=np.stack([features.extract_hog(features.load_image_bgr(str(ROOT/r['path'])),tuple(config['image_size']),config['hog']) for r in records])
        np.savez_compressed(cache,X=X)
    else:
        weights=args.weights or ROOT/'models/baseline2'/WEIGHT_FILE
        if digest(weights)!=WEIGHT_SHA256:raise ValueError('Not the official ResNet18 ImageNet checkpoint')
        extractor=ResNet18Features(weights,'cpu');chunks=[]
        for start in range(0,len(records),args.batch_size):
            chunks.append(extractor.extract([read_rgb(ROOT/r['path']) for r in records[start:start+args.batch_size]]))
            print(f'FEATURES {min(start+args.batch_size,len(records))}/{len(records)}',flush=True)
        X=np.concatenate(chunks);np.savez_compressed(cache,X=X)
    write_json(cache.with_suffix('.json'),cache_spec)
    if len(X)!=len(records) or not np.isfinite(X).all():raise ValueError('Invalid feature cache')
    mask=np.array([r['split']=='train' for r in records]);y=np.array([r['label'] for r in records])
    trials=[];best=None
    candidates=([dict(hidden_layer_sizes=(32,),alpha=.01),dict(hidden_layer_sizes=(64,),alpha=.01)]
                if name=='baseline1' else [dict(C=.1),dict(C=1.0),dict(C=10.0)])
    if getattr(args,'original_only',False):candidates=candidates[:1]
    for params in candidates:
        classifier=(MLPClassifier(**params,solver='adam',activation='relu',max_iter=1000,
                                  early_stopping=False,n_iter_no_change=30,random_state=42)
                    if name=='baseline1' else LogisticRegression(**params,solver='lbfgs',max_iter=3000,random_state=42))
        estimator=make_pipeline(StandardScaler(),classifier)
        fit_started=time.perf_counter()
        with warnings.catch_warnings(record=True) as captured,threadpool_limits(limits=4):
            warnings.simplefilter('always')
            fitargs={'mlpclassifier__sample_weight':model.balanced_sample_weights(y[mask])} if name=='baseline1' else {}
            estimator.fit(X[mask],y[mask],**fitargs)
        probs=estimator.predict_proba(X[~mask])[:,list(estimator.classes_).index('FAIL')]
        predicted=estimator.predict(X[~mask])
        result=metrics([r for r in records if r['split']=='validation'],y[~mask],predicted,probs)
        trial=dict(parameters=params,validation=result,warnings=[str(w.message) for w in captured],n_iter=classifier.n_iter_)
        trial.update(fit_seconds=time.perf_counter()-fit_started,
                     loss_curve=getattr(classifier,'loss_curve_',None),
                     scaler_fit_samples=int(estimator.steps[0][1].n_samples_seen_),fit_split='train only')
        trial['n_iter']=np.asarray(trial['n_iter']).tolist();trials.append(trial)
        print('TRIAL',name,params,'F1',result['f1'],'accuracy',result['accuracy'],flush=True)
        # F1 first, then fewer false PASS; ties retain simpler first candidate.
        key=(result['f1'],-result['false_PASS'])
        if best is None or key>best[0]:best=(key,estimator,params)
    _,estimator,selected=best
    allprob=estimator.predict_proba(X);classes=list(estimator.classes_)
    predicted=estimator.predict(X);p_fail=allprob[:,classes.index('FAIL')]
    results={split:metrics([r for r in records if r['split']==split],y[mask if split=='train' else ~mask],
                          predicted[mask if split=='train' else ~mask],p_fail[mask if split=='train' else ~mask])
             for split in ('train','validation')}
    dest.mkdir(parents=True,exist_ok=True)
    if name=='baseline1':
        config['mlp'].update(selected,hidden_layer_sizes=list(selected['hidden_layer_sizes']),max_iter=1000,
                             n_iter_no_change=30,class_weight='balanced',solver='adam')
        config['supported_counts_note']='Fixed reference_counts; audited object counts and reference_label_all_one.'
        config['freeze']['bundle_file']=(dest/'baseline1.joblib').relative_to(ROOT).as_posix()
        config_path=dest/'config.json' if getattr(args,'dataset_config',None) else ROOT/'config/baseline1.json'
        write_json(config_path,config)
        checksum=digest(config_path)
        bundle=model.make_bundle(config,estimator,X.shape[1],checksum,
                                 dict(data_manifest_sha256=evidence['manifest_sha256'],scene_confirmation=evidence['scene_confirmation'],
                                      official_held_out_test='NOT_RUN',trained_on=data_description))
        model.save_bundle(bundle,str(dest/'baseline1.joblib'))
        version=digest(dest/'baseline1.joblib')
        write_json(dest/'config.json',config)
        write_json(dest/'freeze.json',dict(bundle_checksum=version,config_checksum=checksum,
                                           stage='candidate',official_held_out_test='NOT_RUN'))
    else:
        weights=args.weights or ROOT/'models/baseline2'/WEIGHT_FILE
        scaler,clf=estimator.steps[0][1],estimator.steps[1][1]
        # sklearn stores the coefficient of classes_[1] == PASS; export P(FAIL).
        sign=1 if list(clf.classes_)[1]=='FAIL' else -1
        manifest=dict(schema='baseline2/v1',supported_fixed_counts=settings['expected_counts'],
                      preprocessing=PREPROCESSING,backbone='torchvision ResNet18 IMAGENET1K_V1',
                      backbone_sha256=WEIGHT_SHA256,development_set_sha256=evidence['manifest_sha256'],
                      training=dict(seed=42,**selected,solver='lbfgs',max_iter=3000,backbone_frozen=True,
                                    source=data_description,scene_confirmation=evidence['scene_confirmation'],official_held_out_test='NOT_RUN'),
                      versions=environment()['dependencies'],classifier=dict(mean=scaler.mean_.tolist(),scale=scaler.scale_.tolist(),
                            coef=(sign*clf.coef_[0]).tolist(),intercept=float(sign*clf.intercept_[0]),fail_probability_threshold=.5))
        manifest['model_version']=model_version(manifest);version=manifest['model_version']
        write_json(dest/'baseline2.json',manifest)
        # One local backbone shared by training and deploy; adapter resolves this explicit relative path.
        # Keep a complete portable package: this is a required deployment copy, not a smoke model.
        shutil.copy2(weights,dest/WEIGHT_FILE)
    files={p.name:dict(sha256=digest(p),size_bytes=p.stat().st_size) for p in dest.iterdir() if p.is_file()}
    info=dict(model=name,model_version=version,status='TRAINED_CANDIDATE',data=evidence,
              selection_rule=('Original selected hyperparameters only; no search' if getattr(args,'original_only',False)
                              else 'validation FAIL F1, then fewer false PASS; exact ties keep first candidate'),
              trials=trials,selected_parameters=selected,metrics=results,environment=environment(),
              command=subprocess.list2cmdline([sys.executable,'-m','scripts.train_'+name,*sys.argv[1:]]),
              training_seconds=time.perf_counter()-started,files=files,official_held_out_test='NOT_RUN')
    write_json(report/'training_record.json',info);write_json(dest/'training_record.json',info)
    write_json(report/'development_manifest.json',records)
    for split in ('train','validation'):
        with (report/(split+'_predictions.csv')).open('w',newline='',encoding='utf-8') as f:
            writer=csv.writer(f);writer.writerow(['image_id','true_label','predicted_label','p_fail'])
            writer.writerows((r['image_id'],r['label'],str(v),float(p)) for r,v,p in zip(records,predicted,p_fail) if r['split']==split)
    print(json.dumps(dict(model=name,version=version,metrics=results,seconds=info['training_seconds']),indent=2),flush=True)


def main(model_id=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',choices=('baseline1','baseline2'),default=model_id,required=model_id is None)
    parser.add_argument('--data-dir',type=Path)
    parser.add_argument('--dataset-config',type=Path,help='Explicit versioned data/settings; never replace the frozen default dataset')
    parser.add_argument('--audit-candidate',action='store_true',help='Data checks only with --prepare-only; cannot enable training')
    parser.add_argument('--original-only',action='store_true',help='Retrain only the original selected parameters: MLP32 alpha=.01 or LR C=.1')
    parser.add_argument('--output-dir',type=Path)
    parser.add_argument('--weights',type=Path)
    parser.add_argument('--batch-size',type=int,default=8)
    parser.add_argument('--prepare-only',action='store_true')
    args=parser.parse_args()
    if args.batch_size<1:parser.error('batch-size must be positive')
    train(args)


if __name__=='__main__':main()
