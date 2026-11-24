"""Independent post-training leakage audit; never opens held-out test assets."""
import csv
import hashlib
import json
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from threadpoolctl import threadpool_limits

from src.data.development import ROOT, digest, write_json
from src.models.baseline1.features import load_image_bgr, extract_hog
from src.p07_bridge import load_backend


def main():
    out=ROOT/'outputs/leakage_audit'/str(time.time_ns());out.mkdir(parents=True)
    data=json.loads((ROOT/'outputs/training/data/manifest.json').read_text(encoding='utf-8'))
    records=data['records'];fixed=data['evidence']['expected_counts']
    assert len(records)==308 and {r['split'] for r in records}=={'train','validation'}
    approved={r['image_id']:r for r in records}
    def check_image(r):
        path=(ROOT/r['path']).resolve()
        assert path.parent.name==r['split'] and path.parent.parent.name=='images'
        assert digest(path)==r['sha256']
        with Image.open(path) as im:
            rgb=ImageOps.exif_transpose(im).convert('RGB')
            h=hashlib.sha256(str(rgb.size).encode()+rgb.tobytes()).hexdigest()
        assert h==r['pixel_sha256']
        return dict(image_id=r['image_id'],split=r['split'],sha256=r['sha256'],pixel_sha256=h)
    with ThreadPoolExecutor(max_workers=4) as pool:
        checked=[]
        for r in pool.map(check_image,records):
            checked.append(r)
            if len(checked)%40==0:print('AUDIT_PIXELS',len(checked),flush=True)
    duplicates={}
    for field in ('sha256','pixel_sha256'):
        groups=defaultdict(list)
        for r in checked:groups[r[field]].append(r)
        duplicates[field]=[v for v in groups.values() if len(v)>1]
        assert not duplicates[field]
    validation=[r for r in records if r['split']=='validation']
    trainmask=np.array([r['split']=='train' for r in records])
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    acceptance=max((ROOT/'outputs/acceptance').iterdir(),key=lambda p:p.name)
    evidence=dict(official_held_out_test='NOT_RUN',manifest_sha256=data['evidence']['manifest_sha256'],
                  scene_independence_basis=data['evidence']['scene_confirmation'],
                  image_count=len(records),duplicate_groups=duplicates,models={})
    for name in ('baseline1','baseline2'):
        backend=load_backend('real',name,config)
        assert backend.is_mock is False
        cache_specs=[]
        for p in (ROOT/'outputs/training/features').glob(name+'-*.json'):
            spec=json.loads(p.read_text(encoding='utf-8'))
            if spec['manifest']==data['evidence']['manifest_sha256']:cache_specs.append((p,spec))
        assert len(cache_specs)==1
        cache,signature=cache_specs[0]
        with np.load(cache.with_suffix('.npz'),allow_pickle=False) as f:X=f['X']
        assert X.shape[0]==len(records)
        expected_mean=X[trainmask].mean(axis=0)
        expected_scale=X[trainmask].std(axis=0);expected_scale[expected_scale==0]=1
        if name=='baseline1':
            native=backend.native;estimator=native.bundle['mlp'];scaler=estimator.named_steps['standardscaler']
            mean,scale=scaler.mean_,scaler.scale_
            assert int(scaler.n_samples_seen_)==int(trainmask.sum())==262
        else:
            native=backend.native.adapter;mean,scale=native.mean,native.scale
            assert native.features.backbone.training is False
            assert not any(p.requires_grad for p in native.features.backbone.parameters())
        # HOG cache is float32; sklearn accumulates scaler statistics in float64.
        assert np.allclose(mean,expected_mean,rtol=2e-5,atol=1e-6)
        assert np.allclose(scale,expected_scale,rtol=2e-5,atol=1e-6)
        original={r['image_id']:r for r in csv.DictReader((ROOT/'outputs/training'/name/'validation_predictions.csv').open(encoding='utf-8'))}
        batch={r['image_id']:r for r in csv.DictReader((acceptance/name/'images.csv').open(encoding='utf-8'))}
        checks=[]
        for r in validation:
            # Pass only pixels and requested counts to the deployed GUI backend.
            frame=load_image_bgr(str(ROOT/r['path']))
            with threadpool_limits(limits=4):prediction=backend.infer(frame,fixed)
            raw=original[r['image_id']];cli=batch[r['image_id']]
            assert prediction['predicted_label']==raw['predicted_label']==cli['predicted_label']
            score=prediction['prediction_score']
            p_fail=(score if prediction['predicted_label']=='FAIL' else 1-score) if name=='baseline1' else score
            assert np.isclose(p_fail,float(raw['p_fail']),atol=2e-6)
            assert np.isclose(score,float(cli['prediction_score']),atol=2e-6)
            idx=records.index(r)
            if name=='baseline1':
                feature=extract_hog(frame,tuple(native.bundle['image_size']),native.bundle['hog_params'])
            else:
                # Deliberately unrelated image_id: ID is metadata, not an input feature.
                anonymous=native.predict_frame(frame,fixed,image_id='anonymous.png')
                assert anonymous['predicted_label']==prediction['predicted_label']
                assert np.isclose(anonymous['prediction_score'],score,atol=2e-6)
                feature=native.features.extract([Image.fromarray(frame[...,::-1])])[0]
            assert np.allclose(feature,X[idx],rtol=2e-5,atol=2e-5)
            checks.append(dict(image_id=r['image_id'],true_label=r['label'],training_label=raw['predicted_label'],
                               deployment_label=prediction['predicted_label'],batch_label=cli['predicted_label'],
                               training_p_fail=float(raw['p_fail']),deployment_score=score,batch_score=float(cli['prediction_score']),
                               feature_matches_cache=True))
        write_json(out/(name+'_per_image_comparison.json'),checks)
        evidence['models'][name]=dict(result='PASS',reloaded_real_model=True,validation_images_compared=len(checks),
                        training_deployment_batch_agree=True,train_only_scaler_statistics_match=True,
                        cached_validation_features_match_recomputed_pixels=True,
                        cache_file=cache.with_suffix('.npz').relative_to(ROOT).as_posix(),cache_sha256=digest(cache.with_suffix('.npz')),
                        cache_signature=signature,model_version=backend.metadata['model_version'],
                        feature_inputs='decoded pixels only; no filename, directory, labels or truth',
                        scaler_fit_scope='262 training images only',classifier_fit_scope='X[trainmask], y[trainmask]',
                        selection_scope='46 validation images; no additional optimization in this audit')
    sources=['src/data/development.py','scripts/train_candidates.py','src/models/baseline1/features.py',
             'src/models/baseline1/model.py','src/adapters/baseline1.py','src/adapters/baseline2.py',
             'src/inference/batch.py','src/p07_bridge.py','app.py']
    evidence['audited_source_sha256']={p:digest(ROOT/p) for p in sources}
    evidence['source_review']={
        'fitting':'StandardScaler and classifier share estimator.fit(X[trainmask], y[trainmask]); validation uses predict only',
        'other_fitted_operations':'No PCA, learned feature extraction, calibration or threshold optimization; HOG is deterministic; pretrained ResNet18 frozen/eval',
        'historical_artifacts':'Old synthetic classifier and scripts archived outside production; cache signed by current real-data manifest/preprocessing/backbone/dependencies',
        'test_isolation':'Only approved 308 development record paths opened; no test photos or test answers accessed by this audit',
        'scene_caveat':'User confirmation only; absence of duplicate hashes does not prove physical-scene independence',
        'gui':'App and batch use load_backend; backend is_mock=False; synthetic class exists only in test fixtures; desktop click verification not completed'}
    write_json(out/'audit.json',evidence);write_json(out/'image_hashes.json',checked)
    write_json(ROOT/'outputs/leakage_audit/latest.json',dict(directory=out.relative_to(ROOT).as_posix()))
    print('LEAKAGE_AUDIT_PASS',out,flush=True)


if __name__=='__main__':main()
