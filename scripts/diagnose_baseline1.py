"""Train-chain and synthetic OOD diagnostics; no test-set inputs or model tuning."""
import json
from pathlib import Path
import hashlib
import numpy as np
import cv2
from scipy.spatial.distance import cdist
from sklearn.metrics import confusion_matrix
from threadpoolctl import threadpool_limits
from src.data.development import ROOT, write_json, digest
from src.p07_bridge import load_backend


def main():
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf-8'))
    raw=load_backend('real','baseline1',config)
    native=raw.native
    manifest=json.loads((ROOT/'outputs/training/data/manifest.json').read_text(encoding='utf-8'))
    records=manifest['records'];mask=np.array([r['split']=='train' for r in records])
    cache=next((ROOT/'outputs/training/features').glob('baseline1-*.npz'))
    with np.load(cache,allow_pickle=False) as data:X=data['X']
    scaler=native.bundle['mlp'].named_steps['standardscaler']
    Z=scaler.transform(X);reference=Z[mask]
    distances=cdist(reference,reference);np.fill_diagonal(distances,np.inf)
    limit=float(distances.min(axis=1).max())
    # Training-only leave-one-out support envelope; no OOD/validation threshold search.
    guard_path=ROOT/'models/baseline1/scene_guard.npz'
    if not guard_path.exists():
        np.savez_compressed(guard_path,reference=reference,limit=limit,model_sha256=raw.metadata['model_version'])
    live=load_backend('real','baseline1',dict(config,_live_scene_guard=True))
    with threadpool_limits(limits=4):
        probs=native.bundle['mlp'].predict_proba(X[mask])
        predictions=native.bundle['mlp'].predict(X[mask])
    train=[r for r in records if r['split']=='train']
    output=ROOT/'outputs/baseline1_diagnosis';output.mkdir(parents=True,exist_ok=True)
    classes=list(native.bundle['mlp'].classes_)
    rows=[dict(image_id=r['image_id'],true_label=r['label'],predicted_label=str(p),
               probabilities={c:float(v) for c,v in zip(classes,prob)}) for r,p,prob in zip(train,predictions,probs)]
    write_json(output/'training_diagnostics.json',rows)
    cases={}
    for label in ('PASS','FAIL'):
        for r in [r for r in train if r['label']==label][:3]:
            assert digest(ROOT/r['path'])==r['sha256']
            cases['train_'+r['image_id']]=native.features.load_image_bgr(str(ROOT/r['path']))
    cases.update(black=np.zeros((480,640,3),np.uint8),white=np.full((480,640,3),255,np.uint8),
                 grey=np.full((480,640,3),128,np.uint8),
                 noise=np.random.default_rng(42).integers(0,256,(480,640,3),dtype=np.uint8))
    book=np.full((480,640,3),210,np.uint8)
    cv2.rectangle(book,(80,60),(560,420),(45,90,160),-1)
    for y in range(100,400,25):cv2.putText(book,'UNRELATED TEXT PAGE',(100,y),cv2.FONT_HERSHEY_SIMPLEX,.6,(230,230,230),1)
    cases['synthetic_book_texture']=book
    comparison={}
    for name,image in cases.items():
        with threadpool_limits(limits=4):
            v=native.features.extract_hog(image,tuple(native.bundle['image_size']),native.bundle['hog_params'])
            original=native.model.predict(native.bundle,v[None])[0]
            guarded=live.infer(image,live.fixed)
        comparison[name]=dict(raw=original,system_status=guarded['status'],system_label=guarded['predicted_label'],
                              system_score=guarded['prediction_score'],distance=float(cdist(scaler.transform(v[None]),reference).min()))
    validation_distances=cdist(Z[~mask],reference).min(axis=1)
    summary=dict(scope='Training-chain and synthetic functional diagnostics ONLY; not formal evaluation',
        official_held_out_test='NOT_RUN',model_sha256=digest(native.path),guard_sha256=digest(guard_path),
        classes=classes,training_distribution={label:sum(r['label']==label for r in train) for label in classes},
        training_confusion_matrix=confusion_matrix([r['label'] for r in train],predictions,labels=['PASS','FAIL']).tolist(),
        matrix_order=['PASS','FAIL'],guard_threshold=limit,guard_fit='262 training features only; maximum leave-one-out nearest distance',
        validation_support_coverage=int(sum(validation_distances<=limit)),validation_images=46,
        diagnostics=comparison,preprocessing=native.metadata['preprocessing'],
        scaler_n_samples=int(scaler.n_samples_seen_),classifier_iterations=int(native.bundle['mlp'].steps[-1][1].n_iter_))
    write_json(output/'diagnosis.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2))


if __name__=='__main__':main()
