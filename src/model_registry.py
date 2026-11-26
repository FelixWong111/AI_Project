"""One frozen production registry shared by GUI, batch and P08."""
from __future__ import annotations
import copy,hashlib,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
PRODUCTION_MODELS=('baseline1','baseline2','baseline3')

def portable(value):
    """Repository paths in exported metadata remain usable after cloning."""
    if isinstance(value,dict):return {k:portable(v) for k,v in value.items()}
    if isinstance(value,(list,tuple)):return [portable(v) for v in value]
    if isinstance(value,str):
        for prefix in (str(ROOT)+'\\',ROOT.as_posix()+'/'):
            if value.startswith(prefix):return value[len(prefix):].replace('\\','/')
    return value

def resolve(model,config):
    if model=='final':raise RuntimeError('final NOT_READY: no independent Stage3 improved model has been delivered')
    value=config.get('model_registry')
    if value is None:return config,[]
    path=ROOT/value
    registry=json.loads(path.read_text(encoding='utf-8-sig'))
    if registry.get('schema')!='frozen-model-registry/v1' or set(registry['models'])!=set(PRODUCTION_MODELS):
        raise ValueError('Invalid production model registry')
    entry=registry['models'][model];files=[path]
    for filename,expected in dict(registry['shared_files'],**entry['files']).items():
        p=(ROOT/filename).resolve()
        if Path(filename).is_absolute() or not p.is_relative_to(ROOT):raise ValueError('Registry paths must be repository-relative')
        with p.open('rb') as f:actual=hashlib.file_digest(f,'sha256').hexdigest()
        if actual!=expected:raise RuntimeError('Frozen model SHA256 mismatch: '+filename)
        files.append(p)
    result=copy.deepcopy(config)
    result.update(bridge=entry['bridge'],model_config=entry['model_config'],_live_scene_guard=entry['scene_guard'])
    result['_registry_metadata']=dict(path=value,sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        frozen_version=entry['version'],physical_acceptance=entry.get('physical_acceptance','NOT_APPLICABLE'),
        scene_guard=entry['scene_guard'],no_fallback=True)
    return result,files
