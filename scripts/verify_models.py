"""Verify the unique formal registry; optionally load real models, without inference."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from src.model_registry import PRODUCTION_MODELS,resolve

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--load-models',action='store_true');a=p.parse_args()
    config=json.loads((ROOT/'config/p07.json').read_text(encoding='utf8'));result={}
    for model in PRODUCTION_MODELS:
        settings,files=resolve(model,config)
        result[model]=dict(sha256_verified=True,files=[x.relative_to(ROOT).as_posix() for x in files],
            model_config=settings['model_config'],loaded=False)
        if a.load_models:
            from src.p07_bridge import load_backend
            backend=load_backend('real',model,config);result[model].update(loaded=True,model_version=backend.metadata['model_version'],is_mock=backend.is_mock)
    print(json.dumps(result,ensure_ascii=False,indent=2));return 0
if __name__=='__main__':raise SystemExit(main())
