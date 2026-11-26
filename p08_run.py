"""Run one frozen project model and evaluate its outputs on a separate test set."""
import argparse
import csv
import json
import hashlib
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from src.evaluator import evaluate

ROOT = Path(__file__).resolve().parent


def evaluation_inputs(config_path, check_only=False):
    """Explicit stage lists; checking data never runs a prediction or computes a metric."""
    config=json.loads(Path(config_path).read_text(encoding='utf-8-sig'))
    if config.get('schema')!='inspection-evaluation/v2' or config.get('evaluation_stage') not in ('stage2','stage3','validation'):
        raise ValueError('Unsupported evaluation data config')
    paths={k:(ROOT/config[k]).resolve() for k in ('dataset','input_dir','ground_truth_images','ground_truth_objects','counts_file','split_list','dataset_status')}
    if any(not (p.is_relative_to(ROOT/'datasets') or p.is_relative_to(ROOT/'data')) or p.is_relative_to(ROOT/'data/sealed') for p in paths.values()):
        raise ValueError('Evaluation paths must stay in publishable datasets; sealed local data are forbidden')
    state=json.loads(paths['dataset_status'].read_text(encoding='utf-8'))
    if not check_only and (state.get('status')!='READY' or state.get('frozen') is not True):
        raise ValueError('Dataset is not READY/frozen; formal evaluation remains NOT_RUN')
    from src.evaluator import read_csv, unique, box
    from src.work_orders import load_cases
    cases=load_cases(paths['counts_file'])
    rows=unique(read_csv(paths['ground_truth_images'],['image_id','evaluation_stage']), 'image_id','stage manifest')
    ids={r['image_id'] for r in cases}
    listed={Path(p).name for p in paths['split_list'].read_text(encoding='utf-8').splitlines() if p}
    phase=config['evaluation_stage']
    valid_stage=lambda r: (r['evaluation_stage']=='development' and r.get('split')=='validation') if phase=='validation' else r['evaluation_stage']==phase
    if ids!=set(rows) or ids!=listed or any(not valid_stage(r) for r in rows.values()):
        raise ValueError('Stage list/GT/work-order mismatch')
    if any(not (paths['input_dir']/i).is_file() for i in ids):raise ValueError('Missing listed image')
    for obj in read_csv(paths['ground_truth_objects'],['image_id','class_id','xmin','ymin','xmax','ymax']):
        if obj['image_id'] not in rows:raise ValueError('Object GT crosses evaluation stage')
        row=rows[obj['image_id']];box(obj,int(row['width']),int(row['height']),obj['image_id'])
    return paths,dict(stage=config['evaluation_stage'],images=len(ids),dataset_status=state['status'],predictions='NOT_RUN',metrics='NOT_RUN')


def run(dataset, model, results_root, evaluation_config=None, model_config=None, record_timing=False):
    selected=None
    if evaluation_config:
        selected,_=evaluation_inputs(evaluation_config)
        data=selected['dataset']
    else:
        data = Path(dataset).resolve()
        if (data/'dataset_status.json').exists():
            raise ValueError('Versioned datasets require --evaluation-config with an explicit Stage 2 or Stage 3 list')
    for name in ('ground_truth_images.csv', 'ground_truth_objects.csv', 'expected_counts.json'):
        if not selected and not (data/name).is_file():
            raise ValueError(f'测试集缺少文件: {data/name}')
    if not selected and not (data/'images').is_dir():
        raise ValueError(f'测试集缺少图片文件夹: {data / "images"}')
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = results_root / f'{model}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    run_dir.mkdir()
    pred = run_dir/'predictions'
    input_dir=selected['input_dir'] if selected else data/'images'
    counts_file=selected['counts_file'] if selected else data/'expected_counts.json'
    active_config=Path(model_config).resolve() if model_config else ROOT/'config/p07.json'
    from src.model_registry import resolve
    settings,_=resolve(model,json.loads(active_config.read_text(encoding='utf-8-sig')))
    command = [sys.executable, str(ROOT/'predict.py'), '--input_dir', str(input_dir),
               '--model', model, '--output_dir', str(pred), '--counts_file', str(counts_file),
               '--config', str(active_config)]
    if record_timing:command.append('--record-timing')
    # Explicit UTF-8 prevents Windows console encoding from corrupting saved logs.
    with (run_dir/'command.json').open('w', encoding='utf-8') as f:
        json.dump({'command': command, 'cwd': str(ROOT), 'dataset': str(data),
                   'baseline3_mode': settings.get('model_config',{}).get('baseline3_mode') if model == 'baseline3' else None,
                   'model_config': str(active_config),
                   'model_config_sha256': hashlib.sha256(active_config.read_bytes()).hexdigest()}, f, ensure_ascii=False, indent=2)
    print(f'正在运行 {model}，请稍候……', flush=True)
    proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding='utf-8', errors='replace')
    (run_dir/'prediction.log').write_text(proc.stdout, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'模型运行失败（退出码 {proc.returncode}），请查看 {run_dir / "prediction.log"}')
    result = evaluate(data/'ground_truth_images.csv', data/'ground_truth_objects.csv',
                      counts_file, pred, run_dir/'evaluation')
    if model == 'baseline3':
        extra = subprocess.run([sys.executable, str(ROOT/'p08_object_observations.py'),
                                '--dataset', str(data), '--predictions-dir', str(pred),
                                '--output', str(run_dir/'evaluation'/'object_observations.json')],
                               cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                               encoding='utf-8', errors='replace')
        (run_dir/'object_observations.log').write_text(extra.stdout, encoding='utf-8')
        if extra.returncode:
            print('提示：包含 REVIEW 的检测框统计未完成，详见 object_observations.log。', flush=True)
    print(f'完成：{model}，{result["scored_image_cases"]} 张图片参与 PASS/FAIL 评分。', flush=True)
    print(f'结果保存在：{run_dir}', flush=True)
    return run_dir


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', help='Explicit legacy dataset; otherwise use frozen Stage2 config')
    parser.add_argument('--model', choices=('baseline1','baseline2','baseline3','all'), default='all')
    parser.add_argument('--output', default=str(ROOT/'output'/'p08_runs'))
    parser.add_argument('--evaluation-config',type=Path,help='Explicit sealed Stage 2/Stage 3 input manifest')
    parser.add_argument('--config',type=Path,help='Explicit profile; default config/p07.json resolves the frozen production registry')
    parser.add_argument('--record-timing',action='store_true',help='Record offline processing timing separately from live latency')
    parser.add_argument('--check-data-only',action='store_true',help='Validate manifests only; no model, prediction or evaluation')
    args = parser.parse_args()
    if not args.dataset and not args.evaluation_config:
        args.evaluation_config=ROOT/'datasets/mechanical_components_v2/configs/stage2_evaluation.json'
    models = ('baseline1','baseline2','baseline3') if args.model == 'all' else (args.model,)
    try:
        if args.check_data_only:
            if not args.evaluation_config:raise ValueError('--check-data-only requires --evaluation-config')
            _,checked=evaluation_inputs(args.evaluation_config,check_only=True)
            print(json.dumps(checked,ensure_ascii=False,indent=2));return 0
        completed = {model: run(args.dataset, model, args.output,args.evaluation_config,args.config,args.record_timing) for model in models}
        if len(completed) == 3:
            compare(completed, Path(args.output))
    except (OSError, ValueError, RuntimeError, KeyError) as e:
        print(f'运行失败：{e}', file=sys.stderr)
        return 1
    return 0


def compare(completed, output):
    rows = {}
    for model, directory in completed.items():
        with (directory/'evaluation'/'per_image.csv').open(encoding='utf-8', newline='') as f:
            rows[model] = {r['image_id']: r for r in csv.DictReader(f)}
    common = sorted(i for i in rows['baseline1'] if all(
        i in rows[m] and rows[m][i]['ground_truth_label'] in ('PASS', 'FAIL')
        and rows[m][i]['prediction_status'] in ('PASS', 'FAIL') for m in completed))
    comparison = {'note': '共同可判定子集；REVIEW不算PASS/FAIL，结果不能代表全体。',
                  'common_images': common, 'common_count': len(common), 'models': {}}
    for model in completed:
        cases = [rows[model][i] for i in common]
        comparison['models'][model] = {
            'correct': sum(r['ground_truth_label'] == r['predicted_label'] for r in cases),
            'false_pass': sum(r['ground_truth_label'] == 'FAIL' and r['predicted_label'] == 'PASS' for r in cases),
            'false_fail': sum(r['ground_truth_label'] == 'PASS' and r['predicted_label'] == 'FAIL' for r in cases),
            'run_dir': str(completed[model])}
    target = output / f'共同对比-{datetime.now():%Y%m%d-%H%M%S-%f}.json'
    target.write_text(json.dumps(comparison, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(f'共同可判定图片：{len(common)} 张。对比文件：{target}', flush=True)


if __name__ == '__main__':
    raise SystemExit(main())
