"""Run one frozen project model and evaluate its outputs on a separate test set."""
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from src.evaluator import evaluate

ROOT = Path(__file__).resolve().parent


def run(dataset, model, results_root):
    data = Path(dataset).resolve()
    for name in ('ground_truth_images.csv', 'ground_truth_objects.csv', 'expected_counts.json'):
        if not (data/name).is_file():
            raise ValueError(f'测试集缺少文件: {data/name}')
    if not (data/'images').is_dir():
        raise ValueError(f'测试集缺少图片文件夹: {data / "images"}')
    results_root = Path(results_root).resolve()
    results_root.mkdir(parents=True, exist_ok=True)
    run_dir = results_root / f'{model}-{datetime.now():%Y%m%d-%H%M%S-%f}'
    run_dir.mkdir()
    pred = run_dir/'predictions'
    command = [sys.executable, str(ROOT/'predict.py'), '--input_dir', str(data/'images'),
               '--model', model, '--output_dir', str(pred), '--counts_file', str(data/'expected_counts.json'),
               '--config', str(ROOT/'config'/'p07.json')]
    # Explicit UTF-8 prevents Windows console encoding from corrupting saved logs.
    with (run_dir/'command.json').open('w', encoding='utf-8') as f:
        json.dump({'command': command, 'cwd': str(ROOT), 'dataset': str(data),
                   'baseline3_mode': 'size_verified' if model == 'baseline3' else None}, f, ensure_ascii=False, indent=2)
    print(f'正在运行 {model}，请稍候……', flush=True)
    proc = subprocess.run(command, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          encoding='utf-8', errors='replace')
    (run_dir/'prediction.log').write_text(proc.stdout, encoding='utf-8')
    if proc.returncode:
        raise RuntimeError(f'模型运行失败（退出码 {proc.returncode}），请查看 {run_dir / "prediction.log"}')
    result = evaluate(data/'ground_truth_images.csv', data/'ground_truth_objects.csv',
                      data/'expected_counts.json', pred, run_dir/'evaluation')
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
    parser.add_argument('--dataset', default=str(ROOT/'data'/'test_20261124'))
    parser.add_argument('--model', choices=('baseline1','baseline2','baseline3','all'), default='all')
    parser.add_argument('--output', default=str(ROOT/'output'/'p08_runs'))
    args = parser.parse_args()
    models = ('baseline1','baseline2','baseline3') if args.model == 'all' else (args.model,)
    try:
        completed = {model: run(args.dataset, model, args.output) for model in models}
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
