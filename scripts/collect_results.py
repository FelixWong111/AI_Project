"""Publish existing experiment evidence only; never import models or open images.

Run once with ``python -m scripts.collect_results``. Use ``--verify`` afterwards.
The archive records original and published hashes and removes machine-local paths.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / 'results'
B12 = 'outputs/acceptance/1795511987719847700'
B3 = 'outputs/p06_merge/1795511737509054100'
WO = 'outputs/work_orders_acceptance/1795512320558820300'
AUDIT = 'outputs/leakage_audit/1795502444356390600'
TEXT = {'.json', '.csv', '.log', '.txt', '.md'}


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read(name):
    return json.loads((ROOT / name).read_text(encoding='utf-8-sig'))


def encoded(value):
    return (json.dumps(value, ensure_ascii=False, indent=2) + '\n').encode('utf-8')


def portable(text):
    for prefix in (str(ROOT), ROOT.as_posix()):
        text = text.replace(prefix, '${PROJECT_ROOT}')
    # Covers historic records containing an old/garbled Windows profile name too.
    text = re.sub(r'(?i)[A-Z]:[\\/]+Users[\\/]+[^"\r\n]*?[\\/]+Project(?=[\\/])',
                  '${PROJECT_ROOT}', text)
    text = re.sub(r'(?i)[A-Z]:[\\/]+01 Project[\\/]+Code[\\/]+yolo_module[\\/]+AI_Project',
                  '${P06_PROJECT_ROOT}', text)
    text = re.sub(r'(?i)[A-Z]:[\\/]+Users[\\/]+[^\\/"\r\n]+', '${USERPROFILE}', text)
    return text


def portable_value(value):
    if isinstance(value, str):
        return portable(value)
    if isinstance(value, list):
        return [portable_value(v) for v in value]
    if isinstance(value, dict):
        return {portable(k): portable_value(v) for k, v in value.items()}
    return value  # All numbers, labels, scores and booleans remain unchanged.


def publish_bytes(path):
    original = path.read_bytes()
    if path.suffix == '.json':
        value = json.loads(original.decode('utf-8-sig'))
        sanitized = portable_value(value)
        return (original, False) if sanitized == value else (encoded(sanitized), True)
    value = original.decode('utf-8-sig')
    sanitized = portable(value)
    return (original, False) if sanitized == value else (sanitized.encode('utf-8'), True)


def protected_state():
    files = [p for folder in ('models', 'config') for p in (ROOT / folder).rglob('*') if p.is_file()]
    data = {}
    for folder in ROOT.iterdir():
        if folder.is_dir() and (folder.name.startswith('P03') or folder.name in ('data', '开发', '验证', '测试')):
            for path in folder.rglob('*'):
                if path.is_file():
                    stat = path.stat()
                    data[path.relative_to(ROOT).as_posix()] = (stat.st_size, stat.st_mtime_ns)
    return {'models_and_configs': {p.relative_to(ROOT).as_posix(): sha(p) for p in files},
            'raw_data_metadata': data}


def verify(source_files=True):
    manifest = read('results/file_manifest.json')
    for item in manifest['files']:
        path = ROOT / item['path']
        assert path.stat().st_size == item['size_bytes'], item['path']
        assert sha(path) == item['sha256'], item['path']
        if source_files:
            for source in item['sources']:
                assert sha(ROOT / source['path']) == source['sha256'], source['path']
    files = [p for p in DEST.rglob('*') if p.is_file()]
    groups = defaultdict(list)
    sensitive = []
    patterns = [r'-----BEGIN (?:RSA |OPENSSH |EC )?PRIVATE KEY-----',
                r'\b(?:ghp_|github_pat_|sk-proj-)[A-Za-z0-9_\-]{20,}',
                r'\bAKIA[A-Z0-9]{16}\b', r'(?i)[A-Z]:[\\/]+Users[\\/]',
                r'(?i)https?://[^\s/]+:[^\s/@]+@',
                r'(?i)(?:api[_-]?key|password|access[_-]?token)\s*[=:]\s*[\"\x27][^\"\x27\s]{8,}']
    for p in files:
        assert p.suffix in TEXT, p
        assert p.stat().st_size < 1024 * 1024, p
        content = p.read_text(encoding='utf-8-sig')
        for index, pattern in enumerate(patterns):
            if re.search(pattern, content):
                sensitive.append({'path': p.relative_to(ROOT).as_posix(), 'rule': index})
        if p.suffix == '.json':
            json.loads(content)
        groups[sha(p)].append(p.relative_to(ROOT).as_posix())
    duplicates = [paths for paths in groups.values() if len(paths) > 1]
    assert not duplicates, duplicates
    assert not sensitive, sensitive
    ignored = subprocess.run(['git', 'check-ignore', '--no-index', 'results/README.md'], cwd=ROOT,
                             capture_output=True, text=True)
    assert ignored.returncode == 1, 'results is ignored'
    report = {'status': 'PASS', 'files': len(files),
                      'total_bytes': sum(p.stat().st_size for p in files),
                      'largest': max((p.stat().st_size, p.relative_to(ROOT).as_posix()) for p in files),
                      'duplicate_groups': duplicates, 'sensitive_findings': sensitive,
                      'local_source_files_checked': source_files,
                      'official_held_out_test': 'NOT_RUN'}
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def collect():
    if DEST.exists() and any(DEST.iterdir()):
        raise FileExistsError('results already exists; use --verify. Never overwrite a published archive.')
    before = protected_state()
    outputs = sorted(p for p in (ROOT / 'outputs').rglob('*') if p.is_file())
    output_hashes = {p.relative_to(ROOT).as_posix(): sha(p) for p in outputs}
    versions = {n: read(f'models/{n}/training_record.json')['model_version'] for n in ('baseline1', 'baseline2')}
    versions['baseline3'] = read('models/baseline3/provenance.json')['weights_sha256']
    for n in ('baseline1', 'baseline2'):
        record = read(f'outputs/training/{n}/training_record.json')
        assert record['model_version'] == versions[n], 'Output and installed model versions differ'
        for filename, expected in record['files'].items():
            assert sha(ROOT / 'models' / n / filename) == expected['sha256'], filename
    assert sha(ROOT / 'models/baseline3/best.pt') == versions['baseline3']
    assert read(B3 + '/acceptance.json')['metadata']['weights_sha256'] == versions['baseline3']
    selected = []

    def select(source, destination, purpose, models='', scope='Validation'):
        assert (ROOT / source).is_file(), source
        selected.append((source, destination, purpose, models, scope))

    select('outputs/training/data/manifest.json', 'data_manifests/development.json',
           'Actual shared training/validation manifest with labels, counts, SHA256 and pixel hashes', 'baseline1,baseline2', 'Training+Validation')
    for n in ('baseline1', 'baseline2'):
        select(f'outputs/training/{n}/training_record.json', f'{n}/training_record.json',
               'Selected parameters, all trial metrics, confusion matrices, environment and freeze hashes', n, 'Training+Validation')
        for split in ('train', 'validation'):
            select(f'outputs/training/{n}/{split}_predictions.csv', f'{n}/{split}_predictions.csv',
                   'Original per-image truth, prediction and P(FAIL)', n, 'Training' if split == 'train' else 'Validation')
        feature = next((ROOT / 'outputs/training/features').glob(n + '-*.json'))
        select(feature.relative_to(ROOT).as_posix(), f'data_manifests/feature_provenance/{n}.json',
               'Feature-cache provenance only; no cached feature tensors', n, 'Training+Validation')
    for name in ('train', 'validation'):
        select(f'outputs/work_orders/{name}.json', f'data_manifests/work_orders/{name}.json',
               'Fixed work orders, NOT ground truth', 'baseline1,baseline2,baseline3', 'Training' if name == 'train' else 'Validation')
    for p in sorted((ROOT / 'outputs/work_orders/functional').glob('*.json')):
        select(p.relative_to(ROOT).as_posix(), 'data_manifests/work_orders/functional/' + p.name,
               'Variable-count functional cases; excluded from common performance metrics', 'baseline1,baseline2,baseline3', 'Functional')
    for name in ('diagnosis.json', 'training_diagnostics.json'):
        select('outputs/baseline1_diagnosis/' + name, 'baseline1/diagnostics/' + name,
               'Training-chain/OOD/scene-guard diagnostics, NOT independent evaluation', 'baseline1', 'Training+Synthetic diagnostic')
    select('models/baseline3/training_record.json', 'baseline3/training_record.json',
           'Historical parameters and epoch curves extracted from frozen checkpoint, not rerun', 'baseline3', 'Historical training/validation')
    select('models/baseline3/provenance.json', 'baseline3/provenance.json',
           'Weight identity, frozen settings and missing P06 deliverables', 'baseline3', 'Provenance')
    for p in sorted((ROOT / B3 / 'batch-baseline3').iterdir()):
        if p.suffix in ('.csv', '.json'):
            select(p.relative_to(ROOT).as_posix(), 'baseline3/validation_inference/' + p.name,
                   'Existing real CUDA validation inference, not a new GT-based performance evaluation', 'baseline3')
    for p in sorted((ROOT / AUDIT).glob('*.json')):
        select(p.relative_to(ROOT).as_posix(), 'audits/leakage/' + p.name,
               'Historical leakage/source/probability audit of these frozen classifier weights', 'baseline1,baseline2', 'Training+Validation audit')
    select(B12 + '/acceptance.json', 'integration/baseline12/acceptance.json',
           'Latest completed 46-image reload and deployment consistency regression', 'baseline1,baseline2')
    select('outputs/acceptance/1795502286780346400/acceptance.json',
           'audits/leakage/original_integration_acceptance.json',
           'Distinct original acceptance cited by the historical leakage audit', 'baseline1,baseline2')
    for n in ('baseline1', 'baseline2'):
        for name in ('images.csv', 'run_config.json'):
            select(f'{B12}/{n}/{name}', f'integration/baseline12/{n}/{name}',
                   'Latest real batch regression predictions and run configuration', n)
    for name in ('acceptance.json', 'worker.json'):
        select(B3 + '/' + name, 'integration/baseline3/' + name,
               'Real CUDA worker, rejudge, ID0, batch and GUI controller acceptance', 'baseline3')
    for p in sorted((ROOT / B3 / 'controller').rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.csv'):
            select(p.relative_to(ROOT).as_posix(), 'integration/baseline3/controller/' + p.relative_to(ROOT / B3 / 'controller').as_posix(),
                   'Actual three-model controller exports; native windows isolated, no manual/camera claim', 'baseline1,baseline2,baseline3', 'Integration')
    for name in ('environment.json', 'dependency-review.json', 'conflict-resolution.json', 'final-state.json',
                 'source-head.txt', 'pip-freeze-before.txt', 'unit-tests-final.log'):
        select('outputs/p06_merge/' + name, 'integration/p06_merge/' + name,
               'Completed merge, dependency, environment or final automated-test evidence', 'baseline1,baseline2,baseline3', 'Integration')
    select('docs/p06/integration-requirements-lock.txt', 'integration/environment-after.txt',
           'Observed CUDA/YOLO environment after integration', 'baseline1,baseline2,baseline3', 'Environment')
    # Preserve the latest complete work-order run, including original commands and CSVs.
    # Raw images, video, feature arrays, cache, session JSONL and older repeat runs stay local.
    for p in sorted((ROOT / WO).rglob('*')):
        if p.is_file() and p.suffix in ('.json', '.csv', '.log'):
            select(p.relative_to(ROOT).as_posix(), 'integration/work_orders/' + p.relative_to(ROOT / WO).as_posix(),
                   'Latest completed U01-U09 functional acceptance, not performance samples', 'baseline1,baseline2', 'Functional')

    files, published, source_map = [], {}, {}

    def store(data, destination, source=None, purpose='', models='', scope='', normalized=False):
        digest = hashlib.sha256(data).hexdigest()
        if digest in published:
            item = published[digest]
        else:
            path = DEST / destination
            assert path.resolve().is_relative_to(DEST.resolve()), destination
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
            item = dict(path=path.relative_to(ROOT).as_posix(), size_bytes=len(data), sha256=digest,
                        purpose=purpose, models=models.split(',') if models else [], scope=scope,
                        machine_paths_normalized=normalized, sources=[])
            published[digest] = item
            files.append(item)
        if source:
            item['sources'].append(dict(path=source, sha256=sha(ROOT / source)))
            source_map[source] = item['path']
        return item['path']

    for source, destination, purpose, models, scope in selected:
        data, normalized = publish_bytes(ROOT / source)
        store(data, destination, source, purpose, models, scope, normalized)

    # Derived indexes only: extract stored metrics/errors; do not run a classifier or evaluator.
    completeness = {}
    for n in ('baseline1', 'baseline2'):
        record = read(f'outputs/training/{n}/training_record.json')
        data = dict(model=n, model_version=versions[n], source=f'results/{n}/training_record.json',
                    score_definition='p_fail=P(FAIL), uncalibrated; raw frozen classifier, not scene-guard system',
                    metrics=record['metrics'], official_held_out_test='NOT_RUN')
        store(encoded(data), n + '/metrics.json', purpose='Exact extraction of existing recorded metrics', models=n, scope='Training+Validation')
        stream = io.StringIO(newline='')
        writer = csv.DictWriter(stream, fieldnames=['split', 'error_type', 'image_id', 'true_label', 'predicted_label', 'p_fail'], lineterminator='\n')
        writer.writeheader()
        for split in ('train', 'validation'):
            with (ROOT / f'outputs/training/{n}/{split}_predictions.csv').open(encoding='utf-8-sig', newline='') as f:
                rows = list(csv.DictReader(f))
            original = record['metrics'][split]
            errors = [r for r in rows if r['true_label'] != r['predicted_label']]
            assert len(rows) == (262 if split == 'train' else 46)
            assert len(errors) == original['false_PASS'] + original['false_FAIL']
            for row in errors:
                writer.writerow(dict(split=split, error_type='False PASS' if row['predicted_label'] == 'PASS' else 'False FAIL', **row))
        store(stream.getvalue().encode('utf-8'), n + '/errors.csv', purpose='Errors filtered from existing predictions; header only means zero errors', models=n, scope='Training+Validation')
        completeness[n] = dict(model_version=versions[n], training_images=262, validation_images=46,
                               training_predictions='PRESENT', validation_predictions='PRESENT',
                               truth_scores_metrics_confusion_errors_trials='PRESENT', official_test='NOT_RUN')
    completeness['baseline3'] = dict(weights_sha256=versions['baseline3'], validation_inference_images=46,
        historical_checkpoint_training_record='PRESENT',
        actual_training_validation_manifests='MISSING_FROM_P06', training_per_image_predictions='MISSING_FROM_P06',
        validated_GT_comparison_confusion_and_errors='NOT_PRODUCED_IN_THIS_INTEGRATION',
        referenced_post_GT_fix_and_resolution_reports='MISSING_FROM_P06',
        image_level_score='NOT_APPLICABLE: null by count-rule contract; object scores are in objects.csv',
        official_test='NOT_RUN')
    store(encoded(completeness), 'completeness.json', purpose='Evidence availability, no invented results', scope='Archive index')

    # Inventory every original output, including excluded binary/duplicate/unfinished artifacts.
    by_top = defaultdict(lambda: dict(files=0, bytes=0, extensions=Counter()))
    inventory = io.StringIO(newline='')
    fields = ['source_path', 'size_bytes', 'sha256', 'purpose', 'model_versions', 'archive_path', 'decision']
    writer = csv.DictWriter(inventory, fieldnames=fields, lineterminator='\n'); writer.writeheader()
    same_source_hash = {s['sha256']: item['path'] for item in files for s in item['sources']}
    for p in outputs:
        source = p.relative_to(ROOT).as_posix(); rel = p.relative_to(ROOT / 'outputs')
        top = rel.parts[0]; stat = p.stat(); group = by_top[top]
        group['files'] += 1; group['bytes'] += stat.st_size; group['extensions'][p.suffix.lower()] += 1
        matches = [n for n in versions if n in source]
        if not matches and top in ('training', 'leakage_audit', 'acceptance', 'work_orders_acceptance'):
            matches = ['baseline1', 'baseline2']
        if not matches and top == 'p06_merge': matches = list(versions)
        archive = source_map.get(source) or same_source_hash.get(output_hashes[source], '')
        if source in source_map: decision = 'SELECTED'
        elif archive: decision = 'DUPLICATE_BYTES_ALREADY_PUBLISHED'
        elif p.suffix.lower() not in TEXT: decision = 'EXCLUDED_MEDIA_BACKUP_FEATURE_ARRAY_OR_SESSION_STREAM'
        elif 'cache' in rel.parts: decision = 'EXCLUDED_RUNTIME_CACHE'
        else: decision = 'EXCLUDED_OLDER_INCOMPLETE_REPEAT_OR_LOCAL_ONLY_RECORD'
        purpose = ('training / data / feature provenance' if top == 'training' else
                   'leakage audit' if top == 'leakage_audit' else
                   'system merge / CUDA / controller acceptance' if top == 'p06_merge' else
                   'work-order functional cases/acceptance' if top.startswith('work_orders') else
                   'training-chain and OOD diagnostics' if top == 'baseline1_diagnosis' else
                   'local backup' if top.startswith('refactor_backup') else 'inference / GUI / regression record')
        writer.writerow(dict(source_path=source, size_bytes=stat.st_size, sha256=output_hashes[source],
                             purpose=purpose, model_versions=';'.join(n + ':' + versions[n] for n in matches) or 'NOT_APPLICABLE_OR_UNKNOWN',
                             archive_path=archive, decision=decision))
    store(inventory.getvalue().encode('utf-8'), 'audits/archive/outputs_inventory.csv', purpose='Every output file, size, hash, inferred purpose/version and publication decision', scope='Inventory')
    store(encoded(dict(total_files=len(outputs), total_bytes=sum(p.stat().st_size for p in outputs),
                       by_directory=dict(by_top), model_version_note='Path/category association only; mixed logs may cover multiple models. Individual run metadata remains authoritative.')),
          'audits/archive/inventory_summary.json', purpose='Complete local outputs inventory totals', scope='Inventory')
    store(encoded(dict(models=versions,
                       model_identifiers={**{n: versions[n] for n in ('baseline1', 'baseline2')},
                                          'baseline3': read(B3 + '/acceptance.json')['metadata']['model_version']},
                       data_manifest_record_sha256=read('outputs/training/data/manifest.json')['evidence']['manifest_sha256'],
                       source_git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                       fixed_work_order={str(k): 1 for k in range(1, 6)}, official_held_out_test='NOT_RUN',
                       collection='Only existing evidence copied/extracted; no model imports, image decoding, training, inference or new evaluation.',
                       portable_paths='${PROJECT_ROOT}, ${P06_PROJECT_ROOT}, ${USERPROFILE} replace personal machine prefixes only; numeric values unchanged.',
                       source_to_archive=source_map)),
          'catalog.json', purpose='Versions and canonical original-to-published path map', scope='Archive index')

    after = protected_state()
    after_outputs = {p.relative_to(ROOT).as_posix(): sha(p) for p in (ROOT / 'outputs').rglob('*') if p.is_file()}
    assert before == after, 'Protected models/configs/raw-data metadata changed'
    assert output_hashes == after_outputs, 'Original outputs changed'
    store(encoded(dict(original_outputs_count=len(output_hashes), original_outputs_sha256_unchanged=True,
                       models_and_configs_sha256_unchanged=True, protected_model_config_hashes=before['models_and_configs'],
                       raw_data_file_count=len(before['raw_data_metadata']), raw_data_paths_sizes_mtimes_unchanged=True,
                       raw_data_content_not_opened=True, official_held_out_test='NOT_RUN')),
          'audits/archive/source_integrity.json', purpose='Collection before/after protection checks', scope='Integrity')
    # Manifest excludes itself (no circular self-hash). README/tree are human indexes.
    (DEST / 'README.md').write_text(README, encoding='utf-8')

    def indexes():
        (DEST / 'file_manifest.json').write_bytes(encoded(dict(files=files, source_to_archive=source_map,
            excludes_self_and_human_indexes=True, duplicate_content_stored_once=True,
            transformations='Personal machine prefixes only when present; otherwise exact bytes. Derived metrics/errors explicitly labelled.')))
        paths = {p.relative_to(ROOT).as_posix() for p in DEST.rglob('*') if p.is_file()}
        paths.add('results/DIRECTORY_TREE.txt')
        (DEST / 'DIRECTORY_TREE.txt').write_text('\n'.join(sorted(paths)) + '\n', encoding='utf-8')

    indexes()
    report = verify()
    store(encoded(dict(scope='Checks on archive before this report was added; final counts available via --verify',
                       checks=report, credential_scan='Known token/private-key/credential-URL/personal-profile patterns; not a guarantee against all possible sensitive content.')),
          'audits/archive/publish_check.json', purpose='Actual archive hash/size/deduplication/sensitive-pattern verification', scope='Integrity')
    indexes()
    verify()


README = '''# 正式实验结果索引

本目录通过 Git 交付现有冻结模型的真实证据。来源仍保存在本机 outputs/，没有删除或覆盖；本轮只整理文件、提取已有指标/误判行，没有训练、推理或新增性能评估。正式 **Test: NOT_RUN**。

## P05 / P08 查阅顺序

1. 打开 [catalog.json](catalog.json) 核对当前模型版本，再看 [completeness.json](completeness.json) 了解证据缺项。
2. P05：重点看 [Baseline2 训练记录](baseline2/training_record.json)、[逐图验证预测](baseline2/validation_predictions.csv)、[指标](baseline2/metrics.json)、[误判清单](baseline2/errors.csv)。p_fail 是 P(FAIL)，不是永远代表预测类别的置信度；分数未校准。
3. P08：先用 [统一开发清单](data_manifests/development.json) 核对分区、真值和图片/像素哈希，再看各模型逐图结果、[泄漏审计](audits/leakage/audit.json) 与 integration/ 中的验收。不要将功能测试、Training 拟合成绩、Validation 选择成绩混进 Test 指标。
4. Baseline3 的 [验证推理输出](baseline3/validation_inference/) 包含 images/counts/objects/run_config；对象有置信度，整图计数规则没有概率。它们是已有真实推理记录，不是新的目标级真值评估报告。checkpoint 历史指标也不是本轮复验成绩。
5. [完整文件树](DIRECTORY_TREE.txt)、[逐文件来源与SHA256](file_manifest.json)、[全部 outputs 盘点](audits/archive/outputs_inventory.csv) 支持复核来源与去重决定。

## 目录用途与版本

| 目录 | 内容 | 版本依据 |
|---|---|---|
| baseline1/ | 原始 Training/Validation 预测、训练记录、指标、误判、独立场景门控诊断 | joblib SHA256 67f99643…d01；门控与原始分类器指标分开 |
| baseline2/ | 原始 Training/Validation 预测、LR候选、指标和误判 | model_version 0d6965ae…e1；JSON SHA256 64102315…e55 |
| baseline3/ | P06 checkpoint 训练曲线、来源、现有46张Val真实CUDA推理 | best.pt SHA256 4c5a1f75…b3d |
| data_manifests/ | 两分类器实际训练清单、缓存来源元数据、固定/特殊工单 | manifest records SHA256 44498f23…027；不冒充P06实际训练清单 |
| audits/leakage/ | 历史泄漏审计、逐图三方比较、哈希 | 指向当前相同分类器权重；源码指纹是当时版本 |
| audits/archive/ | outputs全量数量/大小/用途/版本归属/排除原因及保护检查 | 本次只读整理快照 |
| integration/ | CUDA环境、三模型控制器/批处理验收、最新完成工单验收、单元测试日志 | 各run_config和acceptance中的真实版本 |

Baseline1/2 各262张训练、46张验证：训练准确率均100%；验证准确率分别100%、80.4348%。Baseline2验证 False PASS=6、False FAIL=3；Baseline1为0/0，空 errors.csv 只有表头表示真实零误判，不补造样本。矩阵顺序PASS/FAIL、FAIL为positive class。验证集曾用于有限候选选择，不能代替独立Test。

场景独立性沿用已有人工确认；原始 scene_id=TBD 未填造，哈希无重复不证明物理场景独立。Ground Truth 是实际数量/真实标签；work_orders 是要求，不由GT反推。functional 工单只能独立运行，不计共同性能指标。

## 缺项与未执行项

- Baseline3未收到实际训练/验证图片清单、训练逐图预测、GT修正后复验及640/960对照报告；没有补造这些记录，也不把P07共用开发清单当作P06训练证据。
- Baseline3现有验证推理没有另行生成GT混淆矩阵/False PASS/False FAIL或框级指标。本轮归档不新增评估，不将13个预测ID0均触发FAIL解释为未知件识别准确率。
- 现场人工GUI点击、真实相机和20新帧延迟未验收；Final未配置；正式Test未运行。尺度标定、尺寸容差、错规格与未见目录外件能力缺口见 [功能审查](../docs/p06/compliance_review.md)。
- 没有原始图片/标注文件、模型权重、NPZ特征缓存、备份ZIP、视频或截图。图像级标签和数量作为已批准的Training/Validation审计清单保留；不含正式Test真值。单元日志中的STUB/MOCK是明确的测试夹具，不是生产模型或正式性能证据。

## 来源、去重与可移植路径

同内容只保留一份，file_manifest.json 和 catalog.json 将多个原来源映射到同一归档文件。完整记录保留原始实际命令、退出码和运行标识；其中 outputs/ 路径是历史执行位置，通过 source_to_archive 查归档位置，不要求GitHub克隆中存在 outputs/。旧实验的本机路径前缀统一为 `${PROJECT_ROOT}` / `${P06_PROJECT_ROOT}` / `${USERPROFILE}`，不发布个人用户名；数值、标签、参数、模型版本和原始文件名不变。manifest同时记录源文件和公开副本SHA，路径规范化后的文件SHA可能不同。

.gitattributes 对 results 禁用换行自动转换，确保 Windows/Linux 克隆得到相同的证据文件字节与SHA256。完整 outputs 盘点仅按路径/目录推断混合日志所属模型；精确版本以各run_config、训练记录和catalog为准。

只验证已归档文件，不调用任何模型：

```powershell
.\\.venv311\\Scripts\\python.exe -m scripts.collect_results --verify
```

此验证在原工作区还会比对本机来源；新克隆缺少outputs时，使用下述标准库命令仅核对Git内交付文件，不需要安装推理依赖：

```powershell
python -m scripts.collect_results --verify --archive-only
```

复跑推理/审计仍需要各成员自行取得经过批准的Train/Validation原图；Git提供复核报告，不包含原始数据。原始模型仍从 models/ 加载。实验生成脚本继续写 outputs/，本任务没有修改系统集成逻辑。新实验应使用新输出目录和新归档版本，不能覆盖本次冻结证据。
'''


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify', action='store_true', help='Verify archive and original source hashes, no model execution')
    parser.add_argument('--archive-only', action='store_true', help='With --verify, verify the Git archive without local outputs sources')
    args = parser.parse_args()
    if args.archive_only and not args.verify:
        parser.error('--archive-only requires --verify')
    verify(source_files=not args.archive_only) if args.verify else collect()
