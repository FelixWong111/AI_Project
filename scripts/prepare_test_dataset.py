"""Create a filename-based evaluation package without touching the P03 source."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from PIL import Image, ImageOps

from src.data.development import ROOT, digest, read_csv, write_json
from src.work_orders import load_cases, reference_counts

CLASSES = ('non_catalogue', 'long_screw', 'thick_screw', 'thin_screw', 'hex_nut', 'flat_washer')
COUNT_FIELDS = tuple(name + '_count' for name in CLASSES)


def discover_source():
    candidates = [p.parent for package in ROOT.glob('P03_*')
                  for p in package.rglob('ground_truth_images.csv')
                  if any(part.startswith('P03_测试集_新版20261124') for part in p.parts)]
    if len(candidates) != 1:
        raise ValueError('Specify --source: expected exactly one new P03 test package')
    return candidates[0]


def tree_hashes(directory):
    return {p.relative_to(directory).as_posix(): digest(p)
            for p in sorted(directory.rglob('*')) if p.is_file()}


def validate_source(source):
    images = read_csv(source / 'ground_truth_images.csv')
    objects = read_csv(source / 'ground_truth_objects.csv')
    by_id = {row['image_id']: row for row in images}
    if not images or len(by_id) != len(images):
        raise ValueError('Empty manifest or duplicate image_id')
    if any(row['split'].lower() != 'test' for row in images):
        raise ValueError('Only test rows are allowed in this package')
    names = (source / 'classes.txt').read_text(encoding='utf-8-sig').splitlines()
    if tuple(names) != CLASSES:
        raise ValueError('Project class IDs must remain 0 through 5')
    image_dir = source / 'images/test'
    if {p.name for p in image_dir.glob('*.jpg')} != set(by_id):
        raise ValueError('Image directory and manifest differ')
    by_object, seen = defaultdict(list), set()
    for obj in objects:
        identity = (obj['image_id'], obj['object_id'])
        if identity in seen or identity[0] not in by_id or not identity[1]:
            raise ValueError(f'Invalid object identity: {identity}')
        seen.add(identity)
        by_object[identity[0]].append(obj)
    byte_hashes, pixel_hashes = set(), set()
    for index, row in enumerate(images):
        iid = row['image_id']
        if Path(iid).name != iid or not iid.endswith('.jpg'):
            raise ValueError('image_id must be the actual JPG filename')
        path = image_dir / iid
        with Image.open(path) as im:
            rgb = ImageOps.exif_transpose(im).convert('RGB')
            width, height = rgb.size
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
        sha = digest(path)
        if sha in byte_hashes or pixels in pixel_hashes:
            raise ValueError(f'Duplicate test photo: {iid}')
        byte_hashes.add(sha)
        pixel_hashes.add(pixels)
        if (width, height) != (int(row['width']), int(row['height'])):
            raise ValueError(f'Dimension mismatch: {iid}')
        counts, boxes = Counter(), []
        for obj in by_object[iid]:
            cid = int(obj['class_id'])
            x1, y1, x2, y2 = [float(obj[k]) for k in ('xmin', 'ymin', 'xmax', 'ymax')]
            if cid not in range(6) or not 0 <= x1 < x2 <= width or not 0 <= y1 < y2 <= height:
                raise ValueError(f'Invalid class/box: {iid}')
            if obj['ignore'] not in ('0', '1'):
                raise ValueError(f'Invalid ignore flag: {iid}')
            if obj['ignore'] == '0':
                counts[cid] += 1
            boxes.append([cid, x1, y1, x2, y2])
        if [int(row[f]) for f in COUNT_FIELDS] != [counts[c] for c in range(6)]:
            raise ValueError(f'Object counts differ from image counts: {iid}')
        label = row['reference_label_all_one']
        if label not in ('PASS', 'FAIL', 'REVIEW'):
            raise ValueError(f'Unknown reference label: {iid}')
        derived = 'PASS' if counts[0] == 0 and all(counts[c] == 1 for c in range(1, 6)) else 'FAIL'
        if label != 'REVIEW' and label != derived:
            raise ValueError(f'Label/count conflict: {iid}')
        labelme = json.loads(path.with_suffix('.json').read_text(encoding='utf-8-sig'))
        if (labelme['imageWidth'], labelme['imageHeight']) != (width, height) or labelme['imagePath'] != iid:
            raise ValueError(f'Labelme image reference differs: {iid}')
        json_boxes = []
        for shape in labelme['shapes']:
            points = np.asarray(shape['points'])
            json_boxes.append([CLASSES.index(shape['label']), *points.min(0), *points.max(0)])
        yolo_boxes = []
        for line in (source / 'labels/test' / path.with_suffix('.txt').name).read_text().splitlines():
            cid, x, y, w, h = map(float, line.split())
            if not cid.is_integer():
                raise ValueError('YOLO class ID must be integer')
            yolo_boxes.append([int(cid), (x-w/2)*width, (y-h/2)*height, (x+w/2)*width, (y+h/2)*height])
        for other, tolerance in ((json_boxes, .001), (yolo_boxes, .02)):
            if len(other) != len(boxes) or not np.allclose(sorted(other), sorted(boxes), atol=tolerance):
                raise ValueError(f'Annotation formats disagree: {iid}')
        if (index + 1) % 12 == 0:
            print(f'Validated {index + 1}/{len(images)} photos', flush=True)
    return sorted(images, key=lambda r: r['image_id']), objects


def prepare(source, destination):
    source, destination = source.resolve(), destination.resolve()
    if destination.exists():
        raise FileExistsError('Destination exists; use a new directory, never overwrite a test version')
    if destination.is_relative_to(source) or not destination.is_relative_to(ROOT):
        raise ValueError('Destination must be outside P03 and inside the project')
    if any(part.startswith('P03_') for part in destination.relative_to(ROOT).parts):
        raise ValueError('P03 must remain read-only')
    original_hashes = tree_hashes(source)
    images, objects = validate_source(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix='.prepare-test-', dir=destination.parent) as temp:
        staging = Path(temp) / 'package'
        (staging / 'images').mkdir(parents=True)
        (staging / 'labels').mkdir()
        for row in images:
            iid = row['image_id']
            for suffix in ('.jpg', '.json'):
                filename = Path(iid).with_suffix(suffix).name
                shutil.copy2(source / 'images/test' / filename, staging / 'images' / filename)
            filename = Path(iid).with_suffix('.txt').name
            shutil.copy2(source / 'labels/test' / filename, staging / 'labels' / filename)
            # This is a user-confirmed identity, not a fabricated capture-record SceneID.
            row['scene_id'] = iid
            if row['reference_label_all_one'] == 'REVIEW':
                row['annotation_status'] = 'REVIEW'
        with (staging / 'ground_truth_images.csv').open('w', encoding='utf-8-sig', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(images[0]))
            writer.writeheader()
            writer.writerows(images)
        for name in ('ground_truth_objects.csv', 'classes.txt'):
            shutil.copy2(source / name, staging / name)
        fixed = reference_counts()
        write_json(staging / 'expected_counts.json', [dict(image_id=r['image_id'], expected_counts=fixed) for r in images])
        assert len(load_cases(staging / 'expected_counts.json')) == len(images)
        labels = dict(Counter(r['reference_label_all_one'] for r in images))
        info = dict(version='test-20261124-clean-v1', purpose='held_out_evaluation_only',
                    model_evaluation='NOT_RUN', images=len(images), scenes=len(images), labels=labels,
                    scene_id_rule='scene_id equals image_id; no separate ID or mapping',
                    scene_confirmation=dict(source='User confirmation in this session, 2026-11-24',
                                            statement='认为所有scene不一样', independently_verified=False),
                    source=source.relative_to(ROOT).as_posix(), source_files=len(original_hashes),
                    source_tree_sha256=hashlib.sha256(json.dumps(original_hashes, sort_keys=True).encode()).hexdigest(),
                    expected_counts=fixed, non_catalogue_expected=0,
                    object_instances=len(objects), class_instances=dict(sorted(Counter(o['class_id'] for o in objects).items())),
                    review_images=[r['image_id'] for r in images if r['reference_label_all_one'] == 'REVIEW'],
                    changes=['Use image filenames everywhere; discard duplicate and alternative-count cases',
                             'scene_id equals filename on explicit user confirmation',
                             'Normalize annotation_status to REVIEW for existing REVIEW image; counts and boxes unchanged'],
                    checks=dict(images_decode=True, no_internal_byte_or_pixel_duplicates=True,
                                labelme_yolo_csv_agree=True, cases_load_with_project_cli=True,
                                source_unchanged=True))
        if tree_hashes(source) != original_hashes:
            raise RuntimeError('Source changed during preparation; refusing to publish')
        write_json(staging / 'dataset_info.json', info)
        readme = f'''# 测试集 test-20261124-clean-v1

仅用于三个模型的共同独立评估，不用于训练、特征拟合或调参。模型评估尚未运行。

36 张图片；按用户本次人工确认计 36 个独立场景。scene_id 直接等于真实图片名，例如 01.jpg；该确认不是由哈希推断，也不是 P2 原始拍摄编号。

固定工单：ID 1–5 各 1，ID 0 恒为 0。真值：19 PASS、16 FAIL、1 REVIEW。34.jpg 为 REVIEW，保留原有数量和5个框，annotation_status 已同步为 REVIEW。PASS/FAIL 指标只计35张，其余单独报告；不要仅凭数量把34.jpg重新派生为PASS。默认共同目标级评估也单独处理该图，不能未经说明把不可判定图混入分母。

## 文件

images/：36张原始JPG及36份同名Labelme JSON，Labelme可直接打开此目录。
labels/：36份同名YOLO TXT；项目类别0=non_catalogue，1–5为五类目标，见classes.txt，禁止自动减1重映射。
ground_truth_images.csv：统一真实文件名、scene_id、split=test、尺寸、状态、数量及参考标签。
ground_truth_objects.csv：170个原始目标框（含REVIEW图5框），框和ignore不改。
expected_counts.json：36条课程格式输入，仅含image_id与expected_counts，无真值或框。
dataset_info.json：来源、用户场景确认、版本和整理核验。
SHA256SUMS.txt：整理后文件校验值，不包含自身。

没有 TEST_01 用例ID、原编号映射、32/38重复用例或额外工单。全部直接使用实际图片名；原图片名保留，不为补齐序号重新命名。

## 后续正式批处理

在项目根目录运行（本轮只整理，不执行这些推理命令）：

    python predict.py --input_dir {destination.relative_to(ROOT).as_posix()}/images --model baseline1 --output_dir outputs/test-baseline1 --counts_file {destination.relative_to(ROOT).as_posix()}/expected_counts.json
    python predict.py --input_dir {destination.relative_to(ROOT).as_posix()}/images --model baseline2 --output_dir outputs/test-baseline2 --counts_file {destination.relative_to(ROOT).as_posix()}/expected_counts.json
    python predict.py --input_dir {destination.relative_to(ROOT).as_posix()}/images --model baseline3 --output_dir outputs/test-baseline3 --counts_file {destination.relative_to(ROOT).as_posix()}/expected_counts.json

Baseline3命令需真实P06权重和适配器就绪。每次用新输出目录。推理只读取images与expected_counts；真值由独立评估器读取。Baseline1用原始基线输出，P07场景门控另行报告，不混用指标。

原P03包完全保留。这个派生副本只整理必要文件，不复制旧映射表和拍摄表；来源在dataset_info中记录。data/默认被Git忽略，交接时要连同本目录真实文件一起打包。
'''
        (staging / 'README.md').write_text(readme, encoding='utf-8')
        checksums = tree_hashes(staging)
        (staging / 'SHA256SUMS.txt').write_text(''.join(f'{sha}  {name}\n' for name, sha in checksums.items()), encoding='utf-8')
        # Check every copied payload is byte-identical, including labels and source images.
        for old, sha in original_hashes.items():
            if old.startswith('images/test/'):
                new = staging / 'images' / Path(old).name
            elif old.startswith('labels/test/'):
                new = staging / 'labels' / Path(old).name
            elif old in ('ground_truth_objects.csv', 'classes.txt'):
                new = staging / old
            else:
                continue
            if not new.is_file() or digest(new) != sha:
                raise RuntimeError(f'Copied payload changed: {old}')
        staging.rename(destination)
    print(json.dumps(dict(destination=destination.relative_to(ROOT).as_posix(), **info['labels'],
                          images=info['images'], scenes=info['scenes'], objects=info['object_instances']), ensure_ascii=False))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/test_20261124')
    args = parser.parse_args()
    prepare(args.source or discover_source(), args.output)
