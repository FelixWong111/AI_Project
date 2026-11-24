"""Six-photo ruler/outer-contour experiment; never a production calibration.

No YOLO calls, labels, training data, acceptance thresholds or frozen-config edits.
ROIs below are manually selected for these six EXIF-corrected photographs only.
Adjacent visible minor ruler ticks are provisionally interpreted as 1 mm.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import uuid

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageOps

ROOT = Path(__file__).resolve().parents[1]
OBJECT_ROI = (650, 1000, 2700, 3200)
THRESHOLDS = (100, 120, 140)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_display(path):
    with Image.open(path) as raw:
        raw.load()
        orientation = raw.getexif().get(274, 1)
        if type(orientation) is not int or orientation not in range(1, 9):
            raise ValueError(f'{path}: INVALID_EXIF')
        meta = dict(original_width=raw.width, original_height=raw.height,
                    exif_orientation=orientation)
        image = ImageOps.exif_transpose(raw).convert('RGB')
    meta.update(display_width=image.width, display_height=image.height)
    return image, meta


def validate_roi(gray, roi):
    x0, y0, x1, y1 = roi
    if not (0 <= x0 < x1 <= gray.shape[1] and 0 <= y0 < y1 <= gray.shape[0]):
        raise ValueError(f'INVALID_ROI: {roi} for {gray.shape}')
    return gray[y0:y1, x0:x1]


def detect_ticks(gray, roi, axis):
    if axis not in ('x', 'y'):
        raise ValueError('INVALID_AXIS')
    crop = validate_roi(gray, roi)
    kernel = np.ones((35, 1) if axis == 'x' else (1, 35), np.uint8)
    strokes = cv2.morphologyEx((crop < 100).astype(np.uint8), cv2.MORPH_OPEN, kernel)
    signal = strokes.sum(axis=0 if axis == 'x' else 1)
    indices = np.flatnonzero(signal > 25)
    groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
    offset = roi[0] if axis == 'x' else roi[1]
    return np.array([float(np.average(g, weights=signal[g])) + offset
                     for g in groups if len(g)])


def regular_run(points):
    points = np.asarray(points, dtype=float)
    if points.ndim != 1 or len(points) < 20 or not np.isfinite(points).all():
        raise ValueError('INSUFFICIENT_OR_INVALID_TICKS')
    gaps = np.diff(points)
    if np.any(gaps <= 0):
        raise ValueError('UNORDERED_TICKS')
    median = np.median(gaps)
    breaks = np.flatnonzero((gaps < .9 * median) | (gaps > 1.1 * median)) + 1
    run = max(np.split(points, breaks), key=len)
    if len(run) < 20:
        raise ValueError('NO_REGULAR_20_TICK_SEQUENCE')
    return run


def fit_ticks(points):
    """Fit even-index ticks; odd-index ticks never enter the regression.

    Reject missing/spurious ticks here; explicit regular-run selection is separate.
    Tick sequence selection uses all positions, so holdout is only an internal
    consistency diagnostic, not an unbiased external calibration validation.
    """
    points = np.asarray(points, dtype=float)
    run = regular_run(points)
    if len(run) != len(points):
        raise ValueError('IRREGULAR_OR_MISSING_TICKS')
    index = np.arange(len(points), dtype=float)
    fit = np.arange(len(points)) % 2 == 0
    slope, intercept = np.polyfit(index[fit], points[fit], 1)
    if not np.isfinite([slope, intercept]).all() or slope <= 0:
        raise ValueError('INVALID_SCALE')
    residual = points - (slope * index + intercept)
    held = residual[~fit]
    return dict(px_per_mm=float(slope), intercept_px=float(intercept),
                fit_count=int(fit.sum()), holdout_count=int((~fit).sum()),
                tick_count=len(points), span_mm=len(points) - 1,
                first_tick_px=float(points[0]), last_tick_px=float(points[-1]),
                holdout_rmse_px=float(np.sqrt(np.mean(held ** 2))),
                holdout_max_abs_px=float(np.max(np.abs(held))),
                holdout_rmse_mm=float(np.sqrt(np.mean(held ** 2)) / slope),
                holdout_max_abs_mm=float(np.max(np.abs(held)) / slope))


def candidate_contour(gray, roi, threshold=120):
    crop = validate_roi(gray, roi)
    mask = (crop < threshold).astype(np.uint8) * 255
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [c for c in contours if cv2.contourArea(c) >= 1000]
    if not contours:
        return None
    contours.sort(key=cv2.contourArea, reverse=True)
    # Do not silently select among multiple substantial candidate objects.
    if len(contours) > 1 and cv2.contourArea(contours[1]) > .1 * cv2.contourArea(contours[0]):
        raise ValueError('MULTIPLE_SUBSTANTIAL_CONTOURS')
    contour = contours[0]
    x, y, w, h = cv2.boundingRect(contour)
    if x <= 1 or y <= 1 or x + w >= crop.shape[1] - 1 or y + h >= crop.shape[0] - 1:
        raise ValueError('CONTOUR_TOUCHES_ROI_BOUNDARY')
    return contour + np.array([[[roi[0], roi[1]]]], np.int32)


def metric_envelope(contour, sx, sy):
    if not np.isfinite([sx, sy]).all() or min(sx, sy) <= 0:
        raise ValueError('INVALID_SCALE')
    metric = contour.astype(np.float32) / np.array([sx, sy], np.float32)
    rect = cv2.minAreaRect(metric)
    long, short = sorted(rect[1], reverse=True)
    if not np.isfinite([long, short]).all() or short <= 0:
        raise ValueError('DEGENERATE_CONTOUR')
    corners = cv2.boxPoints(rect) * np.array([sx, sy], np.float32)
    return dict(long_mm_candidate=float(long), short_mm_candidate=float(short),
                envelope_xy=corners.tolist(), contour_area_px=float(cv2.contourArea(contour)))


def ruler_rois(name):
    if name in ('2.jpg', '3.jpg'):
        return {'x': (800, 3310, 2800, 3365), 'y': (450, 1400, 600, 3100)}
    return {'x': (700, 3600, 2800, 3655), 'y': (350, 150, 560, 3100)}


def font(size):
    try:
        return ImageFont.truetype('C:/Windows/Fonts/arial.ttf', size)
    except OSError:
        return ImageFont.load_default(size=size)


def tick_overlay(image, roi, axis, raw, selected):
    crop = image.crop(roi)
    draw = ImageDraw.Draw(crop)
    used = {float(value): i for i, value in enumerate(selected)}
    for value in raw:
        index = used.get(float(value))
        color = '#ff3030' if index is None else ('#00a6ff' if index % 2 == 0 else '#ff00cc')
        pos = int(round(value - (roi[0] if axis == 'x' else roi[1])))
        if axis == 'x':
            draw.line((pos, 0, pos, 12), fill=color, width=3)
        else:
            draw.line((crop.width - 14, pos, crop.width - 1, pos), fill=color, width=3)
    return crop


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def ruler_contact(folder):
    """Readable tick-selection overview; images are diagnostic crops only."""
    sheet = Image.new('RGB', (1500, 1320), 'white')
    draw = ImageDraw.Draw(sheet)
    for index in range(1, 7):
        top = (index - 1) * 220
        draw.text((10, top), f'{index}.jpg: x below; y rotated 90 deg. Blue=fit, magenta=holdout, red=excluded',
                  fill='black', font=font(22))
        for axis, y in (('x', top + 30), ('y', top + 100)):
            with Image.open(folder / f'{index}_{axis}.png') as raw:
                crop = raw.convert('RGB')
            if axis == 'y':
                crop = crop.transpose(Image.Transpose.ROTATE_90)
            crop.thumbnail((1480, 115))
            sheet.paste(crop, (10, y))
    sheet.save(folder / 'contact.jpg', quality=94)


def generate(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError('SOURCE_OUTPUT_OVERLAP')
    if output.exists():
        raise FileExistsError(f'Output exists; choose a new directory: {output}')
    paths = [source / f'{i}.jpg' for i in range(1, 7)]
    if not source.is_dir() or any(not p.is_file() or p.is_symlink() for p in paths):
        raise ValueError('Expected six ordinary files: 1.jpg ... 6.jpg')
    before = {p.name: sha256(p) for p in paths}
    output.parent.mkdir(parents=True, exist_ok=True)
    # Ordinary mkdir inherits the workspace ACL on Windows. tempfile.mkdtemp's
    # private ACL can make published images unreadable to the user/viewer.
    stage = output.parent / (output.name + '_incomplete_' + uuid.uuid4().hex)
    stage.mkdir()
    for directory in ('previews', 'ruler_checks', 'object_crops'):
        (stage / directory).mkdir()
    scales, ticks, objects, sources, thumbnails = [], [], [], [], []
    for path in paths:
        image, meta = load_display(path)
        if image.size != (3024, 4032):
            raise ValueError(f'{path.name}: photo-specific ROIs require display 3024x4032')
        gray = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2GRAY)
        annotated = image.copy(); draw = ImageDraw.Draw(annotated)
        per_axis = {}
        rois = ruler_rois(path.name)
        for axis, roi in rois.items():
            raw = detect_ticks(gray, roi, axis)
            selected = regular_run(raw)
            fit = fit_ticks(selected); per_axis[axis] = fit
            scales.append(dict(filename=path.name, axis=axis, raw_peak_count=len(raw),
                               rejected_peak_count=len(raw) - len(selected), roi=json.dumps(roi), **fit))
            index_by_value = {float(v): i for i, v in enumerate(selected)}
            for value in raw:
                index = index_by_value.get(float(value))
                ticks.append(dict(filename=path.name, axis=axis, pixel_position=float(value),
                                  role='EXCLUDED' if index is None else ('FIT' if index % 2 == 0 else 'HOLDOUT'),
                                  relative_mm='' if index is None else index,
                                  residual_px='' if index is None else float(value - fit['intercept_px'] - fit['px_per_mm'] * index)))
            tick_overlay(image, roi, axis, raw, selected).save(stage / 'ruler_checks' / f'{path.stem}_{axis}.png')
            draw.rectangle(roi, outline='#0088ff', width=4)
        sx, sy = per_axis['x']['px_per_mm'], per_axis['y']['px_per_mm']
        draw.rectangle(OBJECT_ROI, outline='#ff9900', width=3)
        main = None
        for threshold in THRESHOLDS:
            row = dict(filename=path.name, threshold=threshold, status='',
                       long_mm_candidate='', short_mm_candidate='', contour_area_px='', reason='')
            try:
                contour = candidate_contour(gray, OBJECT_ROI, threshold)
                if contour is None:
                    row['status'] = 'NO_CONTOUR'
                else:
                    envelope = metric_envelope(contour, sx, sy)
                    row.update({k: v for k, v in envelope.items() if k != 'envelope_xy'})
                    row['status'] = 'CANDIDATE_OUTER_ENVELOPE'
                    if threshold == 120:
                        main = (contour, envelope)
            except ValueError as error:
                row.update(status='REJECTED', reason=str(error))
            objects.append(row)
        if main is not None:
            contour, envelope = main
            draw.line([tuple(p) for p in contour[:, 0]] + [tuple(contour[0, 0])], fill='#00ff44', width=5)
            corners = [tuple(p) for p in envelope['envelope_xy']]
            draw.line(corners + [corners[0]], fill='#ff00cc', width=4)
            x, y, w, h = cv2.boundingRect(contour)
            crop = annotated.crop((max(0, x - 120), max(0, y - 120), min(image.width, x + w + 120), min(image.height, y + h + 120)))
            crop.save(stage / 'object_crops' / f'{path.stem}.png')
            message = f"outer envelope ~{envelope['long_mm_candidate']:.2f} x {envelope['short_mm_candidate']:.2f} mm"
        else:
            message = 'No usable contour at threshold 120; see CSV (not a class ID)'
        lines = [f'{path.name}  CANDIDATE ONLY / NOT SIZE ACCEPTANCE',
                 f'local x={sx:.3f}, y={sy:.3f} px/mm; threshold=120', message,
                 'green: dark contour (may include shadow); magenta: outer envelope']
        draw.rectangle((0, 0, image.width, 250), fill='white')
        for i, line in enumerate(lines):
            draw.text((25, 10 + 56 * i), line, fill='black', font=font(40))
        annotated.thumbnail((900, 1200))
        annotated.save(stage / 'previews' / path.name, quality=92)
        thumb = annotated.copy(); thumb.thumbnail((450, 600)); thumbnails.append(thumb)
        sources.append(dict(filename=path.name, **meta, sha256_before=before[path.name],
                            sha256_after=sha256(path), status='SOURCE_UNCHANGED'))
        if sources[-1]['sha256_before'] != sources[-1]['sha256_after']:
            raise ValueError(f'{path.name}: SOURCE_CHANGED')
    # Check the whole source set again, after all processing.
    if before != {p.name: sha256(p) for p in paths}:
        raise ValueError('SOURCE_SET_CHANGED')
    contact = Image.new('RGB', (1350, 1200), 'white')
    for i, thumb in enumerate(thumbnails):
        contact.paste(thumb, ((i % 3) * 450, (i // 3) * 600))
    contact.save(stage / 'contact.jpg', quality=94)
    ruler_contact(stage / 'ruler_checks')
    write_csv(stage / 'scale_fits.csv', scales)
    write_csv(stage / 'tick_points.csv', ticks)
    write_csv(stage / 'outer_envelopes.csv', objects)
    write_csv(stage / 'source_integrity.csv', sources)
    provenance = dict(created_utc=datetime.now(timezone.utc).isoformat(), source=str(source),
                      tool_sha256=sha256(__file__), python=sys.version, opencv=cv2.__version__, numpy=np.__version__,
                      ruler_tick_mm_assumption=1, calibration_status='CANDIDATE_NOT_ACCEPTED',
                      object_roi=OBJECT_ROI, thresholds=THRESHOLDS, sources=sources)
    (stage / 'run.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')
    report = make_report(scales, objects)
    (stage / 'report.md').write_text(report, encoding='utf-8')
    # New path only: never replace an existing experiment. Incomplete stages stay
    # clearly named if any exception occurs; original inputs are never written.
    if output.exists():
        raise FileExistsError(output)
    stage.rename(output)
    return output


def make_report(scales, objects):
    rows = ['| Photo | x px/mm | y px/mm | x holdout max (mm) | y holdout max (mm) |',
            '|---|---:|---:|---:|---:|']
    for name in [f'{i}.jpg' for i in range(1, 7)]:
        x, y = [next(s for s in scales if s['filename'] == name and s['axis'] == a) for a in ('x', 'y')]
        rows.append(f"| {name} | {x['px_per_mm']:.4f} | {y['px_per_mm']:.4f} | {x['holdout_max_abs_mm']:.4f} | {y['holdout_max_abs_mm']:.4f} |")
    measurements = ['| Photo | Threshold 120 outer envelope (candidate mm) | Successful thresholds | Successful long / short range (mm) |', '|---|---|---|---|']
    for name in [f'{i}.jpg' for i in range(1, 7)]:
        candidates = [o for o in objects if o['filename'] == name and o['status'] == 'CANDIDATE_OUTER_ENVELOPE']
        main = next((o for o in candidates if o['threshold'] == 120), None)
        if main:
            long = [o['long_mm_candidate'] for o in candidates]; short = [o['short_mm_candidate'] for o in candidates]
            accepted = ', '.join(str(o['threshold']) for o in candidates)
            measurements.append(f"| {name} | {main['long_mm_candidate']:.2f} × {main['short_mm_candidate']:.2f} | {accepted} | {min(long):.2f}–{max(long):.2f} / {min(short):.2f}–{max(short):.2f} |")
        else:
            measurements.append(f'| {name} | No usable contour | None | See outer_envelopes.csv |')
    return '\n'.join([
        '# P06 候选局部尺度与轮廓测量实验（未验收）', '',
        '## 范围与数据', '',
        '仅读取 `data/p01/measure/1.jpg`–`6.jpg`。6 张均为原始 4032×3024、EXIF=6，方向校正后为 3024×4032。'
        '只生成预览和测量记录；未读取 Train/Validation/Test，未运行 YOLO、训练或修改冻结配置。原图逐文件 SHA-256 前后核验见 source_integrity.csv。', '',
        '## 方法与刻度留出结果', '',
        '在每张方向校正图的底部横尺、左侧竖尺手动选择 ROI（具体坐标见 scale_fits.csv）。'
        '暗色阈值 100、35 像素线状形态学开运算和投影提取刻度中心；取间距在中位数 ±10% 内的最长连续段，至少 20 点。'
        '不跨过缺失刻度补编号；排除点完整记入 tick_points.csv。相邻小刻度暂按 1 mm，必须人工确认刻度单位与序列。'
        '偶数序号拟合 pixel = intercept + px_per_mm × relative_mm，奇数序号仅计算残差。', '',
        *rows, '',
        '**这些留出误差是同图、同尺的内部一致性诊断。ROI 和连续段选择使用了图像信息，不是独立物理精度验证；'
        '不能据此宣称毫米级验收精度。** 横纵尺分别拟合两个局部一维尺度；没有求内参、畸变、透视单应性或整个平面的误差模型。', '',
        '`ruler_checks/` 蓝色=拟合点，紫色=留出点，红色=排除点。请核对没有漏掉/重复计数小刻度。'
        'x/y 只取坐标投影，没有把标尺倾斜和透视完全校正。2、3 号图比其余图的局部 px/mm 低约 3%–4%，'
        '当前证据不支持合并为一个全局尺度；原因可能涉及成像几何/标尺平面等，尚未确认。', '',
        '## 五个实物的轮廓预览', '',
        '在固定内部 ROI 中，以阈值 120 提取暗色外轮廓、7×7 闭运算，保留面积至少 1000 px² 的轮廓。'
        '另用阈值 100、140 记录敏感性；多个显著轮廓或接触 ROI 边缘时拒绝。'
        '局部 x/y 尺度换算后求最小外接矩形。绿色是候选轮廓，紫色是外包矩形；阴影可能被包含。'
        '2/3/4 是三个螺钉示例，5 是垫圈，6 是螺母，仅为人工视觉描述，不是自动分类或已确认的规格身份。', '',
        *measurements, '',
        '1.jpg 无候选轮廓；没有输出任何 ID 0。三个螺钉的外包尺寸包含头部且受阴影影响，'
        '**不是螺纹直径或规定的杆长**。垫圈为外包尺寸而非内孔直径，螺母外包尺寸不是经确认的对边距离。'
        '表中小数位只保留计算记录，不表示实际测量精度。阈值区间不是置信区间。'
        '3.jpg 在阈值 140 出现第二个显著轮廓，被明确拒绝；未把这个失败当成成功测量。', '',
        '## 尚未通过的验收条件与下一步', '',
        '1. 人工确认每张尺的 1 mm 小刻度、候选段和轮廓是否对齐；确认标尺刻度面与工件测量面的高度关系。',
        '2. 在有效检测区中心、四角补同平面网格或多位置标尺；需要校正镜头畸变时另拍多姿态棋盘格/ChArUco。'
        '当前照片不能代替完整相机标定。',
        '3. 提供这五件实物的卡尺测量值：螺钉按规定量杆长/螺纹外径，垫圈内外径，螺母对边等。'
        '以实测偏差检验轮廓算法，而非用标称 M4/M5 当准确真值。',
        '4. 后续只用 Train 调分割/测量参数并记录调参图片；参数固定后在独立 Validation 检查。'
        '本轮未调 Train，也未完成 Validation 的物理尺寸验证。',
        '5. 未知缩放/裁剪的训练图不能直接套 3024×4032 的像素尺度；摄像头 1280×720 等模式需单独验证。'
        '同一相机和支架的用户记录有价值，但这批 EXIF 只有方向，不能独立核实焦距、对焦和机位。',
        '6. 容差须课程/教师确认；之后用错误规格开发样本验证拒识。Test 仍由 P03/P08 封存，本轮不接 PASS/FAIL。', '',
        '## 输出与复现', '',
        '`previews/` 为缩小到最长边 1200 的六张方向校正预览，`contact.jpg` 为联系图；'
        '`object_crops/` 是原像素采样的局部轮廓图，`ruler_checks/` 是原像素标尺 ROI 叠图。'
        'scale_fits.csv、tick_points.csv、outer_envelopes.csv、source_integrity.csv 和 run.json 保存参数与证据。', '',
        '```powershell',
        'conda run -n ai-p06-yolo python -B tools/p06_measure_candidate.py --output output/p06_measure_candidate',
        '```', '',
        '输出存在时拒绝覆盖。此工具仅适用于这六张图的已记录 ROI，不是通用尺度标定器或最终规格检验器。', '',
        '技术依据：[OpenCV 4.10 camera calibration](https://docs.opencv.org/4.10.0/dc/dbb/tutorial_py_calibration.html)、'
        '[planar homography](https://docs.opencv.org/4.10.0/d9/dab/tutorial_homography.html)。'
        '本实验没有完成这些完整标定流程。', '',
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT / 'output/p06_measure_candidate',
                        help='New output directory; existing paths are rejected.')
    args = parser.parse_args()
    try:
        print(generate(ROOT / 'data/p01/measure', args.output))
    except (ValueError, OSError) as error:
        parser.exit(1, f'Candidate experiment failed: {error}\n')


if __name__ == '__main__':
    main()
