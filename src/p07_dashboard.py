"""OpenCV demonstration dashboard; hit regions share the render coordinates."""
from __future__ import annotations

import cv2
import numpy as np

from .p07_bridge import CLASSES, MODELS, PRODUCTION_MODELS
from .work_orders import reference_counts

WIDTH, HEIGHT = 1600, 1000
BG = (24, 20, 16)
CARD = (43, 36, 29)
INK = (245, 240, 232)
MUTED = (169, 153, 133)
BLUE = (245, 155, 65)
GREEN = (151, 214, 70)
RED = (105, 103, 250)
AMBER = (81, 194, 248)


def regions():
    controls = [("model:" + m, (150+i*140, 82, 282+i*140, 126)) for i, m in enumerate(PRODUCTION_MODELS)]
    controls += [(name, rect) for name, rect in [
        ("camera", (150, 144, 260, 184)), ("image", (270, 144, 380, 184)),
        ("video", (390, 144, 500, 184)), ("close", (510, 144, 600, 184)),
        ("camera-", (685, 144, 725, 184)), ("camera-index", (733, 144, 807, 184)),
        ("camera+", (815, 144, 855, 184)), ("toggle", (900, 82, 1060, 126)),
        ("snapshot", (1072, 82, 1232, 126)), ("record", (1244, 82, 1404, 126)),
        ("quit", (1416, 82, 1576, 126)), ("counts-file", (1118, 904, 1340, 938)),
        ("diagnostics", (1350, 904, 1576, 938))]]
    controls += [('diag-prev', (860, 250, 938, 278)), ('diag-next', (946, 250, 1028, 278))]
    for c in range(1, 6):
        y = 482+(c-1)*39
        controls += [(f"count-:{c}", (1364, y, 1390, y+30)),
                     (f"count:{c}", (1394, y, 1454, y+30)),
                     (f"count+:{c}", (1458, y, 1484, y+30))]
    return controls


def hit_test(x, y):
    return next((action for action, (x1, y1, x2, y2) in regions() if x1 <= x < x2 and y1 <= y < y2), None)


def text(canvas, value, x, y, scale=.55, color=INK, thickness=1, max_width=None):
    value = str(value)
    if max_width:
        original = value
        while value and cv2.getTextSize(value, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)[0][0] > max_width:
            value = value[:-1]
        if value != original:
            value = value[:-3] + "..."
    cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def card(canvas, x, y, w, h, title):
    cv2.rectangle(canvas, (x, y), (x+w, y+h), CARD, -1)
    text(canvas, title.upper(), x+18, y+25, .46, MUTED)


def render(packet, result, expected, model, mode, state):
    canvas = np.full((HEIGHT, WIDTH, 3), BG, np.uint8)
    result = result or {}
    text(canvas, "MECHANICAL COMPONENT", 24, 35, .77, INK, 2)
    text(canvas, "INSPECTION STATION  /  P07", 24, 58, .44, MUTED)
    text(canvas, "MOCK INFERENCE  |  SYNTHETIC RESULTS" if mode == "mock" else "REAL MODEL BACKEND", 1050, 37, .55, AMBER if mode == "mock" else GREEN, 1)
    text(canvas, "MODEL", 24, 109, .55, MUTED)
    text(canvas, "INPUT", 24, 170, .55, MUTED)
    text(canvas, "CAMERA", 615, 170, .44, MUTED)
    text(canvas, "Select a source to begin" if state.get("source", "none") == "none" else "Click controls or use keyboard shortcuts", 900, 170, .5, MUTED)
    labels = {"camera": "Camera", "image": "Image", "video": "Video", "close": "Close", "camera-": "-", "camera+": "+",
              "camera-index": str(state.get("camera_index", 0)), "toggle": "Pause" if state.get("source", "none") != "none" and state.get("detecting", True) else "Start",
              "snapshot": "Snapshot", "record": "Stop REC" if state.get("recording") else "Record", "quit": "Quit",
              "counts-file": "Import counts JSON", "diagnostics": "Sizes / Reasons", "scenario": "Next MOCK case"}
    for action, rect in regions():
        if action.startswith('diag-'):
            continue
        if action.startswith("count") and ":" in action:
            continue
        x1, y1, x2, y2 = rect
        selected = action == "model:"+model or action == state.get("source")
        fill = (94, 65, 33) if selected else (65, 53, 42)
        if action == "record" and state.get("recording"):
            fill = (64, 54, 135)
        cv2.rectangle(canvas, (x1, y1), (x2, y2), fill, -1)
        if selected:
            cv2.rectangle(canvas, (x1, y1), (x2, y2), BLUE, 2)
        label = action.split(":", 1)[1] if action.startswith("model:") else labels[action]
        text(canvas, label, x1+12, y1+27, .53, INK, 1)
    # Letterbox the exact inferred image; source pixels and result coordinates remain unchanged.
    card(canvas, 24, 208, 1050, 690, "Inspection view")
    if packet is not None:
        image = packet.image.copy()
        scale = min(1018/image.shape[1], 604/image.shape[0])
        image = cv2.resize(image, (max(1, int(image.shape[1]*scale)), max(1, int(image.shape[0]*scale))))
        for obj in result.get("objects", []):
            x1, y1, x2, y2 = [int(obj[k]*scale) for k in ("xmin", "ymin", "xmax", "ymax")]
            color = RED if obj["class_id"] == 0 or obj.get('inspection_rejection') else GREEN
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            label = f'ID{obj["class_id"]} {CLASSES[obj["class_id"]]} {obj["score"]:.3f}'
            if obj.get('class_assignment') == 'size_rule_non_catalogue':
                label = 'non_catalogue (confirmed size rule)'
            text(image, label, max(2, x1), max(18, y1-6), .5, color, 1)
        x, y = 40+(1018-image.shape[1])//2, 254+(604-image.shape[0])//2
        canvas[y:y+image.shape[0], x:x+image.shape[1]] = image
    else:
        text(canvas, "Ready for inspection", 255, 505, 1.1, INK, 2)
        text(canvas, "Choose Camera, Image or Video in the control bar", 220, 548, .65, MUTED)
        text(canvas, "Camo camera: select its index, then click Camera", 235, 583, .57, MUTED)
    text(canvas, "MOCK  /  Only inference results are simulated" if mode == "mock" else
         f'{state.get("deployment", "Real model")} / {state.get("version", "")} / L: reload model', 42, 882, .53, AMBER if mode == "mock" else GREEN)
    card(canvas, 1094, 208, 482, 90, "Current model")
    text(canvas, model.upper(), 1112, 273, .95, BLUE, 2)
    capability = "IMAGE LEVEL" if model in MODELS[:2] else "OBJECT LEVEL"
    text(canvas, capability, 1370, 268, .44, MUTED)
    if model == 'baseline3':
        text(canvas, result.get('baseline3_mode', state.get('model_mode', 'loading')), 1112, 292, .4, AMBER)
    card(canvas, 1094, 310, 482, 104, "Inspection result")
    status = result.get("status", "PREVIEW")
    color = GREEN if status == "PASS" else RED if status in ("FAIL", "ERROR") else AMBER
    text(canvas, status, 1112, 375, .72 if len(status) > 10 else 1.05, color, 2)
    score = result.get("prediction_score")
    meaning = f'{score:.3f} | {state.get("score_meaning") or result.get("score_type")}' if score is not None else "Decision: N/A  |  Score: N/A"
    if score is None and status in ('PASS', 'FAIL'):
        meaning = 'Score: N/A (count-rule decision)' if model in MODELS[2:] else 'Score: N/A'
    text(canvas, meaning, 1112, 399, .45, MUTED)
    card(canvas, 1094, 426, 482, 260, "Counts summary   /   click expected to edit")
    for label, x in (("CLASS", 1112), ("OBS", 1306), ("EXPECTED", 1384), ("DIFF", 1512)):
        text(canvas, label, x, 471, .4, MUTED)
    counts = result.get("counts")
    for c in range(1, 6):
        y = 482+(c-1)*39
        cv2.line(canvas, (1112, y+34), (1558, y+34), (63, 53, 43), 1)
        obs = counts[str(c)] if counts is not None else None
        diff = obs-expected[str(c)] if obs is not None else None
        text(canvas, f'{c} {CLASSES[c]}', 1112, y+21, .48)
        text(canvas, obs if obs is not None else "N/A", 1310, y+21, .5)
        for action, rect in regions():
            if action not in (f"count-:{c}", f"count:{c}", f"count+:{c}"):
                continue
            x1, y1, x2, y2 = rect
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (65, 53, 42), -1)
            label = str(expected[str(c)]) if action.startswith("count:") else ("+" if action.startswith("count+") else "-")
            text(canvas, label, x1+5, y1+21, .44, BLUE, max_width=x2-x1-8)
        text(canvas, f'{diff:+d}' if diff is not None else "N/A", 1512, y+21, .5, RED if diff else MUTED)
    card(canvas, 1094, 698, 482, 62, "Non-catalogue / ID 0")
    id0 = result.get("non_catalogue_count")
    text(canvas, f'WARNING: {id0} forbidden object(s)' if id0 else ("Not supported / not available" if id0 is None else "0 detected"), 1112, 746, .57, RED if id0 else MUTED, 1)
    card(canvas, 1094, 772, 482, 120, "Performance & source")
    perf = [f'FPS capture {state.get("capture_fps", 0):.1f}  /  processing {state.get("processing_fps", 0):.1f}',
            f'Infer {state.get("inference_ms", "N/A")} ms / proxy {state.get("processing_ms", "N/A")} ms',
            f'{state.get("source", "none")} | {"CHECK SOURCE" if state.get("source_status") else ("RUN" if state.get("detecting", True) else "PAUSED")} | {state.get("resolution", "N/A")} | #{packet.frame_id if packet else "N/A"}',
            f'Device: {state.get("device", "N/A")}']
    for i, line in enumerate(perf):
        text(canvas, line, 1112, 817+i*20, .43, MUTED, max_width=445)
    fixed_text = ' / '.join(str(v) for v in reference_counts().values())
    text(canvas, (f"Fixed counts: {fixed_text} | " if model in MODELS[:2] else "") + "Source status: " + str(state.get("source_status") or "Ready"), 24, 926, .5, AMBER, max_width=1040)
    text(canvas, result.get("message") or state.get("message", ""), 24, 950, .46, MUTED, max_width=1040)
    cv2.line(canvas, (24, 963), (1576, 963), (65, 53, 42), 1)
    text(canvas, "1-3 Models   Space Start/Pause   C Camera   I Image   V Video   S Snapshot   R Record   L Reload   Q Quit", 24, 988, .48, MUTED)
    if state.get("loading"):
        text(canvas, "Loading " + state["loading"] + "...", 730, 109, .47, AMBER)
    if state.get("pending_counts"):
        text(canvas, "Updating counts...", 730, 130, .42, AMBER)
    if model == 'baseline3' and state.get('show_diagnostics'):
        draw_diagnostics(canvas, result, state.get('diagnostic_page', 0))
    return canvas


def draw_diagnostics(canvas, result, page=0):
    """Paged, read-only view over real evidence; no additional inference."""
    card(canvas, 40, 250, 1018, 610, 'Size evidence / D toggle / [ ] object / provisional is not certified mm')
    text(canvas, '< Prev', 867, 272, .42, BLUE)
    text(canvas, 'Next >', 950, 272, .42, BLUE)
    objects = result.get('objects', [])
    inspection = result.get('inspection', {})
    lines = [('Status: ' + str(result.get('status', 'PREVIEW')), INK)]
    if not inspection and result.get('message'):
        lines.append((result['message'], AMBER))
    lines += [(s, RED) for s in inspection.get('failure_reasons', [])]
    lines += [(s, AMBER) for s in inspection.get('review_reasons', [])[:5]]
    if objects:
        obj = objects[page % len(objects)]
        ev = obj.get('size_evidence', {})
        measurement = ev.get('measurement', {})
        cal = ev.get('calibration') or inspection.get('calibration') or {}
        detector_class = obj.get('detector_class_id', obj['class_id'])
        lines.append((f"Object {obj['detection_id']} / {len(objects)}: ID{obj['class_id']} {CLASSES[obj['class_id']]} | detector ID{detector_class} {CLASSES[detector_class]} {obj['score']:.3f}", BLUE))
        trusted = bool(cal.get('validated') and inspection.get('acceptance_confirmed')
                       and ev.get('measurement_policy_reason') == 'VALIDATED_SCOPE_MATCHED')
        lines.append((f"Calibration: {cal.get('calibration_id', 'unavailable')} / validated={cal.get('validated', False)}", AMBER))
        lines.append(('Units: ' + ('validated scope; see evidence' if trusted else 'PROVISIONAL estimates only; not certified physical dimensions'), GREEN if trusted else AMBER))
        lines.append(('Measurement: ' + measurement.get('status', 'NOT_AVAILABLE') + ' / ' + (ev.get('reason') or measurement.get('reason') or ''), MUTED))
        lines.append(('Policy: ' + ev.get('measurement_policy_reason', 'NOT_AVAILABLE'), AMBER))
        lines.append(('Size check: ' + ev.get('conclusion', 'INSUFFICIENT_EVIDENCE'), AMBER))
        for key, dim in measurement.get('dimensions', {}).items():
            lines.append((f"{key}: {dim['mm_estimate']:.3f} mm {'(provisional)' if not trusted else ''}", INK))
        for comparison in ev.get('comparisons', []):
            if comparison['class_id'] != detector_class: continue
            for key, value in comparison['dimensions'].items():
                low, high = value['allowed_mm']
                lines.append((f"{key}: reference {(low+high)/2:g} +/- {(high-low)/2:g} mm (provisional allowance); {value['state']}", MUTED))
                error = value.get('max_error_mm')
                lines.append((f"  measured error bound: {error if error is not None else 'NOT VALIDATED'} mm; evidence {value.get('evidence_id') or 'none'}", MUTED))
        if not ev.get('comparisons'):
            for spec in inspection.get('specifications', []):
                if spec['class_id'] == detector_class:
                    for key, value in spec['dimensions'].items():
                        lines.append((f"{key}: reference {value['reference_mm']:g} +/- {value['provisional_allowance_mm']:g} mm; UNVERIFIED", MUTED))
        # The independent development candidate uses empirical visible-size
        # ranges. Show the actual decision range alongside supplier references
        # so a provisional PASS is not mistaken for nominal-tolerance acceptance.
        for key, value in ev.get('development_acceptance_bands', {}).items():
            lines.append((f"Development band {key}: {value['min_mm']:.3f}..{value['max_mm']:.3f} mm / NOT approved tolerance", AMBER))
        quality = ev.get('quality', {})
        if quality.get('ruler_scale_envelope') is not None:
            lines.append((f"Scale-position envelope: {100*quality['ruler_scale_envelope']:.2f}% / physical error bound NOT VALIDATED", AMBER))
        if obj.get('inspection_rejection'):
            lines.append((obj['inspection_rejection'], RED))
    else:
        lines.append(('No object measurements. See diagnostics.json for calibration and foreground status.', MUTED))
    for i, (line, color) in enumerate(lines[:25]):
        text(canvas, line, 58, 300+i*21, .46, color, max_width=980)
    text(canvas, 'Snapshot exports complete per-object evidence to diagnostics.json', 58, 843, .44, BLUE)
