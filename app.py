"""P07 live inspection UI. Public inspection implementation remains owned by P08."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path
from collections import deque

import cv2

from src.camera import FrameProvider, Rate
from src.p07_devices import list_cameras, select_camera
from src.p07_bridge import CLASSES, MODELS, SCENARIOS, UnavailableBackend, load_backend, validate_counts
from src.p07_output import Journal, Recorder, dump, export_result
from src.p07_ui import InferenceWorker, present, current_result
from src.work_orders import load_cases, reference_counts
from src.p07_dashboard import hit_test


def select_file(title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(title=title)
    finally:
        root.destroy()


def ask_integer(title: str, value: int, maximum: int) -> int | None:
    import tkinter as tk
    from tkinter import simpledialog
    root = tk.Tk()
    root.withdraw()
    try:
        return simpledialog.askinteger(title, title, initialvalue=value, minvalue=0, maxvalue=maximum, parent=root)
    finally:
        root.destroy()


def load_expected(path: str, image_id: str) -> dict:
    entries = load_cases(path)
    matches = [x for x in entries if x.get("image_id") == image_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one record for image_id=" + image_id)
    return validate_counts(matches[0]["expected_counts"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path(__file__).resolve().parent / "config/p07.json")
    parser.add_argument("--backend", choices=("auto", "mock", "real"), default="real",
                        help="All production models use real inference; unavailable models report ERROR")
    parser.add_argument("--model", choices=MODELS, default="baseline1")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--camera", type=int)
    source.add_argument("--camera-name", help="Select a camera by friendly name, e.g. Camo")
    source.add_argument("--image", type=Path)
    source.add_argument("--video", type=Path)
    parser.add_argument("--capture-backend", choices=("auto", "dshow", "msmf", "any"))
    parser.add_argument("--list-cameras", action="store_true", help="Print device names and backend-specific indices, then exit")
    parser.add_argument("--counts-file", type=Path)
    parser.add_argument("--image-id", help="Dataset ID for camera/video; otherwise snapshot filename is used")
    parser.add_argument("--output-dir", type=Path, default=Path("output/p07"))
    parser.add_argument("--scenario", choices=SCENARIOS, default="pass")
    parser.add_argument("--headless", action="store_true", help="Offline smoke run, no display latency claim")
    parser.add_argument("--max-frames", type=int, default=0)
    parser.add_argument("--record", action="store_true", help="Start annotated MP4 recording on the first displayed result")
    parser.add_argument("--timeout", type=float, default=15.0, help="Headless no-result timeout")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    if args.list_cameras:
        print(json.dumps(list_cameras(args.capture_backend or "auto"), ensure_ascii=False, indent=2))
        return 0
    config = json.loads(args.config.read_text(encoding="utf-8-sig"))
    required = {"camera_backend", "width", "height", "max_count", "record_fps", "reconnect_seconds"}
    if not isinstance(config, dict) or not required.issubset(config):
        parser.error("config must be an object containing " + ", ".join(sorted(required)))
    if args.capture_backend:
        config["camera_backend"] = args.capture_backend
    if config["camera_backend"] not in ("auto", "dshow", "msmf", "any"):
        parser.error("invalid camera backend")
    if args.camera_name:
        selected = select_camera(args.camera_name, config["camera_backend"])
        args.camera = selected["index"]
        # Keep the name and index tied to one backend; another backend may reorder devices.
        config["camera_backend"] = selected["backend"]
        logging.info("Selected %s (index %s / %s)", selected["name"], args.camera, selected["backend"])
    try:
        camera_devices = list_cameras(config["camera_backend"])
    except RuntimeError:
        camera_devices = []
    for name in ("width", "height", "max_count", "record_fps", "reconnect_seconds"):
        if type(config[name]) not in (int, float) or not math.isfinite(config[name]) or config[name] <= 0:
            parser.error(name + " must be positive")
        if name in ("width", "height", "max_count") and type(config[name]) is not int:
            parser.error(name + " must be an integer")
    try:
        expected = validate_counts(config.get("expected_counts", reference_counts()))
        if args.counts_file:
            expected = load_expected(str(args.counts_file), args.image.name if args.image else args.image_id or "")
    except (ValueError, OSError) as exc:
        parser.error('Invalid work order: ' + str(exc))
    if max(expected.values()) > config["max_count"]:
        parser.error("counts exceed configured GUI max_count")
    model = args.model
    def ui_load(mode, selected, settings):
        try:
            # Reread deployment settings on every explicit model selection/reload.
            fresh = json.loads(args.config.read_text(encoding="utf-8-sig"))
            fresh["_p07_config_path"] = str(args.config.resolve())
            fresh['_live_scene_guard'] = True
            return load_backend(mode, selected, fresh)
        except Exception as exc:
            logging.error("%s unavailable (no MOCK fallback): %s", selected, exc)
            return UnavailableBackend(selected, exc)
    backend = ui_load(args.backend, model, config)
    journal = Journal(args.output_dir, dict(config, backend=args.backend, model=model))
    def record_load():
        dump(journal.path / (model + "-metadata.json"), backend.metadata)
        with (journal.path / "model-loads.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(dict(timestamp=time.time(), model=model, is_mock=backend.is_mock,
                                         metadata=backend.metadata), ensure_ascii=False) + "\n")
    record_load()
    logging.info("Session: %s", journal.path)
    worker = InferenceWorker()
    provider = None
    recorder = None
    generation = 0
    detecting = True
    cached = None
    displayed = None
    last_submitted = None
    processing_rate = Rate()
    last_processing_ms = None
    scenario_index = SCENARIOS.index(args.scenario)
    camera_index = args.camera if args.camera is not None else 0
    message = "Select camera (c), image (i), or video (v)"
    last_progress = time.perf_counter()
    completed = 0
    dirty_counts = False
    loading = None
    auto_record = args.record
    clicks = deque()
    requested_model = None
    show_diagnostics = False
    diagnostic_page = 0
    title = "P07 Mechanical Component Inspection"

    def stop_recording():
        nonlocal recorder
        if recorder:
            recorder.close()
            recorder = None

    def switch_source(kind, value):
        nonlocal provider, generation, cached, displayed, last_submitted, last_progress, message
        stop_recording()
        if provider:
            provider.close()
        generation += 1
        cached = displayed = last_submitted = None
        provider = FrameProvider(kind, value, config, journal).start()
        message = "Opened " + kind + ": " + str(value)
        last_progress = time.perf_counter()

    def save_snapshot(canvas):
        nonlocal message
        if not cached:
            raise ValueError("No completed result to export")
        stamp = time.time_ns()
        image_id = args.image_id or (Path(provider.value).name if provider.kind == "image" else f'frame-{cached["packet"].frame_id}.png')
        output = journal.path / ("MOCK-" if backend.is_mock else "REAL-") / str(stamp)
        metadata = dict(backend.metadata)
        metadata.update(model=model, output_capability="image_level" if model in MODELS[:2] else "object_level",
                        class_catalogue=metadata.get("class_catalogue", dict(enumerate(CLASSES))),
                        hardware=dict(journal.hardware, inference_device=metadata.get("hardware")),
                        thresholds=metadata.get("thresholds", "TBD"), preprocessing=metadata.get("preprocessing", "TBD"),
                        coordinate_convention=metadata.get("coordinate_convention") if model in MODELS[:2] else "original image pixels, top-left xyxy",
                        score_convention=metadata.get("score_convention", "TBD"), rejection=metadata.get("rejection", "TBD"),
                        size_tolerance=metadata.get("size_tolerance", "TBD"), calibration=metadata.get("calibration", "TBD"),
                        evidence_mode="mock" if backend.is_mock else "real", frame_id=cached["packet"].frame_id, scene_id=None)
        if model in MODELS[:2]:
            metadata["supported_fixed_counts"] = backend.fixed if backend.is_mock else metadata.get("supported_fixed_counts", "TBD")
        export_result(output, image_id, model, cached["result"], cached["expected"], metadata)
        for name, image in (("annotated.png", canvas), ("source.png", cached["packet"].image)):
            ok, data = cv2.imencode(".png", image)
            if not ok:
                raise OSError("PNG encoder failed")
            data.tofile(output / name)
        logging.info("Saved %s", output)
        message = "Snapshot and result CSV saved to session output folder"

    try:
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1440, 900)
            def mouse_event(event, x, y, flags, userdata):
                if event == cv2.EVENT_LBUTTONUP:
                    action = hit_test(x, y)
                    if action:
                        clicks.append(action)
            cv2.setMouseCallback(title, mouse_event)
        if args.image:
            switch_source("image", args.image)
        elif args.video:
            switch_source("video", args.video)
        elif args.camera is not None:
            switch_source("camera", camera_index)
        elif args.headless:
            parser.error("headless requires an input source")

        while True:
            if requested_model and not worker.busy and not loading:
                loading = (requested_model, worker.pool.submit(ui_load, args.backend, requested_model, config))
                message = "Loading " + requested_model
                requested_model = None
            if loading and loading[1].done():
                selected, future = loading
                loading = None
                try:
                    candidate = future.result()
                    stop_recording()
                    backend, model = candidate, selected
                    record_load()
                    generation += 1
                    cached = displayed = last_submitted = None
                    processing_rate = Rate()
                    message = "Loaded " + model
                except Exception as exc:
                    message = "Model load failed: " + str(exc)
            fresh = worker.poll()
            new_measurement = False
            if fresh and fresh["generation"] == generation:
                cached = fresh
                displayed = fresh["packet"]
                dirty_counts = fresh["expected"] != expected
                new_measurement = not fresh["rejudge"]
                last_progress = time.perf_counter()
            latest = provider.latest if provider else None
            if not loading and not requested_model and not worker.busy and cached and dirty_counts:
                # The bridge's shared rejudge handles count-only changes without inference.
                worker.submit(backend, model, cached["packet"], dict(expected), SCENARIOS[scenario_index], generation, cached["result"])
                dirty_counts = False
            elif not loading and not requested_model and detecting and latest and not worker.busy and latest.frame_id != last_submitted:
                worker.submit(backend, model, latest, dict(expected), SCENARIOS[scenario_index], generation)
                last_submitted = latest.frame_id
            if not detecting and not cached:
                displayed = latest
            shown_expected = expected
            active_backend = provider.backend.lower() if provider else config["camera_backend"]
            device_name = next((d["name"] for d in camera_devices if d["index"] == camera_index and d["backend"] == active_backend), "not selected / see --list-cameras")
            inference_device = backend.metadata.get('hardware', {}).get('device', 'cpu' if model in MODELS[:2] else 'N/A')
            ui_state = dict(source=provider.kind if provider else "none", device=(str(inference_device)+' | '+device_name) if provider and provider.kind == "camera" else inference_device,
                            camera_index=camera_index, detecting=detecting, recording=recorder is not None,
                            capture_fps=latest.capture_fps if latest else 0, processing_fps=processing_rate.fps,
                            resolution=f"{displayed.image.shape[1]}x{displayed.image.shape[0]}" if displayed else "N/A",
                            inference_ms=f'{cached["inference_ms"]:.2f}' if cached and cached["expected"] == expected and cached["inference_ms"] is not None else "N/A",
                            processing_ms=f'{last_processing_ms:.2f}' if last_processing_ms is not None else "N/A",
                            pending_counts=dirty_counts or bool(cached and expected != cached['expected']),
                            source_status=provider.error if provider else "Choose Camera, Image or Video above",
                            message=message + (" | MOCK case: " + SCENARIOS[scenario_index] if backend.is_mock else ""),
                            score_meaning=("P(FAIL), uncalibrated" if model == "baseline2" else "P(label), uncalibrated") if model in MODELS[:2] else "",
                            version=str(backend.metadata.get("model_version", ""))[:16],
                            deployment=backend.metadata.get("deployment_stage", ""),
                            loading=loading[0] if loading else requested_model)
            pending = loading is not None or requested_model is not None
            shown_result = current_result(model, backend, expected, cached, pending)
            effective_mode = "real" if model in MODELS[:2] else "mock" if backend.is_mock else "real"
            ui_state.update(show_diagnostics=show_diagnostics, diagnostic_page=diagnostic_page,
                            model_mode=backend.metadata.get('baseline3_mode', 'external'))
            canvas = present(displayed, shown_result, shown_expected, model, effective_mode, ui_state)
            if auto_record and displayed is not None:
                recorder = Recorder(journal.path / (f'{effective_mode}-{time.time_ns()}.mp4'), config["record_fps"], (canvas.shape[1], canvas.shape[0]))
                auto_record = False
            if args.headless:
                key = -1
                time.sleep(.001)
            else:
                cv2.imshow(title, canvas)
                key = cv2.waitKey(1) & 0xff
                if cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
                    break
            if new_measurement:
                now = time.perf_counter()
                last_processing_ms = (now - displayed.available_at) * 1000
                processing_rate.tick(now)
                row = dict(frame_id=displayed.frame_id, source=displayed.source,
                           mode="mock" if backend.is_mock else ("real_headless" if args.headless else "real"),
                           model=model, model_version=backend.metadata.get("model_version", "TBD"),
                           width=displayed.image.shape[1], height=displayed.image.shape[0], inference_ms=cached["inference_ms"],
                           processing_ms=last_processing_ms, processing_fps=processing_rate.fps, status=cached["result"]["status"],
                           expected_counts=json.dumps(cached["expected"], sort_keys=True))
                journal.latency(row)
                completed += 1
                if recorder:
                    recorder.write(canvas)
                if args.headless:
                    save_snapshot(canvas)
                    if cached["result"]["status"] == "ERROR":
                        return 1
            if args.max_frames and completed >= args.max_frames:
                break
            if args.headless and provider and provider.finished and not worker.busy:
                final_packet = provider.latest
                if final_packet is None:
                    logging.error("Input unavailable: %s", provider.error)
                    return 1
                if cached and cached["packet"].frame_id == final_packet.frame_id:
                    return 0 if completed else 1
            if args.headless and time.perf_counter() - last_progress > args.timeout:
                logging.error("No completed frame: %s", provider.error if provider else message)
                return 1
            if args.headless:
                continue
            try:
                action = clicks.popleft() if clicks else None
                key_actions = {"camera": "c", "image": "i", "video": "v", "close": "x", "camera-": "-", "camera+": "+",
                               "toggle": " ", "snapshot": "s", "record": "r", "quit": "q", "counts-file": "e", "scenario": "n"}
                if action in key_actions:
                    key = ord(key_actions[action])
                elif action and action.startswith("model:"):
                    key = ord(str(MODELS.index(action.split(":")[1])+1))
                elif action == 'diagnostics':
                    key = ord('d')
                elif action in ('diag-prev', 'diag-next'):
                    key = ord('[' if action == 'diag-prev' else ']')
                elif action == "camera-index":
                    value = ask_integer("Camera index", camera_index, 999)
                    if value is not None and value != camera_index:
                        camera_index = value
                        if provider and provider.kind == "camera":
                            switch_source("camera", camera_index)
                elif action and action.startswith(("count:", "count-:", "count+:")):
                    kind, c = action.split(":")
                    value = ask_integer("Expected " + CLASSES[int(c)], expected[c], config["max_count"]) if kind == "count" else max(0, min(config["max_count"], expected[c] + (1 if kind == "count+" else -1)))
                    if value is not None and value != expected[c]:
                        expected = validate_counts(dict(expected, **{c: value}))
                        dirty_counts = True
                if key in (ord("q"), 27):
                    break
                if key == ord('d'):
                    show_diagnostics = not show_diagnostics
                elif key in (ord('['), ord(']')):
                    diagnostic_page += 1 if key == ord(']') else -1
                if key == ord(" "):
                    if provider:
                        detecting = not detecting
                    else:
                        message = "Choose Camera, Image or Video before starting"
                elif key in map(ord, "1234l"):
                    selected = model if key == ord("l") else MODELS[key-ord("1")]
                    stop_recording()
                    generation += 1
                    model = selected
                    cached = displayed = last_submitted = None
                    dirty_counts = False
                    last_processing_ms = None
                    processing_rate = Rate()
                    requested_model = selected
                    message = "Requested " + selected
                elif key == ord("n") and model in MODELS[2:] and backend.is_mock and not pending:
                    scenario_index = (scenario_index+1) % len(SCENARIOS)
                    generation += 1
                    cached = displayed = last_submitted = None
                elif key == ord("c"):
                    switch_source("camera", camera_index)
                elif key == ord("x"):
                    if provider:
                        provider.close()
                    provider = None
                    generation += 1
                    cached = displayed = None
                    stop_recording()
                elif key in (ord("+"), ord("="), ord("-")):
                    camera_index = max(0, camera_index + (-1 if key == ord("-") else 1))
                    if provider and provider.kind == "camera":
                        switch_source("camera", camera_index)
                elif key in (ord("i"), ord("v")):
                    path = select_file("Open image" if key == ord("i") else "Open video")
                    if path:
                        switch_source("image" if key == ord("i") else "video", path)
                elif key == ord("e"):
                    path = select_file("Open expected_counts.json")
                    if path:
                        image_id = Path(provider.value).name if provider and provider.kind == "image" else args.image_id or ""
                        values = load_expected(path, image_id)
                        if max(values.values()) > config["max_count"]:
                            raise ValueError("Increase max_count in config before loading these counts")
                        expected = values
                        dirty_counts = True
                elif key == ord("s"):
                    if dirty_counts or (cached and cached["expected"] != expected):
                        raise ValueError("Wait for count rejudgement before saving")
                    save_snapshot(canvas)
                elif key == ord("r"):
                    if recorder:
                        stop_recording()
                    else:
                        recorder = Recorder(journal.path / (f'{effective_mode}-{time.time_ns()}.mp4'), config["record_fps"], (canvas.shape[1], canvas.shape[0]))
            except (ValueError, OSError, RuntimeError, ImportError) as exc:
                message = str(exc)
                logging.error("%s", message)
        return 0
    finally:
        stop_recording()
        worker.close()
        try:
            if provider:
                provider.close()
        finally:
            journal.close()
            if not args.headless:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (ValueError, OSError, RuntimeError, ImportError, cv2.error) as error:
        logging.error("%s", error)
        raise SystemExit(1)
