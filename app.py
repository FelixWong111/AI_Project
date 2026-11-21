"""P07 live inspection UI. Public inspection implementation remains owned by P08."""
from __future__ import annotations

import argparse
import json
import logging
import math
import time
from pathlib import Path

import cv2

from src.camera import FrameProvider, Rate
from src.p07_devices import list_cameras, select_camera
from src.p07_bridge import CLASSES, MODELS, SCENARIOS, load_backend, validate_counts
from src.p07_output import Journal, Recorder, dump, export_result
from src.p07_ui import InferenceWorker, present


def select_file(title: str) -> str:
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(title=title)
    finally:
        root.destroy()


def load_expected(path: str, image_id: str) -> dict:
    entries = json.loads(Path(path).read_text(encoding="utf-8-sig"))
    if not isinstance(entries, list):
        raise ValueError("expected_counts.json must be a list")
    seen = set()
    for entry in entries:
        if not isinstance(entry, dict) or set(entry) != {"image_id", "expected_counts"}:
            raise ValueError("Each case must contain only image_id and expected_counts")
        if not isinstance(entry["image_id"], str) or not entry["image_id"] or entry["image_id"] in seen:
            raise ValueError("image_id must be a nonempty unique string")
        seen.add(entry["image_id"])
        validate_counts(entry["expected_counts"])
    matches = [x for x in entries if x.get("image_id") == image_id]
    if len(matches) != 1:
        raise ValueError("Expected exactly one record for image_id=" + image_id)
    return validate_counts(matches[0]["expected_counts"])


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config/p07.json"))
    parser.add_argument("--backend", choices=("mock", "real"), default="mock")
    parser.add_argument("--model", choices=MODELS, default="baseline3")
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
    required = {"camera_backend", "width", "height", "max_count", "record_fps", "reconnect_seconds", "expected_counts", "mock_fixed_counts"}
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
    expected = validate_counts(config["expected_counts"])
    if args.counts_file:
        expected = load_expected(str(args.counts_file), args.image.name if args.image else args.image_id or "")
    if max(expected.values()) > config["max_count"]:
        parser.error("counts exceed configured GUI max_count")
    model = args.model
    backend = load_backend(args.backend, model, config)
    journal = Journal(args.output_dir, dict(config, backend=args.backend, model=model))
    dump(journal.path / (model + "-metadata.json"), backend.metadata)
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
        if not cached:
            raise ValueError("No completed result to export")
        stamp = time.time_ns()
        image_id = args.image_id or (Path(provider.value).name if provider.kind == "image" else f'frame-{cached["packet"].frame_id}.png')
        output = journal.path / ("MOCK-" if args.backend == "mock" else "REAL-") / str(stamp)
        metadata = dict(backend.metadata)
        metadata.update(model=model, output_capability="image_level" if model in MODELS[:2] else "object_level",
                        class_catalogue=dict(enumerate(CLASSES)), hardware=journal.hardware,
                        thresholds=metadata.get("thresholds", "TBD"), preprocessing=metadata.get("preprocessing", "TBD"),
                        coordinate_convention="original image pixels, top-left xyxy; edge convention from bridge metadata",
                        score_convention=metadata.get("score_convention", "TBD"), rejection=metadata.get("rejection", "TBD"),
                        size_tolerance=metadata.get("size_tolerance", "TBD"), calibration=metadata.get("calibration", "TBD"),
                        evidence_mode=args.backend, frame_id=cached["packet"].frame_id, scene_id=None)
        if model in MODELS[:2]:
            metadata["supported_fixed_counts"] = backend.fixed if backend.is_mock else metadata.get("supported_fixed_counts", "TBD")
        export_result(output, image_id, model, cached["result"], cached["expected"], metadata)
        for name, image in (("annotated.png", canvas), ("source.png", cached["packet"].image)):
            ok, data = cv2.imencode(".png", image)
            if not ok:
                raise OSError("PNG encoder failed")
            data.tofile(output / name)
        logging.info("Saved %s", output)

    try:
        if not args.headless:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, 1360, 850)
            for c in "12345":
                cv2.createTrackbar("Expected " + c, title, expected[c], config["max_count"], lambda value: None)
        if args.image:
            switch_source("image", args.image)
        elif args.video:
            switch_source("video", args.video)
        elif args.camera is not None:
            switch_source("camera", camera_index)
        elif args.headless:
            parser.error("headless requires an input source")

        while True:
            if loading and loading[1].done():
                selected, future = loading
                loading = None
                try:
                    candidate = future.result()
                    stop_recording()
                    backend, model = candidate, selected
                    dump(journal.path / (model + "-metadata.json"), backend.metadata)
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
            if not loading and not worker.busy and cached and dirty_counts:
                # The bridge's shared rejudge handles count-only changes without inference.
                worker.submit(backend, model, cached["packet"], dict(expected), SCENARIOS[scenario_index], generation, cached["result"])
                dirty_counts = False
            elif not loading and detecting and latest and not worker.busy and latest.frame_id != last_submitted:
                worker.submit(backend, model, latest, dict(expected), SCENARIOS[scenario_index], generation)
                last_submitted = latest.frame_id
            if not detecting and not cached:
                displayed = latest
            shown_expected = cached["expected"] if cached else expected
            active_backend = provider.backend.lower() if provider else config["camera_backend"]
            device_name = next((d["name"] for d in camera_devices if d["index"] == camera_index and d["backend"] == active_backend), "not selected / see --list-cameras")
            info = [f'Source: {provider.kind if provider else "none"} | {("RUN" if detecting else "PAUSED") if provider else "NO INPUT"}',
                    f'Device: {device_name if provider and provider.kind == "camera" else "N/A"}',
                    f'Camera index: {camera_index} | Mock case: {SCENARIOS[scenario_index] if args.backend == "mock" else "N/A"}',
                    f'Frame: {displayed.frame_id if displayed else "N/A"} | {displayed.image.shape[1] if displayed else 0}x{displayed.image.shape[0] if displayed else 0}',
                    f'Capture FPS: {latest.capture_fps if latest else 0:.2f} | Processing FPS: {processing_rate.fps:.2f}',
                    f'Inference ms: {format(cached["inference_ms"], ".2f") if cached and cached["inference_ms"] is not None else "N/A"}',
                    f'Processing proxy ms: {format(last_processing_ms, ".2f") if last_processing_ms is not None else "N/A"}',
                    ('COUNTS UPDATE PENDING' if dirty_counts or (cached and expected != cached['expected']) else ''),
                    (provider.error if provider else message)[:62],
                    ('RECORDING (constant FPS)' if recorder else message)[:62]]
            canvas = present(displayed, cached["result"] if cached else None, shown_expected, model, args.backend, info)
            if auto_record and displayed is not None:
                recorder = Recorder(journal.path / (f'{args.backend}-{time.time_ns()}.mp4'), config["record_fps"], (canvas.shape[1], canvas.shape[0]))
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
                           mode="mock" if args.backend == "mock" else ("real_headless" if args.headless else "real"),
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
                updated = {c: cv2.getTrackbarPos("Expected " + c, title) for c in "12345"}
                if updated != expected:
                    expected = validate_counts(updated)
                    dirty_counts = True
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    detecting = not detecting
                elif key in map(ord, "1234"):
                    selected = MODELS[key-ord("1")]
                    # Load away from event loop using the same idle worker executor.
                    if not worker.busy and not loading:
                        loading = (selected, worker.pool.submit(load_backend, args.backend, selected, config))
                        message = "Loading " + selected
                elif key == ord("n") and args.backend == "mock":
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
                        for c in "12345":
                            cv2.setTrackbarPos("Expected " + c, title, values[c])
                elif key == ord("s"):
                    if dirty_counts or (cached and cached["expected"] != expected):
                        raise ValueError("Wait for count rejudgement before saving")
                    save_snapshot(canvas)
                elif key == ord("r"):
                    if recorder:
                        stop_recording()
                    else:
                        recorder = Recorder(journal.path / (f'{args.backend}-{time.time_ns()}.mp4'), config["record_fps"], (canvas.shape[1], canvas.shape[0]))
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
