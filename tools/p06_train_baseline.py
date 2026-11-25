"""Accepted P03 Train/Validation: explicit dry-run, official load check, smoke or training."""
import argparse
from collections import Counter
from contextlib import chdir, redirect_stdout, redirect_stderr
import csv
from importlib.metadata import version
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.data.p06_labelme import CLASSES
from tools.p06_build_dataset import check_sources, validate_artifacts, validate_project
from tools.p06_preview_annotations import digest, font

DATA = ROOT / "data/p03/development_v2"
ACCEPTANCE = ROOT / "output/p06_acceptance_development_v2"
RUNTIME = ROOT / "output/p06_runtime_p03"
WEIGHTS = ROOT / "output/p06-cache/yolo11n.pt"
RUN = ROOT / "output/p06_runs/p03_baseline3_640_seed42"
PARAMS = dict(epochs=100, imgsz=640, batch=8, patience=30, device=0, seed=42,
              workers=4, cache=False, deterministic=True, amp=True, val=True, save=True,
              plots=True, pretrained=True)
VERSIONS = {"numpy":"1.26.4", "opencv-python":"4.10.0.84", "torch":"2.7.1+cu126",
            "torchvision":"0.22.1+cu126", "ultralytics":"8.3.236", "Pillow":"10.4.0"}
WEIGHT_HASH = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def validate_yaml(config):
    expected = dict(path=".", train="images/train", val="images/val", nc=6,
                    names=dict(enumerate(CLASSES)))
    require(config == expected, "dataset.yaml paths or six-class mapping differ from the published contract; test/download keys are forbidden.")


def inspect_environment(torch):
    installed = {name: version(name) for name in VERSIONS}
    require(Path(sys.prefix).name == "ai-p06-yolo" and sys.version_info[:3] == (3,11,18),
            "Use ai-p06-yolo with Python 3.11.18.")
    require(installed == VERSIONS, f"Unexpected package versions: {installed}")
    require(torch.version.cuda == "12.6" and torch.cuda.is_available(), "CUDA 12.6 unavailable; training refused.")
    device = torch.cuda.get_device_properties(0)
    require("RTX 4060 Laptop" in device.name, f"Unexpected CUDA device: {device.name}")
    tensor = torch.tensor([1.,2.,3.], device="cuda:0").square().sum()
    require(tensor.is_cuda and tensor.item() == 14, "CUDA tensor check failed.")
    return dict(python=sys.version.split()[0], executable=sys.executable, packages=installed,
                cuda=torch.version.cuda, gpu=device.name, total_vram_bytes=device.total_memory,
                cuda_tensor_result=tensor.item())


def is_label_cache(path, data):
    return path in (data/"labels/train.cache", data/"labels/val.cache")


def snapshot(data):
    """Protect every published file except the two official regenerable label caches."""
    return {p.relative_to(data).as_posix(): digest(p) for p in sorted(data.rglob("*"))
            if p.is_file() and not is_label_cache(p, data)}


def training_params(smoke=False):
    return dict(PARAMS, epochs=1 if smoke else 100)


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, value):
        self.console.write(value)
        self.log.write(value)
        self.log.flush()
        return len(value)

    def flush(self):
        self.console.flush()
        self.log.flush()

    def isatty(self):
        return False



def preflight(data, weights, run):
    import torch
    from tools.p06_accepted_data import accepted_plan
    require(not run.exists(), f"Run directory already exists: {run}")
    require(data.resolve() == DATA and weights.resolve() == WEIGHTS,
            "Use the published development dataset and pinned official yolo11n.pt; Test and legacy inputs are forbidden.")
    require(weights.is_file() and digest(weights) == WEIGHT_HASH, "Missing or unexpected official pretrained weights")
    branch = subprocess.check_output(["git","-C",str(ROOT),"branch","--show-current"], text=True).strip()
    require(branch == "feat/p06-baseline3", f"Unexpected branch: {branch}")
    environment = inspect_environment(torch)
    plan = accepted_plan(data, ACCEPTANCE)
    return plan, dict(environment=environment, branch=branch, weights_sha256=WEIGHT_HASH,
        data=str(data), source=str(data), summary=plan["summary"], dataset_sha256=snapshot(data),
        acceptance_sha256=plan["acceptance_sha256"], limitations=plan["limitations"],
        git_status=subprocess.check_output(["git","-C",str(ROOT),"status","--short"],text=True))


def configure_runtime():
    cache = ROOT/"output/p06-cache/baseline-runtime"
    config = cache/"Ultralytics"
    config.mkdir(parents=True, exist_ok=True)
    os.environ.update(YOLO_CONFIG_DIR=str(cache), MPLCONFIGDIR=str(cache/"matplotlib"), YOLO_OFFLINE="true")
    (cache/"matplotlib").mkdir(exist_ok=True)
    # check_det_dataset searches this local cache for Arial.
    import shutil
    if not (config/"Arial.ttf").exists():
        shutil.copyfile(Path(os.environ["WINDIR"])/"Fonts/arial.ttf", config/"Arial.ttf")


def save_json(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False)+"\n", encoding="utf-8")


def validate_best(model, data, run, plan, *, imgsz=640):
    def verify_device(validator):
        require(validator.device.type == "cuda", "Independent validation device was not CUDA.")

    seen = {}
    model.add_callback("on_val_start", verify_device)
    model.add_callback("on_val_end", lambda validator: seen.update(images=int(validator.seen)))
    metrics = model.val(data=str(data/"dataset.yaml"), split="val", imgsz=imgsz, batch=8,
                        device=0, workers=4, plots=True, agnostic_nms=False, project=str(run), name="best_validation")
    supports = plan["summary"]["val"]["classes"]
    require(seen.get("images") == plan["summary"]["val"]["images"] and list(map(int, metrics.nt_per_class)) == supports, "Independent validation support differs from accepted split")
    indices = {int(cls): i for i,cls in enumerate(metrics.box.ap_class_index)}
    classes = []
    for cid, name in enumerate(CLASSES):
        row = dict(class_id=cid, name=name, gt=supports[cid], precision=None, recall=None, ap50=None, ap50_95=None)
        if supports[cid] and cid in indices:
            row.update(zip(("precision","recall","ap50","ap50_95"), map(float,metrics.box.class_result(indices[cid]))))
        classes.append(row)
    matrix = metrics.confusion_matrix.matrix.astype(int).tolist()
    result = dict(zip(("precision","recall","map50","map50_95"),map(float,metrics.box.mean_results())))
    result.update(classes=classes, matrix=matrix, matrix_axis="rows predicted, columns GT, last index background",
                  id0_false_positives=sum(matrix[0][1:]), device="cuda:0", save_dir=str(run/"best_validation"))
    save_json(run/"best_validation/metrics.json",result)
    return result


def iou(a, b):
    intersection = max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = (a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
    return intersection/union if union else 0


def compare_predictions(predictions, truth):
    """Greedy confidence-order, class-agnostic IoU>=0.5 diagnostic, not official AP."""
    available = set(range(len(truth)))
    confusions, extras = [], []
    for pred in sorted(predictions,key=lambda row: -row["confidence"]):
        best = max(sorted(available), key=lambda idx:iou(pred["xyxy"],truth[idx]["xyxy"]), default=None)
        if best is not None and iou(pred["xyxy"],truth[best]["xyxy"]) >= .5:
            available.remove(best)
            if pred["class_id"] != truth[best]["class_id"]:
                confusions.append([truth[best]["class_id"],pred["class_id"]])
        else:
            extras.append({"class_id":pred["class_id"],
                           "possible_duplicate":any(iou(pred["xyxy"],obj["xyxy"])>=.5 for obj in truth)})
    return dict(confusions_gt_to_pred=confusions, unmatched_gt=[truth[i]["class_id"] for i in sorted(available)],
                unmatched_predictions=extras)


def predict_val(model, data, run, plan, *, imgsz=640):
    from PIL import Image, ImageOps, ImageDraw
    out = run/"val_predictions"
    out.mkdir()
    rows = []
    for row in plan["records"]:
        if row["split"] != "val":
            continue
        path = data/"images/val"/row["png_name"]
        result, = model.predict(str(path), imgsz=imgsz, conf=.25, iou=.7, device=0, agnostic_nms=False, verbose=False)
        require(result.boxes.data.is_cuda, f"Prediction not on CUDA: {path.name}")
        require(result.orig_shape == (row["height"],row["width"]), f"Wrong prediction dimensions: {path.name}")
        predictions = [dict(xyxy=box[:4],confidence=box[4],class_id=int(box[5]))
                       for box in result.boxes.data.cpu().tolist()]
        with Image.fromarray(result.plot()[:,:,::-1]) as image:
            preview = ImageOps.contain(image,(800,800))
            preview.save(out/path.with_suffix(".jpg").name,quality=85)
            preview.close()
        counts = Counter(p["class_id"] for p in predictions)
        rows.append(dict(image_id=row["image_id"],image_level_source_status=row["image_status"],
                         detections=len(predictions),class_counts=[counts[i] for i in range(6)],
                         predictions=predictions,**compare_predictions(predictions,row["objects"])))
    save_json(out/"prediction_summary.json",rows)
    with (out/"prediction_summary.csv").open("w",encoding="utf-8-sig",newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["image_id","detections",*CLASSES,"unmatched_gt","confusions_gt_to_pred","unmatched_predictions"])
        for row in rows:
            writer.writerow([row["image_id"],row["detections"],*row["class_counts"],
                             row["unmatched_gt"],row["confusions_gt_to_pred"],row["unmatched_predictions"]])
    for start in range(0,len(rows),16):
        sheet = Image.new("RGB",(1600,1900),"#eef1f5")
        draw = ImageDraw.Draw(sheet)
        draw.text((15,10),"MODEL predictions | conf .25 IoU .7 | agnostic NMS False",font=font(22),fill="black")
        for index,row in enumerate(rows[start:start+16]):
            x,y=index%4*400,index//4*450+65
            with Image.open(out/Path(row["image_id"]).with_suffix(".jpg")) as image:
                thumb=ImageOps.contain(image,(390,390))
                sheet.paste(thumb,(x+(400-thumb.width)//2,y+30))
            draw.text((x+5,y),f'{row["image_id"]} | {row["detections"]}',font=font(18),fill="black")
        sheet.save(out/f"contact_{start//16+1:02}.jpg",quality=85)
        sheet.close()
    return rows


def write_report(run, record, metrics, rows):
    lines=["# P06 accepted P03 workflow", "", "Engineering smoke only." if record.get("smoke") else "P03 baseline experiment.",
        "Data summary: "+json.dumps(record["summary"]), "Parameters: "+json.dumps(record["parameters"]),
        "Training: "+json.dumps(record["training"]),
        "Independent best.pt validation: "+json.dumps(metrics),
        "Validation counts: "+json.dumps([dict(image_id=r["image_id"],counts=r["class_counts"]) for r in rows]),
        "Preview JPEGs are display-only thumbnails (maximum edge 800), not training copies.",
        "No Test access. Cross-Test leakage/historical exposure remain unverified by P06.",
        "Provisional scene IDs are based on user confirmation; P03 CSV is unchanged.",
        "One epoch verifies execution, backward updates and checkpoint reload; it does not prove recognition/rejection quality."]
    (run/"training_report.md").write_text("\n\n".join(lines),encoding="utf-8")


def train(data, weights, run, plan, record, params):
    import torch
    from ultralytics import YOLO
    from ultralytics.utils import LOGGER, SETTINGS
    # Update only in-memory settings; the AMP check resolves the same verified weight.
    dict.__setitem__(SETTINGS,"weights_dir",str(weights.parent))
    run.mkdir(parents=True,exist_ok=False)
    save_json(run/"preflight.json",record)
    save_json(run/"fixed_params.json",params)
    handler = logging.FileHandler(run/"run.log",encoding="utf-8")
    LOGGER.addHandler(handler)
    state = {}
    handles = []
    tracked = {}


    def started(trainer):
        require(trainer.device.type=="cuda" and bool(trainer.amp), "Trainer did not activate CUDA and AMP.")
        state.update(device=str(trainer.device), amp=bool(trainer.amp), gradient_calls=0,
                     gradient_finite=True, gradient_nonzero=False)
        name, parameter = next((n, p) for n, p in reversed(list(trainer.model.named_parameters()))
                               if p.requires_grad and p.ndim > 1)
        tracked.update(parameter=parameter, before=parameter.detach().clone())
        state["tracked_parameter"] = name

        def observe(gradient):
            state["gradient_calls"] += 1
            state["gradient_finite"] &= bool(torch.isfinite(gradient).all().item())
            state["gradient_nonzero"] |= bool(torch.count_nonzero(gradient).item())
            # Observation only: returning None leaves the gradient untouched.
        handles.append(parameter.register_hook(observe))

    def epoch_finished(trainer):
        state["parameter_max_abs_change"] = float(
            (tracked["parameter"].detach()-tracked["before"]).abs().max().item())
        require(state["gradient_calls"] > 0 and state["gradient_finite"] and
                state["gradient_nonzero"] and state["parameter_max_abs_change"] > 0,
                "Missing finite nonzero backward/parameter update evidence.")


    def checkpoint(trainer):
        if trainer.best_fitness == trainer.fitness:
            state["best_epoch"] = trainer.epoch+1
        save_json(run/"progress.json",dict(state,epochs_completed=trainer.epoch+1))

    try:
        # cache=False disables pixel caching; official label caches remain enabled.
        with chdir(data):
            model = YOLO(str(weights))
            require(model.task=="detect" and len(model.names)==80 and model.names[0]=="person","Unexpected pretrained checkpoint.")
            model.add_callback("on_train_start",started)
            model.add_callback("on_model_save",checkpoint)
            model.add_callback("on_train_epoch_end",epoch_finished)
            torch.cuda.reset_peak_memory_stats(0)
            start = perf_counter()
            # The directory was just exclusively reserved by this invocation.
            model.train(data=str(data/"dataset.yaml"),project=str(run.parent),name=run.name,exist_ok=True,**params)
            for handle in handles:
                handle.remove()
            elapsed = perf_counter()-start
            peak = torch.cuda.max_memory_allocated(0)
            with (run/"results.csv").open(newline="") as stream:
                epochs = len(list(csv.DictReader(stream)))
            best = run/"weights/best.pt"
            require(best.is_file() and (run/"weights/last.pt").is_file() and state.get("best_epoch"),"Missing checkpoints or best epoch evidence.")
            record["training"] = dict(state,epochs_completed=epochs,early_stop=epochs<params["epochs"],
                                      seconds=elapsed,peak_allocated_bytes=peak,best_weight=str(best),best_weight_sha256=digest(best))
            require(epochs == params["epochs"] or epochs > 0, "No epoch completed.")
            with (run/"results.csv").open(newline="") as stream:
                import math
                require(all(math.isfinite(float(value)) for row in csv.DictReader(stream)
                            for value in row.values()), "Non-finite training results.")
            save_json(run/"training_state.json",record["training"])
            del model
            torch.cuda.empty_cache()
            evaluation = YOLO(str(best))
            require(list(evaluation.names.values()) == list(CLASSES),"Best checkpoint class order changed.")
            metrics = validate_best(evaluation,data,run,plan)
            rows = predict_val(evaluation,data,run,plan)
            require(snapshot(data)==record["dataset_sha256"],"Published dataset changed during run.")
            from tools.p06_accepted_data import check_sources as check_accepted_sources
            check_accepted_sources(plan)
            write_report(run,record,metrics,rows)
            save_json(run/"completion.json",dict(status="PASS",dataset_unchanged=True,
                validation_images=len(rows),training=record["training"],metrics=metrics))
    except BaseException:
        (run/"failure.log").write_text(traceback.format_exc(),encoding="utf-8")
        raise
    finally:
        for handle in handles:
            handle.remove()
        LOGGER.removeHandler(handler)
        handler.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run",action="store_true",help="Read-only checks; may be combined with --smoke")
    mode=parser.add_mutually_exclusive_group()
    mode.add_argument("--check-data",action="store_true")
    mode.add_argument("--smoke",action="store_true",help="Exactly one epoch")
    mode.add_argument("--train",action="store_true",help="Explicit future 100-epoch training; requires separate authorization")
    parser.add_argument("--dataset",type=Path,default=DATA)
    parser.add_argument("--output-dir",type=Path, help="New result directory; existing paths are refused.")
    args = parser.parse_args()
    if not (args.dry_run or args.check_data or args.smoke or args.train):
        parser.error("Choose --dry-run, --check-data, --smoke or explicitly --train")
    try:
        data = args.dataset.resolve()
        run = (args.output_dir or (RUN.parent/"p03_smoke_1epoch_640_seed42" if args.smoke else RUN)).resolve()
        require(run.is_relative_to(ROOT/"output/p06_runs"), "Output must stay under output/p06_runs.")
        plan, record = preflight(data,WEIGHTS,run)
        params = training_params(args.smoke)
        record.update(parameters=params, run=str(run), smoke=args.smoke)
        print(json.dumps({k:v for k,v in record.items() if k!="dataset_sha256"},indent=2))
        if not args.dry_run:
            configure_runtime()
            from tools.p06_accepted_data import prepare_runtime
            loading=prepare_runtime(plan,RUNTIME)
            record.update(data=str(RUNTIME),runtime_storage="NTFS hardlinks for immutable image consumption; copied small labels",official_loader=loading,dataset_sha256=snapshot(RUNTIME))
            if args.check_data:
                print(json.dumps(loading,indent=2))
                return 0
            # Exclusive reservation occurs in train(); never truncate an existing log.
            run.parent.mkdir(parents=True,exist_ok=True)
            log_path = run.parent/(run.name+"-console.log")
            with log_path.open("x",encoding="utf-8") as log:
                with redirect_stdout(Tee(sys.stdout,log)), redirect_stderr(Tee(sys.stderr,log)):
                    train(RUNTIME,WEIGHTS,run,plan,record,params)
        return 0
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}",file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
