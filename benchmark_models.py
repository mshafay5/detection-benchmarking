#!/usr/bin/env python3
"""Benchmark supported Ultralytics object detection models on a YOLO dataset."""

from __future__ import annotations

import argparse
import csv
import json
import logging
import re
import sys
import traceback
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import yaml
from ultralytics import RTDETR, YOLO


ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data.yaml"
LOCAL_SAMPLE_DATA = ROOT / "Dataset_Leaf_Yolo26" / "data_local.yaml"
DEFAULT_PROJECT = ROOT / "benchmark_runs"
DEFAULT_RESULTS_CSV = ROOT / "benchmark_results.csv"


@dataclass(frozen=True)
class ModelSpec:
    display_name: str
    pretrained: str | None
    scratch: str | None
    model_class: str
    supported: bool = True
    note: str = ""


DEFAULT_MODELS: list[ModelSpec] = [
    ModelSpec("YOLOv11x", "yolo11x.pt", "yolo11x.yaml", "yolo"),
    ModelSpec("YOLOv11l", "yolo11l.pt", "yolo11l.yaml", "yolo"),
    ModelSpec("YOLOv11m", "yolo11m.pt", "yolo11m.yaml", "yolo"),
    ModelSpec("YOLOv12x", "yolo12x.pt", "yolo12x.yaml", "yolo"),
    ModelSpec("YOLOv12l", "yolo12l.pt", "yolo12l.yaml", "yolo"),
    ModelSpec("YOLOv12m", "yolo12m.pt", "yolo12m.yaml", "yolo"),
    ModelSpec("YOLOv26x", "yolo26x.pt", "yolo26x.yaml", "yolo"),
    ModelSpec("YOLOv26l", "yolo26l.pt", "yolo26l.yaml", "yolo"),
    ModelSpec("YOLOv26m", "yolo26m.pt", "yolo26m.yaml", "yolo"),
    ModelSpec("RT-DETRx", "rtdetr-x.pt", "rtdetr-x.yaml", "rtdetr"),
    ModelSpec("RT-DETRl", "rtdetr-l.pt", "rtdetr-l.yaml", "rtdetr"),
    ModelSpec(
        "RT-DETR v2x",
        None,
        None,
        "rtdetr",
        supported=False,
        note=(
            "Unsupported by the installed Ultralytics build. "
            "Closest supported filename: rtdetr-x.pt."
        ),
    ),
    ModelSpec(
        "RT-DETR v2l",
        None,
        None,
        "rtdetr",
        supported=False,
        note=(
            "Unsupported by the installed Ultralytics build. "
            "Closest supported filename: rtdetr-l.pt."
        ),
    ),
]


RESULT_FIELDS = [
    "timestamp",
    "display_name",
    "model_file",
    "init",
    "status",
    "epochs",
    "imgsz",
    "batch",
    "device",
    "seed",
    "optimizer",
    "run_dir",
    "weights",
    "mAP@50",
    "mAP@50-95",
    "Precision",
    "Recall",
    "note",
]


def setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def parse_batch(value: str) -> int | float:
    if value.lower() == "auto":
        return -1
    if "." in value:
        return float(value)
    return int(value)


def split_path(dataset_root: Path, split_value: str) -> Path:
    path = Path(split_value)
    if path.is_absolute():
        return path
    return dataset_root / path


def validate_data_yaml(data_yaml: Path) -> dict[str, Any]:
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset YAML not found: {data_yaml}")

    with data_yaml.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    dataset_root = split_path(data_yaml.parent, str(data.get("path", data_yaml.parent)))
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    summary: dict[str, Any] = {
        "yaml": str(data_yaml),
        "dataset_root": str(dataset_root),
        "nc": data.get("nc"),
        "names": data.get("names"),
        "splits": {},
    }

    for yaml_key, folder_name in (("train", "train"), ("val", "valid"), ("test", "test")):
        split_value = data.get(yaml_key)
        if not split_value:
            raise ValueError(f"Missing '{yaml_key}' in dataset YAML")
        images_dir = split_path(dataset_root, str(split_value))
        labels_dir = images_dir.parent / "labels"
        if not images_dir.exists():
            raise FileNotFoundError(f"{yaml_key} images folder not found: {images_dir}")
        if not labels_dir.exists():
            raise FileNotFoundError(f"{folder_name} labels folder not found: {labels_dir}")

        image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in image_exts)
        labels = sorted(p for p in labels_dir.iterdir() if p.suffix.lower() == ".txt")
        missing_labels = sorted({p.stem for p in images} - {p.stem for p in labels})
        missing_images = sorted({p.stem for p in labels} - {p.stem for p in images})

        if missing_labels or missing_images:
            raise ValueError(
                f"{folder_name} split has missing pairs: "
                f"{len(missing_labels)} missing labels, {len(missing_images)} missing images"
            )

        summary["splits"][folder_name] = {
            "images": len(images),
            "labels": len(labels),
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
        }

    return summary


def select_models(model_filters: list[str] | None) -> list[ModelSpec]:
    if not model_filters:
        return DEFAULT_MODELS

    filters = {m.lower() for m in model_filters}
    selected = []
    for spec in DEFAULT_MODELS:
        names = {
            spec.display_name.lower(),
            (spec.pretrained or "").lower(),
            (spec.scratch or "").lower(),
            slugify(spec.display_name).lower(),
        }
        if names & filters:
            selected.append(spec)

    missing = filters - {
        item
        for spec in DEFAULT_MODELS
        for item in (
            spec.display_name.lower(),
            (spec.pretrained or "").lower(),
            (spec.scratch or "").lower(),
            slugify(spec.display_name).lower(),
        )
    }
    if missing:
        logging.warning("Unknown model filters ignored: %s", ", ".join(sorted(missing)))
    return selected


def load_existing_results(results_csv: Path) -> dict[str, dict[str, str]]:
    if not results_csv.exists():
        return {}
    with results_csv.open("r", newline="", encoding="utf-8") as f:
        return {row["display_name"]: row for row in csv.DictReader(f)}


def write_results(results_csv: Path, rows_by_model: dict[str, dict[str, Any]]) -> None:
    rows = [rows_by_model[key] for key in sorted(rows_by_model)]
    with results_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in RESULT_FIELDS})

    xlsx_path = results_csv.with_suffix(".xlsx")
    try:
        import pandas as pd

        pd.DataFrame(rows, columns=RESULT_FIELDS).to_excel(xlsx_path, index=False)
        logging.info("Wrote results: %s and %s", results_csv, xlsx_path)
    except Exception as exc:  # noqa: BLE001
        logging.warning("Wrote CSV but could not write XLSX: %s", exc)


def result_row(
    spec: ModelSpec,
    model_file: str,
    args: argparse.Namespace,
    status: str,
    run_dir: Path | None = None,
    weights: Path | None = None,
    metrics: dict[str, float] | None = None,
    note: str = "",
) -> dict[str, Any]:
    metrics = metrics or {}
    return {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "display_name": spec.display_name,
        "model_file": model_file,
        "init": args.init,
        "status": status,
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "run_dir": str(run_dir) if run_dir else "",
        "weights": str(weights) if weights else "",
        "mAP@50": metrics.get("mAP@50", ""),
        "mAP@50-95": metrics.get("mAP@50-95", ""),
        "Precision": metrics.get("Precision", ""),
        "Recall": metrics.get("Recall", ""),
        "note": note or spec.note,
    }


def model_source(spec: ModelSpec, init: str) -> str | None:
    return spec.pretrained if init == "pretrained" else spec.scratch


def model_constructor(spec: ModelSpec):
    return RTDETR if spec.model_class == "rtdetr" else YOLO


def is_completed(run_dir: Path, existing_row: dict[str, str] | None) -> bool:
    metric_file = run_dir / "benchmark_metrics.json"
    best = run_dir / "weights" / "best.pt"
    if metric_file.exists() and best.exists():
        return True
    return bool(existing_row and existing_row.get("status") == "completed")


def extract_metrics(metrics_obj: Any) -> dict[str, float]:
    box = getattr(metrics_obj, "box", None)
    if box is None:
        raise AttributeError("Validation result does not contain detection box metrics")
    return {
        "mAP@50": float(getattr(box, "map50")),
        "mAP@50-95": float(getattr(box, "map")),
        "Precision": float(getattr(box, "mp")),
        "Recall": float(getattr(box, "mr")),
    }


def train_and_validate(spec: ModelSpec, model_file: str, args: argparse.Namespace) -> dict[str, Any]:
    run_name = slugify(spec.display_name)
    run_dir = args.project / run_name
    constructor = model_constructor(spec)
    last_weights = run_dir / "weights" / "last.pt"

    if args.resume and last_weights.exists():
        logging.info("Resuming %s from %s", spec.display_name, last_weights)
        model = constructor(str(last_weights))
        resume = True
    else:
        logging.info("Starting %s with %s", spec.display_name, model_file)
        model = constructor(model_file)
        resume = False

    train_kwargs = {
        "data": str(args.data),
        "epochs": args.epochs,
        "imgsz": args.imgsz,
        "batch": args.batch,
        "device": args.device,
        "seed": args.seed,
        "optimizer": args.optimizer,
        "project": str(args.project),
        "name": run_name,
        "exist_ok": True,
        "patience": args.epochs,
        "workers": args.workers,
        "verbose": True,
        "resume": resume,
    }
    if args.cache:
        train_kwargs["cache"] = True

    model.train(**train_kwargs)

    best_weights = run_dir / "weights" / "best.pt"
    weights = best_weights if best_weights.exists() else last_weights
    if not weights.exists():
        raise FileNotFoundError(f"No trained weights found for {spec.display_name} in {run_dir}")

    logging.info("Validating %s with %s", spec.display_name, weights)
    val_model = constructor(str(weights))
    val_metrics = val_model.val(
        data=str(args.data),
        split=args.val_split,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(run_dir),
        name=f"val_{args.val_split}",
        exist_ok=True,
    )

    metrics = extract_metrics(val_metrics)
    metrics_path = run_dir / "benchmark_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump({"model": asdict(spec), "metrics": metrics}, f, indent=2)

    return result_row(
        spec,
        model_file,
        args,
        status="completed",
        run_dir=run_dir,
        weights=weights,
        metrics=metrics,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    default_data = DEFAULT_DATA if DEFAULT_DATA.exists() else LOCAL_SAMPLE_DATA
    parser.add_argument("--data", type=Path, default=default_data, help="YOLO dataset YAML.")
    parser.add_argument("--project", type=Path, default=DEFAULT_PROJECT, help="Folder for per-model runs.")
    parser.add_argument("--results-csv", type=Path, default=DEFAULT_RESULTS_CSV, help="Results CSV path.")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for every supported model.")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size for training and validation.")
    parser.add_argument("--batch", type=parse_batch, default=8, help="Batch size, or 'auto'.")
    parser.add_argument("--device", default=None, help="Ultralytics device string. Default: 0 if CUDA is available, else cpu.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--optimizer", default="auto", help="Optimizer setting passed to Ultralytics.")
    parser.add_argument("--workers", type=int, default=8, help="Data loader workers.")
    parser.add_argument("--val-split", default="val", choices=["val", "test"], help="Split used for final metrics.")
    parser.add_argument("--init", choices=["pretrained", "scratch"], default="pretrained", help="Use .pt weights or local .yaml configs.")
    parser.add_argument("--models", nargs="*", help="Optional subset by display name or filename.")
    parser.add_argument("--no-resume", dest="resume", action="store_false", help="Do not resume incomplete runs from last.pt.")
    parser.add_argument("--force", action="store_true", help="Retrain even if a model is already completed.")
    parser.add_argument("--cache", action="store_true", help="Enable Ultralytics dataset caching.")
    parser.add_argument("--dry-run", action="store_true", help="Validate configuration and print the planned runs without training.")
    parser.add_argument("--verbose", action="store_true", help="Enable debug logging.")
    parser.set_defaults(resume=True)
    args = parser.parse_args()

    args.data = args.data.resolve()
    args.project = args.project.resolve()
    args.results_csv = args.results_csv.resolve()
    if args.device is None:
        args.device = "0" if torch.cuda.is_available() else "cpu"
    return args


def main() -> int:
    args = parse_args()
    setup_logging(args.verbose)

    logging.info("Python: %s", sys.executable)
    logging.info("Torch: %s | CUDA available: %s", torch.__version__, torch.cuda.is_available())
    if torch.cuda.is_available():
        logging.info("CUDA device 0: %s", torch.cuda.get_device_name(0))

    dataset_summary = validate_data_yaml(args.data)
    logging.info("Dataset YAML: %s", dataset_summary["yaml"])
    logging.info("Dataset root: %s", dataset_summary["dataset_root"])
    for split_name, split_info in dataset_summary["splits"].items():
        logging.info(
            "%s split: %s images, %s labels",
            split_name,
            split_info["images"],
            split_info["labels"],
        )

    models = select_models(args.models)
    if not models:
        raise ValueError("No models selected")

    rows_by_model = load_existing_results(args.results_csv)
    args.project.mkdir(parents=True, exist_ok=True)

    for spec in models:
        model_file = model_source(spec, args.init)
        run_dir = args.project / slugify(spec.display_name)
        existing_row = rows_by_model.get(spec.display_name)

        if not spec.supported or not model_file:
            logging.warning("%s is skipped: %s", spec.display_name, spec.note)
            rows_by_model[spec.display_name] = result_row(
                spec,
                model_file or "",
                args,
                status="unsupported",
                run_dir=run_dir,
                note=spec.note,
            )
            write_results(args.results_csv, rows_by_model)
            continue

        if is_completed(run_dir, existing_row) and not args.force:
            logging.info("%s already completed; skipping. Use --force to retrain.", spec.display_name)
            continue

        if args.dry_run:
            logging.info("Dry run: would train %s using %s into %s", spec.display_name, model_file, run_dir)
            rows_by_model[spec.display_name] = result_row(
                spec,
                model_file,
                args,
                status="planned",
                run_dir=run_dir,
                note="Dry run only; training was not started.",
            )
            continue

        try:
            rows_by_model[spec.display_name] = train_and_validate(spec, model_file, args)
        except Exception as exc:  # noqa: BLE001
            logging.error("%s failed: %s", spec.display_name, exc)
            logging.debug("Traceback:\n%s", traceback.format_exc())
            rows_by_model[spec.display_name] = result_row(
                spec,
                model_file,
                args,
                status="failed",
                run_dir=run_dir,
                note=str(exc),
            )
        finally:
            write_results(args.results_csv, rows_by_model)

    if args.dry_run:
        write_results(args.results_csv, rows_by_model)
        logging.info("Dry run complete. No training was started.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
