#!/usr/bin/env python3

from __future__ import annotations

import csv
import inspect
import json
import random
import shutil
import traceback
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import torch
import yaml
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm

from transformers import (
    RTDetrForObjectDetection,
    RTDetrImageProcessor,
    Trainer,
    TrainingArguments,
)

try:
    from torchmetrics.detection.mean_ap import MeanAveragePrecision
    TORCHMETRICS_AVAILABLE = True
except Exception:
    TORCHMETRICS_AVAILABLE = False


# ============================================================
# HARD-CODED SETTINGS
# ============================================================

ROOT = Path(__file__).resolve().parent

DATA_YAML = ROOT / "Dataset_Leaf_Yolo26" / "data_hf.yaml"

PROJECT_DIR = ROOT / "hf_rtdetr_benchmark_runs"
SUMMARY_CSV = ROOT / "hf_rtdetr_benchmark_results.csv"
SUMMARY_XLSX = ROOT / "hf_rtdetr_benchmark_results.xlsx"

INFERENCE_IMAGE_FOLDER = ROOT / "smartGlasses" / "pic"

EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 2
GRADIENT_ACCUMULATION_STEPS = 4
LEARNING_RATE = 1e-5
WEIGHT_DECAY = 1e-4
WARMUP_RATIO = 0.05
NUM_WORKERS = 4
SEED = 42
FP16 = True

CONF_THRESHOLD = 0.35
MAP_THRESHOLD = 0.001
MAX_INFERENCE_IMAGES = 6

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ============================================================
# MODELS TO RUN
# Model #1 runs first, then model #2, then model #3...
# ============================================================

MODEL_SPECS = [
    {
        "display_name": "RT-DETR-r18vd",
        "checkpoint": "PekingU/rtdetr_r18vd",
    },
    {
        "display_name": "RT-DETR-r34vd",
        "checkpoint": "PekingU/rtdetr_r34vd",
    },
    {
        "display_name": "RT-DETR-r50vd",
        "checkpoint": "PekingU/rtdetr_r50vd",
    },
    {
        "display_name": "RT-DETR-r101vd",
        "checkpoint": "PekingU/rtdetr_r101vd",
    },
]


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


SUMMARY_FIELDS = [
    "timestamp",
    "display_name",
    "checkpoint",
    "status",
    "epochs",
    "imgsz",
    "batch",
    "gradient_accumulation_steps",
    "effective_batch",
    "device",
    "seed",
    "run_dir",
    "weights",
    "mAP@50",
    "mAP@50-95",
    "Precision",
    "Recall",
    "num_val_images",
    "num_inference_images",
    "note",
]


# ============================================================
# GENERAL HELPERS
# ============================================================

def slugify(value: str) -> str:
    return (
        value.lower()
        .replace("-", "_")
        .replace(" ", "_")
        .replace("/", "_")
    )


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_data_yaml():
    """
    Creates a Linux-safe data_hf.yaml if it does not exist.
    """
    if DATA_YAML.exists():
        return

    DATA_YAML.parent.mkdir(parents=True, exist_ok=True)

    content = {
        "path": str(ROOT / "Dataset_Leaf_Yolo26"),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "nc": 1,
        "names": {0: "leaf"},
    }

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(content, f, sort_keys=False)

    print(f"Created: {DATA_YAML}")


def read_data_yaml(data_yaml: Path) -> dict[str, Any]:
    with open(data_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    dataset_root = Path(data.get("path", data_yaml.parent))

    if not dataset_root.is_absolute():
        dataset_root = data_yaml.parent / dataset_root

    names = data["names"]

    if isinstance(names, dict):
        id2label = {int(k): str(v) for k, v in names.items()}
    else:
        id2label = {i: str(name) for i, name in enumerate(names)}

    label2id = {v: k for k, v in id2label.items()}

    data["dataset_root"] = dataset_root
    data["id2label"] = id2label
    data["label2id"] = label2id
    data["num_classes"] = len(id2label)

    return data


def resolve_split_dir(dataset_root: Path, split_value: str) -> Path:
    split_path = Path(split_value)

    if split_path.is_absolute():
        return split_path

    return dataset_root / split_path


def find_label_path(image_path: Path) -> Path:
    parts = list(image_path.parts)

    if "images" in parts:
        idx = parts.index("images")
        parts[idx] = "labels"
        return Path(*parts).with_suffix(".txt")

    return image_path.parent.parent / "labels" / f"{image_path.stem}.txt"


def yolo_to_xyxy(
    yolo_values: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    x_center, y_center, box_width, box_height = yolo_values

    x_center *= image_width
    y_center *= image_height
    box_width *= image_width
    box_height *= image_height

    x1 = x_center - box_width / 2
    y1 = y_center - box_height / 2
    x2 = x_center + box_width / 2
    y2 = y_center + box_height / 2

    x1 = max(0.0, x1)
    y1 = max(0.0, y1)
    x2 = min(float(image_width), x2)
    y2 = min(float(image_height), y2)

    return [x1, y1, x2, y2]


def yolo_to_coco_bbox(
    yolo_values: list[float],
    image_width: int,
    image_height: int,
) -> list[float]:
    x1, y1, x2, y2 = yolo_to_xyxy(
        yolo_values,
        image_width=image_width,
        image_height=image_height,
    )

    width = max(1.0, x2 - x1)
    height = max(1.0, y2 - y1)

    return [x1, y1, width, height]


def read_yolo_labels_xyxy(image_path: Path) -> tuple[list[list[float]], list[int]]:
    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    label_path = find_label_path(image_path)

    boxes = []
    labels = []

    if not label_path.exists():
        return boxes, labels

    with open(label_path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    for line in lines:
        parts = line.strip().split()

        if len(parts) < 5:
            continue

        class_id = int(float(parts[0]))
        yolo_values = [float(v) for v in parts[1:5]]

        box = yolo_to_xyxy(
            yolo_values,
            image_width=width,
            image_height=height,
        )

        boxes.append(box)
        labels.append(class_id)

    return boxes, labels


def get_split_image_paths(data_yaml: Path, split: str) -> list[Path]:
    data = read_data_yaml(data_yaml)
    dataset_root = data["dataset_root"]

    split_key = split

    if split == "val" and "val" not in data and "valid" in data:
        split_key = "valid"

    split_dir = resolve_split_dir(dataset_root, str(data[split_key]))

    image_paths = sorted(
        p for p in split_dir.rglob("*")
        if p.suffix.lower() in IMAGE_EXTS
    )

    return image_paths


def validate_dataset():
    data = read_data_yaml(DATA_YAML)
    dataset_root = data["dataset_root"]

    print(f"Dataset YAML: {DATA_YAML}")
    print(f"Dataset root: {dataset_root}")

    for split in ["train", "val", "test"]:
        if split not in data:
            print(f"Split missing in YAML: {split}")
            continue

        image_dir = resolve_split_dir(dataset_root, str(data[split]))

        if not image_dir.exists():
            raise FileNotFoundError(f"{split} images folder not found: {image_dir}")

        image_paths = sorted(
            p for p in image_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        )

        print(f"{split}: {len(image_paths)} images at {image_dir}")


# ============================================================
# DATASET FOR HUGGING FACE RT-DETR TRAINING
# ============================================================

class YoloDetectionDataset(Dataset):
    def __init__(
        self,
        data_yaml: Path,
        split: str,
        image_processor: RTDetrImageProcessor,
    ):
        self.data_yaml = Path(data_yaml)
        self.data = read_data_yaml(self.data_yaml)
        self.dataset_root = self.data["dataset_root"]
        self.image_processor = image_processor

        split_key = split

        if split == "val" and "val" not in self.data and "valid" in self.data:
            split_key = "valid"

        if split_key not in self.data:
            raise ValueError(f"Split '{split}' not found in YAML.")

        self.images_dir = resolve_split_dir(
            self.dataset_root,
            str(self.data[split_key]),
        )

        if not self.images_dir.exists():
            raise FileNotFoundError(f"Images folder not found: {self.images_dir}")

        self.image_paths = sorted(
            p for p in self.images_dir.rglob("*")
            if p.suffix.lower() in IMAGE_EXTS
        )

        if not self.image_paths:
            raise FileNotFoundError(f"No images found in: {self.images_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx: int):
        image_path = self.image_paths[idx]
        label_path = find_label_path(image_path)

        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        annotations = []

        if label_path.exists():
            with open(label_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            for ann_id, line in enumerate(lines):
                parts = line.strip().split()

                if len(parts) < 5:
                    continue

                class_id = int(float(parts[0]))
                yolo_values = [float(v) for v in parts[1:5]]

                bbox = yolo_to_coco_bbox(
                    yolo_values,
                    image_width=width,
                    image_height=height,
                )

                area = bbox[2] * bbox[3]

                annotations.append(
                    {
                        "id": ann_id,
                        "image_id": idx,
                        "category_id": class_id,
                        "bbox": bbox,
                        "area": area,
                        "iscrowd": 0,
                    }
                )

        target = {
            "image_id": idx,
            "annotations": annotations,
        }

        encoding = self.image_processor(
            images=image,
            annotations=target,
            return_tensors="pt",
        )

        return {
            "pixel_values": encoding["pixel_values"].squeeze(0),
            "labels": encoding["labels"][0],
        }


@dataclass
class RTDETRCollator:
    image_processor: RTDetrImageProcessor

    def __call__(self, batch):
        pixel_values = [item["pixel_values"] for item in batch]
        labels = [item["labels"] for item in batch]

        encoding = self.image_processor.pad(
            pixel_values,
            return_tensors="pt",
        )

        output = {
            "pixel_values": encoding["pixel_values"],
            "labels": labels,
        }

        if "pixel_mask" in encoding and encoding["pixel_mask"] is not None:
            output["pixel_mask"] = encoding["pixel_mask"]

        return output


# ============================================================
# VISUALIZATION
# ============================================================

def draw_boxes(
    image_bgr: np.ndarray,
    boxes: list[list[float]],
    labels: list[int],
    scores: list[float] | None,
    id2label: dict[int, str],
    count_text: str | None = None,
    color: tuple[int, int, int] = (255, 0, 0),
) -> np.ndarray:
    output = image_bgr.copy()
    white = (255, 255, 255)

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [int(v) for v in box]
        label_id = int(labels[i])
        label_name = id2label.get(label_id, str(label_id))

        if scores is not None:
            text = f"{label_name} {float(scores[i]):.2f}"
        else:
            text = f"{label_name}"

        cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

        text_size, _ = cv2.getTextSize(
            text,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            2,
        )

        text_w, text_h = text_size
        y_text_top = max(0, y1 - text_h - 8)

        cv2.rectangle(
            output,
            (x1, y_text_top),
            (x1 + text_w + 6, y1),
            color,
            -1,
        )

        cv2.putText(
            output,
            text,
            (x1 + 3, y1 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            white,
            2,
        )

    if count_text is not None:
        cv2.rectangle(output, (10, 10), (280, 55), color, -1)

        cv2.putText(
            output,
            count_text,
            (20, 42),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            white,
            2,
        )

    return output


def save_gt_visuals(
    image_paths: list[Path],
    run_dir: Path,
    id2label: dict[int, str],
    prefix: str,
    max_images: int = 3,
):
    selected = image_paths[:max_images]

    for idx, image_path in enumerate(selected):
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            continue

        boxes, labels = read_yolo_labels_xyxy(image_path)

        annotated = draw_boxes(
            image_bgr=image_bgr,
            boxes=boxes,
            labels=labels,
            scores=None,
            id2label=id2label,
            count_text=f"Labels: {len(boxes)}",
            color=(0, 255, 0),
        )

        output_path = run_dir / f"{prefix}_batch{idx}_labels.jpg"
        cv2.imwrite(str(output_path), annotated)


def run_model_on_image(
    model: RTDetrForObjectDetection,
    image_processor: RTDetrImageProcessor,
    image_path: Path,
    device: str,
    threshold: float,
):
    image_pil = Image.open(image_path).convert("RGB")

    inputs = image_processor(
        images=image_pil,
        return_tensors="pt",
    ).to(device)

    with torch.no_grad():
        outputs = model(**inputs)

    target_sizes = torch.tensor(
        [(image_pil.height, image_pil.width)],
        device=device,
    )

    results = image_processor.post_process_object_detection(
        outputs,
        target_sizes=target_sizes,
        threshold=threshold,
    )

    result = results[0]

    boxes = result["boxes"].detach().cpu().tolist()
    scores = result["scores"].detach().cpu().tolist()
    labels = result["labels"].detach().cpu().tolist()

    return boxes, scores, labels


def save_prediction_visuals(
    model: RTDetrForObjectDetection,
    image_processor: RTDetrImageProcessor,
    image_paths: list[Path],
    run_dir: Path,
    id2label: dict[int, str],
    prefix: str,
    max_images: int = 3,
):
    selected = image_paths[:max_images]

    for idx, image_path in enumerate(selected):
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            continue

        boxes, scores, labels = run_model_on_image(
            model=model,
            image_processor=image_processor,
            image_path=image_path,
            device=DEVICE,
            threshold=CONF_THRESHOLD,
        )

        annotated = draw_boxes(
            image_bgr=image_bgr,
            boxes=boxes,
            labels=labels,
            scores=scores,
            id2label=id2label,
            count_text=f"Pred: {len(boxes)}",
            color=(255, 0, 0),
        )

        output_path = run_dir / f"{prefix}_batch{idx}_pred.jpg"
        cv2.imwrite(str(output_path), annotated)


def save_smartglasses_inference(
    model: RTDetrForObjectDetection,
    image_processor: RTDetrImageProcessor,
    run_dir: Path,
    id2label: dict[int, str],
):
    output_dir = run_dir / "inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    if not INFERENCE_IMAGE_FOLDER.exists():
        print(f"Inference folder not found, skipping: {INFERENCE_IMAGE_FOLDER}")
        return 0

    image_paths = sorted(
        p for p in INFERENCE_IMAGE_FOLDER.iterdir()
        if p.suffix.lower() in IMAGE_EXTS
    )

    image_paths = image_paths[:MAX_INFERENCE_IMAGES]

    if not image_paths:
        print(f"No inference images found in: {INFERENCE_IMAGE_FOLDER}")
        return 0

    rows = []

    for image_path in image_paths:
        image_bgr = cv2.imread(str(image_path))

        if image_bgr is None:
            continue

        boxes, scores, labels = run_model_on_image(
            model=model,
            image_processor=image_processor,
            image_path=image_path,
            device=DEVICE,
            threshold=CONF_THRESHOLD,
        )

        annotated = draw_boxes(
            image_bgr=image_bgr,
            boxes=boxes,
            labels=labels,
            scores=scores,
            id2label=id2label,
            count_text=f"Count: {len(boxes)}",
            color=(255, 0, 0),
        )

        output_path = output_dir / f"{image_path.stem}_count_{len(boxes)}.jpg"
        cv2.imwrite(str(output_path), annotated)

        rows.append(
            {
                "image": image_path.name,
                "detections": len(boxes),
                "conf_threshold": CONF_THRESHOLD,
                "output": str(output_path),
            }
        )

    inference_csv = output_dir / "inference_results.csv"
    inference_xlsx = output_dir / "inference_results.xlsx"

    pd.DataFrame(rows).to_csv(inference_csv, index=False)
    pd.DataFrame(rows).to_excel(inference_xlsx, index=False)

    return len(rows)


# ============================================================
# METRICS
# ============================================================

def evaluate_map(
    model: RTDetrForObjectDetection,
    image_processor: RTDetrImageProcessor,
    image_paths: list[Path],
):
    if not TORCHMETRICS_AVAILABLE:
        return {
            "mAP@50": "",
            "mAP@50-95": "",
            "Precision": "",
            "Recall": "",
            "note": "torchmetrics not installed, mAP skipped",
        }

    metric = MeanAveragePrecision(
        box_format="xyxy",
        iou_type="bbox",
        class_metrics=False,
    )

    model.eval()

    for image_path in tqdm(image_paths, desc="Evaluating mAP"):
        pred_boxes, pred_scores, pred_labels = run_model_on_image(
            model=model,
            image_processor=image_processor,
            image_path=image_path,
            device=DEVICE,
            threshold=MAP_THRESHOLD,
        )

        gt_boxes, gt_labels = read_yolo_labels_xyxy(image_path)

        preds = [
            {
                "boxes": torch.tensor(pred_boxes, dtype=torch.float32),
                "scores": torch.tensor(pred_scores, dtype=torch.float32),
                "labels": torch.tensor(pred_labels, dtype=torch.int64),
            }
        ]

        targets = [
            {
                "boxes": torch.tensor(gt_boxes, dtype=torch.float32),
                "labels": torch.tensor(gt_labels, dtype=torch.int64),
            }
        ]

        metric.update(preds, targets)

    results = metric.compute()

    map_50 = float(results["map_50"].item()) if "map_50" in results else ""
    map_all = float(results["map"].item()) if "map" in results else ""

    return {
        "mAP@50": map_50,
        "mAP@50-95": map_all,
        "Precision": "",
        "Recall": "",
        "note": "",
    }


def save_results_plot(results_csv: Path, output_path: Path):
    try:
        import matplotlib.pyplot as plt

        df = pd.read_csv(results_csv)

        if "eval_loss" not in df.columns:
            return

        plt.figure()
        plt.plot(df["epoch"], df["eval_loss"], label="eval_loss")

        if "loss" in df.columns:
            plt.plot(df["epoch"], df["loss"], label="train_loss")

        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

    except Exception as exc:
        print(f"Could not create results plot: {exc}")


def save_trainer_log_history(trainer: Trainer, run_dir: Path):
    log_rows = []

    for item in trainer.state.log_history:
        row = dict(item)
        log_rows.append(row)

    if not log_rows:
        return

    results_csv = run_dir / "results.csv"
    results_xlsx = run_dir / "results.xlsx"

    df = pd.DataFrame(log_rows)
    df.to_csv(results_csv, index=False)
    df.to_excel(results_xlsx, index=False)

    save_results_plot(results_csv, run_dir / "results.png")


def save_summary(rows: list[dict[str, Any]]):
    df = pd.DataFrame(rows, columns=SUMMARY_FIELDS)
    df.to_csv(SUMMARY_CSV, index=False)
    df.to_excel(SUMMARY_XLSX, index=False)


# ============================================================
# TRAINING
# ============================================================

def make_training_args(output_dir: Path):
    kwargs = {
        "output_dir": str(output_dir),
        "per_device_train_batch_size": BATCH_SIZE,
        "per_device_eval_batch_size": BATCH_SIZE,
        "num_train_epochs": EPOCHS,
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "logging_steps": 20,
        "save_strategy": "epoch",
        "save_total_limit": 2,
        "remove_unused_columns": False,
        "dataloader_num_workers": NUM_WORKERS,
        "fp16": FP16,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "report_to": "none",
        "seed": SEED,
        "load_best_model_at_end": False,
    }

    params = inspect.signature(TrainingArguments.__init__).parameters

    if "eval_strategy" in params:
        kwargs["eval_strategy"] = "epoch"
    else:
        kwargs["evaluation_strategy"] = "epoch"

    return TrainingArguments(**kwargs)


def make_trainer(
    model,
    training_args,
    train_dataset,
    val_dataset,
    collator,
    image_processor,
):
    trainer_kwargs = {
        "model": model,
        "args": training_args,
        "train_dataset": train_dataset,
        "eval_dataset": val_dataset,
        "data_collator": collator,
    }

    params = inspect.signature(Trainer.__init__).parameters

    if "processing_class" in params:
        trainer_kwargs["processing_class"] = image_processor
    else:
        trainer_kwargs["tokenizer"] = image_processor

    return Trainer(**trainer_kwargs)


def train_and_infer_one_model(spec: dict[str, str]) -> dict[str, Any]:
    display_name = spec["display_name"]
    checkpoint = spec["checkpoint"]

    run_name = slugify(display_name)
    run_dir = PROJECT_DIR / run_name
    weights_dir = run_dir / "weights"
    final_model_dir = weights_dir / "final_model"
    val_dir = run_dir / "val_val"

    run_dir.mkdir(parents=True, exist_ok=True)
    weights_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100)
    print(f"Starting model: {display_name}")
    print(f"Checkpoint: {checkpoint}")
    print(f"Run directory: {run_dir}")
    print("=" * 100)

    data_info = read_data_yaml(DATA_YAML)
    id2label = data_info["id2label"]
    label2id = data_info["label2id"]
    num_labels = data_info["num_classes"]

    args_yaml = {
        "display_name": display_name,
        "checkpoint": checkpoint,
        "data": str(DATA_YAML),
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "lr": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "warmup_ratio": WARMUP_RATIO,
        "device": DEVICE,
        "seed": SEED,
        "conf_threshold": CONF_THRESHOLD,
    }

    with open(run_dir / "args.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(args_yaml, f, sort_keys=False)

    image_processor = RTDetrImageProcessor.from_pretrained(
        checkpoint,
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
    )

    model = RTDetrForObjectDetection.from_pretrained(
        checkpoint,
        num_labels=num_labels,
        id2label=id2label,
        label2id=label2id,
        ignore_mismatched_sizes=True,
    )

    model.to(DEVICE)

    train_dataset = YoloDetectionDataset(
        data_yaml=DATA_YAML,
        split="train",
        image_processor=image_processor,
    )

    val_dataset = YoloDetectionDataset(
        data_yaml=DATA_YAML,
        split="val",
        image_processor=image_processor,
    )

    train_image_paths = get_split_image_paths(DATA_YAML, "train")
    val_image_paths = get_split_image_paths(DATA_YAML, "val")

    save_gt_visuals(
        image_paths=train_image_paths,
        run_dir=run_dir,
        id2label=id2label,
        prefix="train",
        max_images=3,
    )

    save_gt_visuals(
        image_paths=val_image_paths,
        run_dir=run_dir,
        id2label=id2label,
        prefix="val",
        max_images=3,
    )

    training_args = make_training_args(run_dir)

    trainer = make_trainer(
        model=model,
        training_args=training_args,
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        collator=RTDETRCollator(image_processor),
        image_processor=image_processor,
    )

    trainer.train()

    save_trainer_log_history(trainer, run_dir)

    trainer.save_model(str(final_model_dir))
    image_processor.save_pretrained(str(final_model_dir))

    model = RTDetrForObjectDetection.from_pretrained(str(final_model_dir))
    image_processor = RTDetrImageProcessor.from_pretrained(str(final_model_dir))
    model.to(DEVICE)
    model.eval()

    save_prediction_visuals(
        model=model,
        image_processor=image_processor,
        image_paths=val_image_paths,
        run_dir=run_dir,
        id2label=id2label,
        prefix="val",
        max_images=3,
    )

    save_prediction_visuals(
        model=model,
        image_processor=image_processor,
        image_paths=val_image_paths,
        run_dir=val_dir,
        id2label=id2label,
        prefix="val",
        max_images=10,
    )

    metrics = evaluate_map(
        model=model,
        image_processor=image_processor,
        image_paths=val_image_paths,
    )

    num_inference_images = save_smartglasses_inference(
        model=model,
        image_processor=image_processor,
        run_dir=run_dir,
        id2label=id2label,
    )

    benchmark_metrics = {
        "model": {
            "display_name": display_name,
            "checkpoint": checkpoint,
        },
        "metrics": metrics,
        "data": {
            "num_val_images": len(val_image_paths),
            "num_inference_images": num_inference_images,
        },
        "settings": args_yaml,
    }

    with open(run_dir / "benchmark_metrics.json", "w", encoding="utf-8") as f:
        json.dump(benchmark_metrics, f, indent=2)

    row = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "display_name": display_name,
        "checkpoint": checkpoint,
        "status": "completed",
        "epochs": EPOCHS,
        "imgsz": IMAGE_SIZE,
        "batch": BATCH_SIZE,
        "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
        "effective_batch": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
        "device": DEVICE,
        "seed": SEED,
        "run_dir": str(run_dir),
        "weights": str(final_model_dir),
        "mAP@50": metrics.get("mAP@50", ""),
        "mAP@50-95": metrics.get("mAP@50-95", ""),
        "Precision": metrics.get("Precision", ""),
        "Recall": metrics.get("Recall", ""),
        "num_val_images": len(val_image_paths),
        "num_inference_images": num_inference_images,
        "note": metrics.get("note", ""),
    }

    print(f"Completed model: {display_name}")
    print(f"Saved run: {run_dir}")

    return row


def main():
    set_seed(SEED)

    ensure_data_yaml()
    validate_dataset()

    PROJECT_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Device: {DEVICE}")
    print(f"Torch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    rows = []

    if SUMMARY_CSV.exists():
        try:
            rows = pd.read_csv(SUMMARY_CSV).to_dict("records")
        except Exception:
            rows = []

    completed_names = {
        row.get("display_name")
        for row in rows
        if row.get("status") == "completed"
    }

    for spec in MODEL_SPECS:
        display_name = spec["display_name"]

        if display_name in completed_names:
            print(f"Skipping already completed model: {display_name}")
            continue

        try:
            row = train_and_infer_one_model(spec)

        except Exception as exc:
            print(f"FAILED: {display_name}")
            print(exc)
            print(traceback.format_exc())

            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "display_name": display_name,
                "checkpoint": spec["checkpoint"],
                "status": "failed",
                "epochs": EPOCHS,
                "imgsz": IMAGE_SIZE,
                "batch": BATCH_SIZE,
                "gradient_accumulation_steps": GRADIENT_ACCUMULATION_STEPS,
                "effective_batch": BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS,
                "device": DEVICE,
                "seed": SEED,
                "run_dir": str(PROJECT_DIR / slugify(display_name)),
                "weights": "",
                "mAP@50": "",
                "mAP@50-95": "",
                "Precision": "",
                "Recall": "",
                "num_val_images": "",
                "num_inference_images": "",
                "note": str(exc),
            }

        rows = [
            old_row for old_row in rows
            if old_row.get("display_name") != display_name
        ]

        rows.append(row)
        save_summary(rows)

    print("=" * 100)
    print("All selected Hugging Face RT-DETR models finished.")
    print(f"Summary CSV:  {SUMMARY_CSV}")
    print(f"Summary XLSX: {SUMMARY_XLSX}")
    print(f"Runs folder:   {PROJECT_DIR}")
    print("=" * 100)


if __name__ == "__main__":
    main()
