# YOLO RT-DETR Benchmark

A small object detection benchmarking runner for YOLO-family models and RT-DETR models on any YOLO-format dataset.

The goal is simple: point the script at a dataset YAML, choose the models you want, and get comparable training and validation metrics without a large framework setup.

## What It Does

- Trains supported YOLO and RT-DETR models through Ultralytics.
- Validates each model and records `mAP@50`, `mAP@50-95`, precision, and recall.
- Saves per-model runs under `benchmark_runs/`.
- Writes summary results to `benchmark_results.csv` and, when `pandas` and `openpyxl` are installed, `benchmark_results.xlsx`.
- Resumes interrupted runs from `weights/last.pt` by default.
- Lets you benchmark all models or a selected subset from one command.

## Setup

Use Python 3.10 or newer.

```bash
git clone https://github.com/<your-username>/yolo-rtdetr-benchmark.git
cd yolo-rtdetr-benchmark

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For GPU training, install a PyTorch build that matches your CUDA version before running the benchmark.

## Dataset

The script expects a standard YOLO dataset YAML:

```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names: ["class_name"]
```

You can copy `data.example.yaml` to `data.yaml` and edit the paths, or pass your dataset YAML directly:

```bash
python benchmark_models.py --data /path/to/data.yaml --dry-run
```

## Run

Check the configuration without training:

```bash
python benchmark_models.py --data /path/to/data.yaml --dry-run
```

Run a full benchmark with the defaults:

```bash
python benchmark_models.py --data /path/to/data.yaml
```

Useful options:

```bash
python benchmark_models.py --data /path/to/data.yaml --epochs 50 --imgsz 640 --batch 8
python benchmark_models.py --data /path/to/data.yaml --models YOLOv11m rtdetr-l.pt
python benchmark_models.py --data /path/to/data.yaml --init scratch
python benchmark_models.py --data /path/to/data.yaml --no-resume
python benchmark_models.py --data /path/to/data.yaml --force
```

## Supported Defaults

The default model list is defined in `DEFAULT_MODELS` inside `benchmark_models.py`.

Included defaults:

- YOLOv11: `yolo11x.pt`, `yolo11l.pt`, `yolo11m.pt`
- YOLOv12: `yolo12x.pt`, `yolo12l.pt`, `yolo12m.pt`
- YOLOv26: `yolo26x.pt`, `yolo26l.pt`, `yolo26m.pt`
- RT-DETR: `rtdetr-x.pt`, `rtdetr-l.pt`

To add another model, add a `ModelSpec` entry in `benchmark_models.py`.

## Outputs

Generated files are ignored by Git:

```text
benchmark_runs/
benchmark_results.csv
benchmark_results.xlsx
*.pt
*.log
```

This keeps the repository focused on the benchmark code while allowing each user to keep datasets, checkpoints, and run artifacts locally.
