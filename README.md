# Detection Benchmarking

A lightweight object detection benchmarking runner for YOLO-family models and RT-DETR models on YOLO-format datasets.

This repository is designed to make detection benchmarking simple and reproducible: provide a dataset YAML, choose the models to evaluate, train each model under the same settings, and export comparable validation metrics.

## Purpose

The goal of this project is to benchmark different object detection algorithms under a consistent training and validation setup.

The benchmark records the following metrics for each model:

| Model | mAP@50 | mAP@50-95 | Precision | Recall |
| ----- | -----: | --------: | --------: | -----: |

The repository currently focuses on YOLO and RT-DETR models, but it is intended to grow over time as more detection algorithms are added.

## Features

* Supports YOLO-format datasets.
* Trains YOLO and RT-DETR models using Ultralytics.
* Runs validation after training.
* Records:

  * `mAP@50`
  * `mAP@50-95`
  * Precision
  * Recall
* Saves each model run separately under `benchmark_runs/`.
* Exports summary results to:

  * `benchmark_results.csv`
  * `benchmark_results.xlsx` when `pandas` and `openpyxl` are installed
* Supports resuming interrupted training from `weights/last.pt`.
* Allows benchmarking all default models or a selected subset.
* Keeps datasets, model weights, logs, and training outputs out of Git.

## Repository Structure

```text
detection-benchmarking/
├── yolo_benchmark_models.py       # Main benchmarking script for yolo models
├── rtdetr_benchmark.py       # Main benchmarking script for RT-DETR models (huggingface)
├── data.example.yaml         # Example YOLO dataset YAML
├── requirements.txt          # Python dependencies
├── README.md                 # Project overview and usage
├── README_BENCHMARK.md       # Detailed benchmark instructions
└── .gitignore                # Excludes datasets, weights, runs, and logs
```

## Setup

Use Python 3.10 or newer.

Clone the repository:

```bash
git clone git@github.com:mshafay5/detection-benchmarking.git
cd detection-benchmarking
```

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For GPU training, make sure your PyTorch installation matches your CUDA version. You can check whether PyTorch detects your GPU with:

```bash
python -c "import torch; print(torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'No GPU detected')"
```

## Dataset Format

The dataset must be in YOLO format.

A typical dataset structure looks like this:

```text
dataset/
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
├── test/
│   ├── images/
│   └── labels/
└── data.yaml
```

A typical YOLO `data.yaml` file looks like this:

```yaml
path: /absolute/path/to/dataset
train: train/images
val: valid/images
test: test/images

nc: 1
names:
  - class_name
```

You can copy the example file and edit it:

```bash
cp data.example.yaml data.yaml
```

Then update the dataset path and class names inside `data.yaml`.

## Dry Run

Before starting any training, run a dry run to check the configuration:

```bash
python benchmark_models.py --data /path/to/data.yaml --dry-run
```

This helps confirm that the dataset YAML and model list are configured correctly.

## Run a Full Benchmark

Run the default benchmark:

```bash
python yolo_benchmark_models.py --data /path/to/data.yaml
```

By default, the script is designed for benchmark training and validation across the configured model list.

## Common Options

Run with custom epochs, image size, and batch size:

```bash
python yolo_benchmark_models.py --data /path/to/data.yaml --epochs 100 --imgsz 640 --batch 8
```

Run only selected models:

```bash
python yolo_benchmark_models.py --data /path/to/data.yaml --models YOLOv11m
```

Train from scratch instead of pretrained weights:

```bash
python yolo_benchmark_models.py --data /path/to/data.yaml --init scratch
```

Disable resume behavior:

```bash
python yolo_benchmark_models.py --data /path/to/data.yaml --no-resume
```

Force rerun even if previous results exist:

```bash
python yolo_benchmark_models.py --data /path/to/data.yaml --force
```

## Default Models

The default model list is defined in `DEFAULT_MODELS` inside `benchmark_models.py`.

Current default model groups include:

### YOLOv11

* `yolo11x.pt`
* `yolo11l.pt`
* `yolo11m.pt`

### YOLOv12

* `yolo12x.pt`
* `yolo12l.pt`
* `yolo12m.pt`

### YOLOv26

* `yolo26x.pt`
* `yolo26l.pt`
* `yolo26m.pt`


Additional detection models can be added by creating new `ModelSpec` entries in `yolo_benchmark_models.py`.

## Output Files

Training and validation outputs are saved locally.

```text
benchmark_runs/
benchmark_results.csv
benchmark_results.xlsx
```

Each model gets its own run directory under `benchmark_runs/`.

The summary table includes:

| Column      | Description                                           |
| ----------- | ----------------------------------------------------- |
| `Model`     | Model name or label                                   |
| `mAP@50`    | Mean average precision at IoU 0.50                    |
| `mAP@50-95` | Mean average precision averaged from IoU 0.50 to 0.95 |
| `Precision` | Validation precision                                  |
| `Recall`    | Validation recall                                     |

## Git Ignore Policy

Generated files are ignored by Git to keep the repository lightweight.

Ignored items include:

```text
benchmark_runs/
benchmark_results.csv
benchmark_results.xlsx
*.pt
*.pth
*.onnx
*.engine
*.log
*.zip
datasets/
data/
```

Do not commit datasets, trained weights, or large benchmark outputs directly to this repository.

## Recommended Workflow

1. Add or update benchmark code.
2. Run a dry run.
3. Run a small smoke test with one model and one epoch.
4. Run the full benchmark.
5. Review `benchmark_results.csv`.
6. Commit only code, configuration examples, and documentation.

Example Git workflow:

```bash
git status
git add README.md benchmark_models.py README_BENCHMARK.md requirements.txt data.example.yaml .gitignore
git commit -m "Update benchmark documentation"
git push
```

## Roadmap

Planned improvements:

* Add more detection algorithms.
* Add configurable experiment profiles.
* Add automatic plotting of benchmark results.
* Add support for exporting results to publication-ready tables.
* Add hardware and runtime logging.
* Add optional test-set evaluation.
* Add better support for non-Ultralytics detectors.

## Repository

GitHub:

```text
https://github.com/mshafay5/detection-benchmarking
```
