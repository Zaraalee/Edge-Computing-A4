# Assignment 4: Knowledge Distillation and Neural Architecture Search for Edge Devices

* **Q1, Knowledge distillation (CIFAR-100):** a ResNet-50 teacher, three lightweight students
  (MobileNetV2, SqueezeNet, EfficientNet-Lite0), each trained once without distillation (baseline)
  and once with distillation (soft targets with temperature, hard labels and feature distillation).
* **Q2, Hardware-aware Neural Architecture Search (CIFAR-10):** a searchable CNN space, an
  architecture generator, an evolutionary search that measures real CPU latency, model size and
  peak memory and rejects architectures that break the deployment constraints, and the best
  architecture retrained from scratch.

All models are exported to ONNX and run on the CPU with ONNX Runtime. Everything was trained and
measured on a 2-core CPU (no GPU).

## Results

**Q1 (CIFAR-100, 10,000 test images)**

| Model | Parameters | Top-1 without KD | Top-1 with KD | KD gain |
|---|---|---|---|---|
| ResNet-50 teacher | 23.7 M | 66.47% (teacher) | not applicable | not applicable |
| MobileNetV2 | 2.35 M | 58.34% | 63.51% | +5.17 |
| SqueezeNet | 0.78 M | 56.63% | 60.98% | +4.35 |
| EfficientNet-Lite0 | 3.50 M | 60.63% | 65.61% | +4.98 |

**Q2 (CIFAR-10):** 58 architectures searched, 18 rejected by the constraints (latency <= 1.5 ms,
size <= 1.0 MB), 40 trained. The best one, trained from scratch, reaches **83.07%** test accuracy
with 28,138 parameters (0.14 MB), 0.8 ms per image on one CPU thread (0.12 ms with ONNX Runtime).

Full tables: `results/*.csv`; report: `report/23i-6102_A_A4_EdgeComputing.pdf`.

## Deliverables checklist

| # | Deliverable | Where it is |
|---|---|---|
| 1 | Teacher model | `checkpoints/teacher_resnet50.pth`, `q1_distillation/train_teacher.py` |
| 2 | Student architectures | `q1_distillation/models.py` (MobileNetV2, SqueezeNet, EfficientNet-Lite0) |
| 3 | Baseline student models | `checkpoints/student_*_baseline.pth` (3 files), `q1_distillation/train_baseline.py` |
| 4 | Knowledge distillation | `q1_distillation/kd.py`, `q1_distillation/train_kd.py`, `checkpoints/student_*_kd.pth` (3 files) |
| 5 | NAS search space | `q2_nas/search_space.py` (search space and `build_model` generator) |
| 6 | NAS algorithm | `q2_nas/evolution_search.py` (evolutionary search), `q2_nas/hardware.py` (constraints) |
| 7 | NAS results | `results/nas_best_config.json`, `results/nas_search_history.csv`, `results/nas_search_history.json` |
| 8 | Optimized architecture | `checkpoints/nas_best_model.pth`, `q2_nas/train_best.py` |
| 9 | Inference pipeline | `infer.py` (all six students and the NAS model) |
| 10 | ONNX deployment | `onnx_models/*.onnx` (7 files), `export_onnx.py`, `onnx_infer.py` |
| 11 | Benchmarking | `benchmark.py`, `results/*_benchmark.csv`, `results/*_test_predictions.csv` |
| 12 | Technical report | `report/23i-6102_A_A4_EdgeComputing.pdf` (LaTeX source `report/A4_report.tex`), `screenshots/` |
| 13 | README | this file |

---

## 1. Installation

Python 3.9+ is enough; no GPU is needed.

```bash
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**Datasets** (not included in the repo):
* CIFAR-10 and CIFAR-100 are downloaded automatically by `torchvision` into `data/` the first time.
* If that download is blocked, put the official archive `cifar-100-python.tar.gz`
  (https://www.cs.toronto.edu/~kriz/cifar.html) into `data/`, and clone the CIFAR-10 image folders:
  `git clone https://github.com/YoongiKim/CIFAR-10-images data/CIFAR-10-images`.
* Both are cached as `data/cifar10_cache.npz` / `data/cifar100_cache.npz` after the first load.

---

## 2. Step-by-step execution

The trained models are already in `checkpoints/` and `onnx_models/`, so steps 7 to 11 can be run
straight away. Steps 1 to 6 retrain everything.

| Step | Command | Output |
|---|---|---|
| 1 | `python q1_distillation/train_teacher.py --epochs 20` | `checkpoints/teacher_resnet50.pth` |
| 2 | `python q1_distillation/train_baseline.py --epochs 15` | `checkpoints/student_*_baseline.pth` (3 files) |
| 3 | `python q1_distillation/train_kd.py --epochs 15 --T 4 --alpha 0.7 --beta 1.0` | `checkpoints/student_*_kd.pth` (3 files) |
| 4 | `python q2_nas/search_space.py` | prints the size of the search space and 3 random networks |
| 5 | `python q2_nas/evolution_search.py` | `results/nas_search_history.csv/.json`, `results/nas_best_config.json` |
| 6 | `python q2_nas/train_best.py --epochs 30` | `checkpoints/nas_best_model.pth` |
| 7 | `python infer.py --model all --index 0 2000 4000 6000 8000` | predictions of every model |
| 8 | `python benchmark.py` | `results/*_benchmark.csv`, per-image prediction CSVs, chart |
| 9 | `python export_onnx.py` | `onnx_models/*.onnx` (7 files) |
| 10 | `python onnx_infer.py --benchmark` | `results/onnx_benchmark.csv` |
| 11 | `python onnx_infer.py --model nas_best --index 0 2000 4000 6000 8000` | ONNX Runtime predictions |

`bash run_all.sh` runs all eleven steps in order and saves each step's console output in `logs/`.
The training scripts save their state after every epoch and continue automatically if they are
interrupted.

---

## 3. Folder layout

```
common/                 shared code
  data.py               CIFAR-10 / CIFAR-100 loading, normalisation, augmentation
  train.py              training loop and evaluation
  utils.py              seeds, latency, size, MACs, peak memory
  registry.py           how to rebuild every trained model from its checkpoint
q1_distillation/        Question 1
  models.py             ResNet-50, MobileNetV2, SqueezeNet, EfficientNet-Lite0
  kd.py                 the distillation framework (soft targets, temperature, hard+soft, features)
  train_teacher.py      teacher training
  train_baseline.py     students WITHOUT distillation
  train_kd.py           students WITH distillation
  runner.py             optimiser recipe, resumable training, history CSV
q2_nas/                 Question 2
  search_space.py       search space + architecture generator
  hardware.py           CPU latency, model size, peak memory, deployment constraints
  evolution_search.py   evolutionary (aging) search
  train_best.py         best architecture trained from scratch
infer.py                unified PyTorch inference (all students + NAS model)
export_onnx.py          ONNX export and check against PyTorch
onnx_infer.py           ONNX Runtime CPU inference and benchmark
benchmark.py            automated PyTorch benchmark, writes CSV files
make_charts.py          charts for the report
checkpoints/            trained models
onnx_models/            exported ONNX models
results/                CSV / JSON results and charts
logs/                   console output of every step
screenshots/            terminal screenshots used in the report
report/                 technical report (PDF + LaTeX source)
```

## 4. Reproducibility

* Fixed random seeds in every script; all hyper-parameters are command-line arguments whose
  defaults are the values used for the reported results.
* The test sets are never used to choose a model: every training run keeps its last epoch, and
  the NAS fitness is measured on a 5,000-image validation split taken from the CIFAR-10 training set.
* Latency is always measured for one image on one CPU thread, after warm-up runs.
