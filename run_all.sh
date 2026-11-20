#!/usr/bin/env bash
# Reproducible end-to-end pipeline. Every step writes its console output to logs/.
set -e
cd "$(dirname "$0")"
mkdir -p logs checkpoints results onnx_models

echo "== Q1-1 Teacher ResNet-50 ==";            python -u q1_distillation/train_teacher.py --epochs 20   | tee logs/q1_01_train_teacher.log
echo "== Q1-2 Students without KD ==";          python -u q1_distillation/train_baseline.py --epochs 15  | tee logs/q1_02_train_baseline.log
echo "== Q1-3 Students with KD ==";             python -u q1_distillation/train_kd.py --epochs 15 --T 4 --alpha 0.7 --beta 1.0 | tee logs/q1_03_train_kd.log
echo "== Q2-0 Search space demo ==";            python -u q2_nas/search_space.py                       | tee logs/q2_00_search_space.log
echo "== Q2-1 Evolutionary search ==";          python -u q2_nas/evolution_search.py                   | tee logs/q2_01_evolution_search.log
echo "== Q2-2 Best architecture from scratch =="; python -u q2_nas/train_best.py --epochs 30            | tee logs/q2_02_train_best.log
echo "== Inference (PyTorch) ==";               python -u infer.py --model all --index 0 2000 4000 6000 8000         | tee logs/03_infer.log
echo "== Benchmark ==";                         python -u benchmark.py                                  | tee logs/04_benchmark.log
echo "== ONNX export ==";                       python -u export_onnx.py                                | tee logs/05_export_onnx.log
echo "== ONNX Runtime benchmark ==";            python -u onnx_infer.py --benchmark                     | tee logs/06_onnx_benchmark.log
echo "== ONNX Runtime inference ==";            python -u onnx_infer.py --model nas_best --index 0 2000 4000 6000 8000 | tee logs/07_onnx_infer.log
python make_charts.py
echo "Done. Results are in results/"
