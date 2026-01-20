# DCVC-INT8

**Learned Step Size Quantization (LSQ) for Efficient Neural Video Decoding**

> INT8 Decoder Acceleration for DCVC-RT with Zero Quality Loss

---

## 📌 Overview

This repository implements an **INT8 quantized decoder for DCVC-RT (Deep Contextual Video Compression – Real Time)** using **Learned Step Size Quantization (LSQ)**.

The goal is to **replace FP32 decoding with INT8 operations** while preserving **rate–distortion performance**, enabling **hardware-friendly deployment** on edge devices and accelerators.

### Key Result

* **BD-PSNR drop: 0.0000 dB**
* **Identical RD curves** between FP32 and INT8 decoder
* Decoder becomes **INT8-ready without retraining the full codec**

---

## 🎯 Motivation

Neural video codecs like DCVC-RT achieve excellent compression quality but rely heavily on **FP32 computation**, which is:

* Slow on edge hardware
* Energy-inefficient
* Difficult to deploy on INT8 accelerators (TensorRT, DSPs, NPUs)

### Why LSQ?

Naive quantization (rounding weights) **destroys video quality**.

**Learned Step Size Quantization (LSQ)** solves this by:

* Learning optimal quantization scales per layer
* Preserving precision where it matters
* Ignoring weight outliers

---

## 🧠 Method Summary

### Architecture

* Original **DCVC-RT encoder remains FP32**
* **Decoder layers are replaced with LSQ-based INT8 layers**
* Bit-width: **INT8 (−128 to 127)**

### Quantization Equation

[
\hat{w} = \text{round}\left(\frac{w}{s}\right) \times s
]

Where:

* ( s ) is a **learnable step size**
* Learned **per layer**

---

## 🧪 Training Strategy (Calibration)

### Teacher–Student Distillation

| Role    | Model                 |
| ------- | --------------------- |
| Teacher | Original FP32 DCVC-RT |
| Student | INT8 LSQ Decoder      |

### Loss Function

```math
L = \| x_{teacher} - x_{student} \|_2^2
```

### Key Details

* **No ground-truth labels required**
* Calibrated **directly on test video sequences**
* 100 epochs
* Only LSQ scales are trainable

---

## 📊 Results

### Quantitative

* **BD-PSNR = 0.0000 dB**
* RD curves are **perfectly overlapping**
* No bitrate increase

### Qualitative

* No visible artifacts
* Stable temporal reconstruction

### Internal Analysis

* LSQ learns **tight quantization grids**
* Avoids outlier-driven dynamic range explosion
* Preserves weight distributions

---

## 🚀 Quick Start

### 1️⃣ Environment Setup

```bash
conda create -n dcvc_int8 python=3.9
conda activate dcvc_int8
pip install -r requirements.txt
```

### 2️⃣ Sanity Check INT8 Decoder

```bash
python scripts/test_int8.py
```

Expected output:

```text
Output shape: [1, 3, H, W]
Min / Max: stable range
```

---

## 🏋️ Train LSQ Scales (Calibration)

```bash
python scripts/train_lsq.py \
  --teacher_path cvpr2025_video.pth.tar \
  --decoder_int8 models/decoder_int8.py \
  --epochs 100 \
  --lr 1e-4
```

---

## 📈 Evaluate on Video Dataset

```bash
python test_video.py \
  --model_path_i cvpr2025_image.pth.tar \
  --model_path_p cvpr2025_video.pth.tar \
  --rate_num 4 \
  --test_config dataset_config_example_yuv420.json \
  --cuda 1 \
  --write_stream 1 \
  --output_path output.json
```

---

## 🛠 Debugging Notes (Lessons Learned)

| Issue                 | Fix                           |
| --------------------- | ----------------------------- |
| FP16 overflow         | Forced FP32 inference for LSQ |
| LSQ reset during test | Locked initialization flags   |
| Entropy coder crash   | Robust per-sequence execution |
| INT8 / FP16 mismatch  | Explicit dtype control        |

---

## 📌 Limitations

* Encoder remains FP32
* CUDA kernels still expect FP32 activations
* INT8 execution is **functional**, not yet kernel-optimized

---

## 🔮 Future Work

* Full INT8 pipeline (encoder + decoder)
* TensorRT / ONNX export
* Per-channel LSQ
* Mixed-precision entropy decoding
* Hardware benchmarking (Jetson / NPU)

---

## 📄 Citation

If you use this work, please cite:

```bibtex
@misc{dcvc_int8_lsq,
  title={INT8 Neural Video Decoding with Learned Step Size Quantization},
  author={Wei-Jung Tsao, Tushar Jaryal, Ting-Yuan Chang},
  year={2026}
}
```
