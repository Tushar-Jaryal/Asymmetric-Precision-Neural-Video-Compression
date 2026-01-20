# DCVC-INT8
Learned Step Size Quantization (LSQ) for Efficient Neural Video Decoding
INT8 Decoder Acceleration for DCVC-RT with Zero Quality Loss

## Overview
This repository implements an INT8 quantized decoder for DCVC-RT (Deep Contextual Video Compression - Real Time) using Learned Step Size Quantization (LSQ).
The goal is to replace FP32 decoding with INT8 operations while preserving rate-distortion performance, enabling hardware-friendly deployment on edge devices and accelerators.

## Key Result
- BD-PSNR drop: 0.0000 db
- Identical RD curves between FP32 adn INT8 Decoder
- Decoder becomes INT8-ready without retraining the full codec

## Motivation
Neural video codec like DCVC-RT achieve excellent compression quality but rely heavily on FP32 computation, which is:
- Slow on edge hardware
- Energy-inefficient
- Difficult to deploy on INT8 acclerators (TensorRT, DSPs, NPUs)

## Why LSQ?
Naive Quantization (rounding weights) destroys video quality
Learned Step Size Quantization (LSQ) solves this by:
- Learned optimal quantization scales per layer
- Preserving precision where it matters
- Ignoring weight outliers

## Method Summary
### Architecture
- Original DCVC-RT encoder remains FP32
- Decoder layers are replaced with LSQ-based INT8 layers
- Bit-width: INT8 (-128 to 127)

### Training Strategy
Teacher-Student Distillation
Role                    Model
Teacher                 Original FP32 DCVC-RT
Student                 INT8 LSQ Decoder

## Results
### Quantitative
- BD-PSNR = 0.0000db
- RD curves are perfectly overlapping
- No bitrate increase

### Qualitative
- No visible artifacts
- Stable temporal reconstruction

### Internal Analysis
- LSQ learns tight quantization grids
- Avoids outlier-driven dynamic range explosion
- Preserves weight distributions
