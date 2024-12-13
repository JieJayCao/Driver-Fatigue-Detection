# Driver-Fatigue-Detection
DualFusion: Efficient Fusion of Time and Frequency Domain Features for EEG-Based Driver Fatigue Detectio.
This is a graduate course final project for ELEC 872 at Dept. of ECE, Queen's University.


## Overview
DualFusion is a deep learning model designed for EEG-based fatigue detection. It employs a dual-stream architecture that combines spectral analysis and temporal features to achieve robust fatigue state classification.

## Key Features
- Dual-stream architecture combining FFT and temporal features
- Support for both intra-subject and inter-subject classification
- Lightweight model design with depthwise separable convolutions
- High recall rate optimized for safety-critical applications

## Model Architecture
The model consists of two main processing streams:
1. Spectral Stream: Processes frequency domain features using FFT and linear embedding
2. Temporal Stream: Extracts temporal features using multi-scale convolutions

## Dataset
The model is evaluated on the SEED-VIG dataset, which contains:
- EEG recordings from 12 subjects
- Binary classification: Alert (0) vs. Fatigue (1) states
- 17 EEG channels

## Requirements
- Python 3.7+
- PyTorch 1.8+
- PyTorch Lightning
- NumPy
- scikit-learn
- tqdm

