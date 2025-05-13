# Energy-Based DINOv2 Novelty Detection for Transfer Learning

This repository contains the implementation of a transfer learning pipeline using the DINOv2 ViT-B/14 model enhanced with energy-based out-of-distribution (OOD) detection. It was developed as part of the COMS 4995 Neural Networks and Deep Learning course at Columbia University.

## Project Summary

The objective is to perform joint classification of superclasses (bird, dog, reptile, novel) and subclasses (87 seen subclasses + 1 novel) on a dataset where the test distribution contains unseen subclass and novel superclass instances. This project fine-tunes a pretrained DINOv2 backbone and incorporates an energy-based scoring mechanism to detect OOD samples during inference.

## Model Overview

- **Backbone**: DINOv2 ViT-B/14 with register tokens
- **Classification Heads**: Dual linear heads for superclass and subclass classification
- **Novelty Detection**: LogSumExp-based energy scoring with adaptive thresholding
- **Post-hoc Calibration**: Softmax energy thresholds determined using held-out validation data

## Environment

- Platform: Google Colab Pro
- GPU: NVIDIA A100 (40 GB)
- RAM: 83.5 GB
- Disk: 112.6 GB
- Python: 3.11
- PyTorch: 2.7.0+cu126
- xFormers: 0.0.30

## Dataset

The dataset must be structured as follows:
- `train_images.zip`, `test_images.zip`
- `train_data.csv`, `superclass_mapping.csv`, `subclass_mapping.csv`

Ensure these are unzipped or preloaded into the Colab session before running the notebook.

## Setup and Usage

1. **Upload Dataset Files**

   Upload the following to your Colab environment before execution:
   - `train_images.zip`
   - `test_images.zip`
   - `train_data.csv`
   - `superclass_mapping.csv`
   - `subclass_mapping.csv`

2. **Run Notebook in Order**

   The notebook is organized into four main sections:
   - **Section V1**: Loads DINOv2 and fine-tunes on seen classes.
   - **Section V2**: Adds energy-based novelty detection for unseen classes.
   - **Section Metrics**: Computes metrics including per-class accuracies and categorical cross-entropy.
   - **Section Version History**: Contains earlier variants (V1.3–V1.7, V2.1) for reference only.

3. **Required Libraries**

   Most dependencies are installed inside the notebook itself:
   - `torch`, `torchvision`, `xformers`, `pandas`, `tqdm`, `matplotlib`, `numpy`

4. **Output**

   The final model:
   - Produces predictions in `sm5726_test_predictions.csv`
 
## Citation

- DINOv2: [Oquab et al., 2023](https://arxiv.org/abs/2304.07193)
- Energy-based OOD: [Liu et al., 2020](https://arxiv.org/abs/2010.03759)

## Author

**Sibi Marappan**  
Columbia University  
Course: COMS 4995 - Neural Networks and Deep Learning (Spring 2025)

