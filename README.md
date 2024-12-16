# DualFusion EEG Classification
DualFusion: Efficient Fusion of Time and Frequency Domain Features for EEG-Based Driver Fatigue Detectio.
This is a graduate course final project for ELEC 872 at Dept. of ECE, Queen's University.

This project implements a deep learning model, DualFusion, for EEG-based classification tasks. The model is designed to classify EEG signals into different states, such as awake and fatigue, using PyTorch and PyTorch Lightning.

## Introduction

DualFusion is a neural network model that combines frequency and temporal representations of EEG signals to improve classification accuracy. It is designed to handle both intra-subject and inter-subject classification tasks.

1. Clone the repository:
   ```bash
   git clone https://github.com/JieJayCao/Driver-Fatigue-Detection.git
   ```
2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Prepare your dataset in the specified format and update the paths in the script.
2. Run the training script:
   ```bash
   python DualFusion.py
   ```


## Configuration

The configuration settings are defined in the `config` dictionary within the script. Key parameters include:


- `subjects_num`: Number of subjects in the dataset.
- `n_epochs`: Number of training epochs.
- `batch_size`: Batch size for training.
- `save_name`: Name of the model checkpoint file.
- `log_path1`: Path to save training logs.
- `num_class`: Number of classes in the classification task.


## Dataset

The SEED-VIG [dataset](https://figshare.com/articles/dataset/Extracted_SEED-VIG_dataset_for_cross-dataset_driver_drowsiness_recognition/26104987?file=47271799) is stored in the `Dataset/SEED-VIG-Subset` directory. Each `.npy` file corresponds to the EEG signals of a single subject.


## Results

The model's performance is evaluated using accuracy, precision, recall, and F1-score. These metrics are logged during training and testing phases. All experimental results of the comparison methods are stored in the `SOTA methods/` directory.
