# ScUnicorn: Blind Super-Resolution for scHi-C Data

ScUnicorn is a blind super-resolution algorithm designed for enhancing single-cell Hi-C (scHi-C) data. By dynamically learning degradation kernels and leveraging iterative restoration loops, ScUnicorn improves the resolution of Hi-C data while maintaining biological fidelity.

## Features
- **Blind Super-Resolution**: No pre-defined LR-HR pairs required.
- **Dynamic Degradation Kernels**: Learn optimal degradation models.
- **Iterative Restoration**: Alternating mechanism for enhanced resolution.
- **Comprehensive Pipeline**: Includes training, evaluation, and inference scripts.

---

## Folder Structure
```
ScUnicorn/
├── configs/                 # Configuration files
│   └── default_config.yaml  # Default settings for training and evaluation
├── data/                    # Placeholder for input/output Hi-C data
├── experiments/             # Logs, checkpoints, and results
├── models/                  # Core model components
│   ├── degradation_kernel.py
│   ├── restoration_loop.py
│   └── scunicorn.py         # Main ScUnicorn pipeline
├── scripts/                 # Execution scripts
│   ├── train.py             # Training script
│   ├── evaluate.py          # Evaluation script
│   ├── generate_hr.py       # Inference script
│   └── test_pipeline.py     # End-to-end test script
├── utils/                   # Utility functions
│   ├── data_loader.py       # Data loading utility
│   ├── metrics.py           # Metric computation (MSE, SSIM)
│   ├── visualization.py     # Visualization tools
│   └── logger.py            # Logging utility
└── README.md                # Project documentation
```

---

## Installation

### Prerequisites
- Python 3.8 or higher
- PyTorch 1.10 or higher
- Required Python libraries listed in `requirements.txt`

### Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/OluwadareLab/Unicorn
   cd scunicorn
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

### 1. Training
To train the ScUnicorn model, run the training script:
```bash
python scripts/train.py
```
Training parameters (e.g., batch size, learning rate) can be configured in `configs/default_config.yaml`.

### 2. Evaluation
Evaluate the trained model on synthetic data:
```bash
python scripts/evaluate.py
```
Update the checkpoint path in `configs/default_config.yaml` as needed.

### 3. Inference
Generate high-resolution Hi-C maps from low-resolution input:
```bash
python scripts/generate_hr.py
```
Specify the input and output file paths in the configuration file.

### 4. Testing the Pipeline
Run an end-to-end test of the ScUnicorn pipeline:
```bash
python scripts/test_pipeline.py
```

---

## Configuration
All hyperparameters and file paths are managed via the `configs/default_config.yaml` file. Customize it to fit your specific requirements.

---


