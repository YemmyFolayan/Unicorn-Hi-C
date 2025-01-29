# ScUnicorn: Blind Super-Resolution for scHi-C Data

ScUnicorn is a blind super-resolution algorithm designed for enhancing single-cell Hi-C (scHi-C) data. By dynamically learning degradation kernels and leveraging iterative restoration loops, ScUnicorn improves the resolution of Hi-C data while maintaining biological fidelity.

## Features
- **Blind Super-Resolution**: No pre-defined LR-HR pairs required.
- **Dynamic Degradation Kernels**: Learn optimal degradation models.
- **Iterative Restoration**: Alternating mechanism for enhanced resolution.
- **CLI-Based Workflow**: Runs via command-line scripts, similar to HiCForecast.

---

## Folder Structure
```
ScUnicorn/
├── configs/                         # Configuration files
│   ├── default_config.yaml          # Default model and training settings
├── example_data/                     # Example dataset directory
│   ├── raw/                          # Raw input files
│   ├── processed/                     # Processed dataset directory
├── models/                           # Model components
│   ├── degradation_kernel.py
│   ├── restoration_loop.py
│   ├── scunicorn.py                   # Main model integration
├── scripts/                           # Execution scripts
│   ├── process_hic_data.py            # Converts raw data to .npy format
│   ├── process_hic_data.sh            # Bash script for data preprocessing
│   ├── makedata.py                     # Extracts patches for training
│   ├── makedata.sh                     # Bash script for patch extraction
│   ├── train_scunicorn.py              # Training script
│   ├── train.sh                         # Bash script for training
│   ├── generate_hr.py                  # Inference script
│   ├── inference.sh                     # Bash script for inference
│   ├── evaluate.py                     # Model evaluation script
├── utils/                             # Utility functions
│   ├── data_loader.py
│   ├── metrics.py
│   ├── visualization.py
│   ├── logger.py
├── trained_model/                     # Pre-trained models
├── logs/                               # Training and evaluation logs
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## Installation

### **Install Dependencies**
```bash
pip install -r requirements.txt
```

---

## Data Preprocessing
### **Step 1: Convert Raw Hi-C Data to .npy Format**
```bash
cd scripts
python3 process_hic_data.py --file_path ./../example_data/raw/Mouse_chr3_500kb.txt --output_path ./../example_data/processed/Mouse_chr3_500kb.npy
python3 process_hic_data.py --file_path ./../example_data/raw/mouse_chr11_50kb.png --output_path ./../example_data/processed/mouse_chr11_50kb.npy
cd ..
```
**Expected Output:**
- `Mouse_chr3_500kb.npy` saved in `processed/`
- `mouse_chr11_50kb.npy` saved in `processed/`

Alternatively, use the bash script:
```bash
cd scripts
bash process_hic_data.sh
```

### **Step 2: Extract Hi-C Patches for Training**
```bash
cd scripts
python3 makedata.py --input_dir ./../example_data/processed/ --output_folder ./../example_data/processed_patches/ --sub_mat_n 64 --chromosomes chr3 chr11
cd ..
```
**Expected Output:**
- `data_chr3_64.npy` and `data_chr11_64.npy` saved in `processed_patches/input_patches/`
- Training patches saved in `processed_patches/train_patches/`

Alternatively, use the bash script:
```bash
cd scripts
bash makedata.sh
```

---

## Training
### **Run Training with CLI Arguments**
```bash
cd scripts
torchrun --nproc_per_node=1 --master_port=4321 train_scunicorn.py \
  --epoch 10 --batch_size 8 --learning_rate 0.001 --data_train_path ./../example_data/processed_patches/train_patches/ \
  --resume_epoch 0 --num_gpu 1 --device_id 0 --num_workers 1 --loss mse --early_stoppage_epochs 5 --early_stoppage_start 100
cd ..
```
**Expected Output:**
- Training logs saved in `logs/`
- Checkpoints saved in `checkpoints/` as `scunicorn_epoch_*.pth`

Alternatively, use the bash script:
```bash
cd scripts
bash train.sh
```

---

## Model Evaluation
### **Run `evaluate.py`**
```bash
cd scripts
python3 evaluate.py --checkpoint_path ./../trained_model/scunicorn_epoch_10.pth --data_val_path ./../example_data/processed_patches/input_patches/data_chr3_64.npy
cd ..
```
**Expected Output:**
- Evaluation results printed and saved in `logs/`

---

## Inference
### **Run `generate_hr.py`**
```bash
cd scripts
python3 generate_hr.py --model_path ./../trained_model/scunicorn_epoch_10.npz --data_path ./../example_data/processed_patches/input_patches/data_chr3_64.npy --output_path ./../ScUnicorn_prediction/scunicorn_hr.npy
cd ..
```
**Expected Output:**
- High-resolution prediction saved as `ScUnicorn_prediction/scunicorn_hr.npy`

Alternatively, use the bash script:
```bash
cd scripts
bash inference.sh
```

---

## Pre-Trained Models & Data Availability

### Citation
Chandrashekar, M. B., Menon, R., Olowofila, S., & Oluwadare, O. (2025). Unicorn: Enhancing Single-Cell Hi-C Data with Blind Super-Resolution for 3D Genome Structure Reconstruction [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14750810

- The trained model is available on [Zenodo](https://zenodo.org/record/14750810).
- The dataset used for training is also available on [Zenodo](https://zenodo.org/record/14750810).

---

## Contributors
- M. B. Chandrashekar
- R. Menon
- S. Olowofila
- O. Oluwadare

---

## License
This project is licensed under the MIT License. See the LICENSE file for details.
