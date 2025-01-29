# UniCorn: Unified System for scHi-C Data Enhancement and 3D Genome Structure Reconstruction

UniCorn integrates two powerful subsystems: **ScUnicorn** for super-resolution enhancement of single-cell Hi-C (scHi-C) data and **3DUnicorn** for 3D genome structure reconstruction. This system enables high-resolution chromatin modeling and structural inference from sparse and noisy Hi-C data.

---

## Features
- **ScUnicorn:** Blind Super-Resolution of scHi-C Data
  - No pre-defined LR-HR pairs required
  - Dynamic degradation kernel learning
  - Iterative restoration for enhanced resolution
- **3DUnicorn:** 3D Genome Structure Reconstruction
  - Converts Hi-C matrices into 3D structural models
  - Optimization-based chromatin modeling
  - Outputs `.pdb` structures and quantitative metrics

---

## 📂 Folder Structure
```
UniCorn/
├── scunicorn/                        # ScUnicorn subsystem
│   ├── configs/                      # Configuration files
│   ├── example_data/                  # Example dataset directory
│   ├── models/                        # Super-resolution model components
│   ├── scripts/                        # ScUnicorn execution scripts
│   ├── utils/                          # Utility functions
│   ├── trained_model/                  # Pre-trained models
│   ├── logs/                           # Training and evaluation logs
├── 3dunicorn/                         # 3DUnicorn subsystem
│   ├── examples/                      # Example input data and configurations
│   ├── src/                           # Python source code for structure modeling
│   ├── Scores/                        # Directory for saving output models
├── requirements.txt                    # Python dependencies
└── README.md                           # Project documentation
```

---

## 🚀 Installation
```bash
pip install -r requirements.txt
```

---

## 🔬 ScUnicorn: Data Preprocessing & Super-Resolution
### **Step 1: Convert Raw Hi-C Data to .npy Format**
```bash
cd scunicorn/scripts
python3 process_hic_data.py --file_path ./../example_data/raw/Mouse_chr3_500kb.txt --output_path ./../example_data/processed/Mouse_chr3_500kb.npy
python3 process_hic_data.py --file_path ./../example_data/raw/mouse_chr11_50kb.png --output_path ./../example_data/processed/mouse_chr11_50kb.npy
cd ../..
```
**Expected Output:**
- `Mouse_chr3_500kb.npy` saved in `processed/`
- `mouse_chr11_50kb.npy` saved in `processed/`

### **Step 2: Extract Hi-C Patches for Training**
```bash
cd scunicorn/scripts
python3 makedata.py --input_dir ./../example_data/processed/ --output_folder ./../example_data/processed_patches/ --sub_mat_n 64 --chromosomes chr3 chr11
cd ../..
```

---

## 🏗 3DUnicorn: 3D Genome Reconstruction
### **Running 3DUnicorn**
```bash
cd 3dunicorn/src
python3 main.py --parameters parameters.txt
```

**Configuration (`parameters.txt`):**
- `NUM`: Number of models to generate
- `OUTPUT_FOLDER`: Path to save results
- `INPUT_FILE`: Path to Hi-C contact file (tuple or matrix format)
- `CONVERT_FACTOR`: distance = 1 / (IF) ^ factor
- `LEARNING_RATE`: Optimization step size
- `MAX_ITERATION`: Maximum optimization iterations

**Expected Output:**
- `*.pdb`: 3D reconstructed genome structure
- `*_Finalscores.txt`: Summary of best model metrics
- `*_pearsoncorr.txt`: Pearson correlation for models
- `*_rmsd.txt`: RMSD values for all models

---

## 📈 Model Evaluation & Inference
### **ScUnicorn Inference**
```bash
cd scunicorn/scripts
python3 generate_hr.py --model_path ./../trained_model/scunicorn_epoch_10.npz --data_path ./../example_data/processed_patches/input_patches/data_chr3_64.npy --output_path ./../ScUnicorn_prediction/scunicorn_hr.npy
cd ../..
```
**Expected Output:**
- High-resolution prediction saved as `ScUnicorn_prediction/scunicorn_hr.npy`

### **3DUnicorn Structure Generation**
```bash
cd 3dunicorn/src
python3 main.py --parameters parameters.txt
```

---

## 📚 Pre-Trained Models & Data Availability
### Citation
Chandrashekar, M. B., Menon, R., Olowofila, S., & Oluwadare, O. (2025). Unicorn: Enhancing Single-Cell Hi-C Data with Blind Super-Resolution for 3D Genome Structure Reconstruction [Data set]. Zenodo. https://doi.org/10.5281/zenodo.14750810

- The trained model is available on [Zenodo](https://zenodo.org/record/14750810).
- The dataset used for training is also available on [Zenodo](https://zenodo.org/record/14750810).

---

## 👨‍💻 Contributors
- M. B. Chandrashekar
- R. Menon
- S. Olowofila
- O. Oluwadare

---

## 📜 License
This project is licensed under the MIT License. See the LICENSE file for details.