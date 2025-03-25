## Folder Structure
```
ScUnicorn/
│
├── configs/                          # Configuration files
│   ├── default_config.yaml           # Default model and training settings
│
├── models/                           # Model components
│   ├── degradation_kernel.py         # Degradation kernel functions
│   ├── restoration_loop.py           # Iterative restoration module
│   ├── scunicorn.py                  # Main model integration
│
├── output/                           # Output of generate_hr.py will be stored here
│
├── scripts/                          # Execution scripts
│   ├── generate_hr.py                # High-resolution generation script
│   ├── training/                     # Model training scripts
│   │   ├── train.py                  # Training script for ScUnicorn
│   │   ├── infer.py                  # Inference script for trained model
│
├── checkpoint/                       # Model checkpoints
│   ├── scunicorn_model.pytorch       # Saved ScUnicorn model checkpoint
│
├── utils/                            # Utility functions
│   ├── data_loader.py                # Data loading utilities
│   ├── metrics.py                    # Performance evaluation metrics
│   ├── visualization.py              # Visualization tools
│   ├── logger.py                     # Logging functions
│
├── data/                             # Dataset storage
│   ├── train.npz                     # Training dataset
│   ├── valid.npz                     # Validation dataset
│   ├── test.npz                      # Testing dataset
│   ├── mouse_test_data/              # Sample test data
│   │   ├── chr3_100kb.txt
│   │   ├── chr11_100kb.txt
│   │   ├── chr3_500kb.txt
│   │   ├── chr11_500kb.txt
│
├── requirements.txt                   # Python dependencies
├── README.md                          # Project documentation
```

## Training Instructions

To train the ScUnicorn model, navigate to the `scripts/training` directory and run:

```bash
python3 train.py --train_data ../../data/train.npz --valid_data ../../data/valid.npz --epochs 50 --batch_size 64 --lr 0.0003
```

This command will:
- Load training data from `data/train.npz`
- Load validation data from `data/valid.npz`
- Train the model for 50 epochs with a batch size of 64
- Save the trained model in `checkpoint/scunicorn.pth`

## Inference Instructions

To run inference using the trained ScUnicorn model, navigate to the `scripts/training` directory and run:

```bash
python3 infer.py --input ../../data/test.npz --checkpoint ../../checkpoint/scunicorn_model.pytorch --output ../../output/
```

This command will:
- Load the trained model from `checkpoint/scunicorn_model.pytorch`
- Run inference on test data from `data/test.npz`
- Save the output predictions in the `output/` directory

## Generating HR Maps

To generate high-resolution (HR) Hi-C maps using the trained ScUnicorn model, navigate to the `scripts/` directory and run:

```bash
python3 generate_hr.py --model_path ../checkpoint/scunicorn_model.pytorch --data_path ../data/mouse_test_data/chr3_100kb.txt --output_image_path ../output/output.png --output_hic_path ../output/output.txt
```

This command will:
- Load the trained model from `checkpoint/scunicorn_model.pytorch`
- Use the input Hi-C file `data/mouse_test_data/chr3_100kb.txt`
- Generate an HR Hi-C map and save it as an image in `output/output.png`
- Save the HR Hi-C matrix in `output/output.txt`
