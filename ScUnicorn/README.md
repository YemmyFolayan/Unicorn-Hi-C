## 1. Important Notice

Due to GitHub’s file size limitations, the `train.npz`, `valid.npz`, and `test.npz` files could not be uploaded to the repository.  
You can **download them from Zenodo** using the following link:
[Download ScUnicorn Dataset](https://zenodo.org/uploads/15079331?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRkYzI5MmQwLTdmYWQtNGIwNi04YWIzLWU4ZmViYzMxNmNlOSIsImRhdGEiOnt9LCJyYW5kb20iOiJlMmI0NmNhYjliNWY5ZjA2N2I5ZThkN2EwMDgzODk3ZCJ9.Ql-dXIRmoFgjZXe4Psw3G-mv_uAmM8bqLrfKhNC92PdoLPgCKEIKaRob73gZrYcNV7hW9Bc3XF_pk6ml8fL22A)

### **1.1 Downloading the Dataset**
Alternatively, you can download the dataset directly using **wget**:
```bash
wget "https://zenodo.org/uploads/15079331?token=eyJhbGciOiJIUzUxMiJ9.eyJpZCI6ImRkYzI5MmQwLTdmYWQtNGIwNi04YWIzLWU4ZmViYzMxNmNlOSIsImRhdGEiOnt9LCJyYW5kb20iOiJlMmI0NmNhYjliNWY5ZjA2N2I5ZThkN2EwMDgzODk3ZCJ9.Ql-dXIRmoFgjZXe4Psw3G-mv_uAmM8bqLrfKhNC92PdoLPgCKEIKaRob73gZrYcNV7hW9Bc3XF_pk6ml8fL22A" -O ScUnicorn_Dataset.zip
```

### **1.2 Where to Place the Downloaded Files**
Once downloaded, place the files in the `data/` directory as follows:

```
ScUnicorn/
│
├── data/
│   ├── train.npz    # Training dataset
│   ├── valid.npz    # Validation dataset
│   ├── test.npz     # Testing dataset
│
```

After placing the files correctly, you will get the folder structure as seen above. You can then proceed with the training instructions below.
