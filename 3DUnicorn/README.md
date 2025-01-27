# 3DUnicorn: For 3D Genome Structure Reconstruction
3DUnicorn enhances 3D genome reconstruction from single-cell Hi-C data using blind super-resolution and optimization for accurate, high-resolution chromatin modeling.

**3DUnicorn**  

---

## 📂 Folder Structure  

- **`examples/`**: Example input data and configurations for running 3DUnicorn.  
- **`src/`**: Python source code, including utilities for:  
  - Input processing  
  - Distance matrix conversion  
  - Structure optimization  
  - Output generation  
- **`Scores/`**: Directory for saving results, including `.pdb` files.  

---

## 📊 Input File Format  

### 1️⃣ **Tuple Input Format**  
A Hi-C contact file with rows in the following format:  
`position_1 position_2 interaction_frequencies`  

---

## 🚀 Usage  

### 🐍 **Running 3DUnicorn (Python)**  

1. Open the command line in the `src/` directory.  
2. Execute the following command:  

```bash
python3 main.py --parameters parameters.txt

🔧 Configuration (parameters.txt)

Define the following parameters in the parameters.txt file:
	•	NUM: Number of models to generate.
	•	OUTPUT_FOLDER: Path to save results.
	•	INPUT_FILE: Path to the Hi-C contact file (tuple or square matrix format).
	•	CONVERT_FACTOR: distance = 1 / (IF) ^ factor		
	•	The program searches for factor within [0.1, 2.0] (default step size: 0.1) if not specified.
	•	CHROMOSOME_LENGTH: For multiple chromosomes, provide a comma-separated list of bead counts per chromosome (align with input 		data). Omit for single chromosomes.
	•	VERBOSE: true or false for controlling gradient output during optimization.
	•	LEARNING_RATE: Adjust the optimization step size (max recommended: 1).
	•	MAX_ITERATION: Maximum optimization iterations (may converge earlier).


🔍 Example

Refer to the examples/ folder for sample inputs and configurations.

📤 Output

3DUnicorn generates the following files:
	•	*.pdb: 3D reconstructed genome structure. Visualize using tools like PyMOL, Chimera, or GenomeFlow.
	•	*_Finalscores.txt: Summarizes the best model generated, including Spearman correlation, Pearson correlation, and other 			key metrics.
	•	*_pearsoncorr.txt: Lists the Pearson correlation values for all generated models.
	•	*_rmsd.txt: Contains the Root Mean Square Deviation (RMSD) values for all generated models.

