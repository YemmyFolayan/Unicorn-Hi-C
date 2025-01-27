# Unicorn: Enhancing Single-Cell Hi-C Data with Blind Super-Resolution for 3D Genome Structure Reconstruction
Unicorn: Enhancing Single-Cell Hi-C Data with Blind Super-Resolution for 3D Genome Structure Reconstruction

**3DUnicorn**

## 1. Content of Folders:

- **examples**: Contains example input data and configurations for running the 3DUnicorn algorithm.
- **src**: Python source code for the 3DUnicorn algorithm, including utilities for input processing, distance matrix conversion, structure optimization, and output generation.
- **Scores**: Output folder where the results are saved, including `.pdb` files and evaluation metrics.

## 2. Input Matrix File Format:

3DUnicorn supports input formats for Hi-C data:

1. **Tuple Input Format**:  
   A Hi-C contact file where each line contains three numbers separated by spaces:
   position_1 position_2 interaction_frequencies

## 3. Usage:

### 3.1 Python:

	To run the Python version of 3DUnicorn, follow these steps:
	
	1. Open the command line in the `src` directory of the project.
	2. Run the Python script using the following command:
		python3 main.py --parameters parameters.txt

  	Configuration (parameters.txt):

	Parameters are configured in the parameters.txt file as follows:
		•	NUM: Number of models to generate.
		•	OUTPUT_FOLDER: Path to the output directory where results will be stored.
		•	INPUT_FILE: Path to the Hi-C contact file in either tuple format or square matrix format.
		•	CONVERT_FACTOR: The factor used to convert interaction frequencies (IF) to distances, computed as:
		[
		\text{distance} = \frac{1}{(\text{IF})^{\text{factor}}}
		]
	If not specified, the program searches for it in the range [0.1, 2.0] with a step size of 0.1.
		•	CHROMOSOME_LENGTH: For datasets containing multiple chromosomes, specify the number of points (or beads) per 				chromosome in a comma-separated list. This must align with the input data. For single chromosomes, omit this 				parameter.
		•	VERBOSE: Set to true or false to control the output of gradient values during optimization.
		•	LEARNING_RATE: The learning rate for optimization. Increase this value to reduce runtime (maximum recommended value: 			1).
		•	MAX_ITERATION: Maximum number of optimization iterations. The process may converge before reaching this limit.
	
	Example:
	
	Refer to the examples/ folder for sample input files and configurations.

## 4. Output

3DUnicorn generates the following output files:

	- *.pdb:
	Contains the 3D reconstructed genome structure. This file can be visualized using tools such as PyMOL, Chimera, or 			GenomeFlow.
 
	- *_Finalscores.txt:
	Summarizes the best model generated, including Spearman correlation, Pearson correlation, and other key metrics.
 
	- *_pearsoncorr.txt:
	Lists the Pearson correlation values for all generated models.
 
	- *_rmsd.txt:
	Contains the Root Mean Square Deviation (RMSD) values for all generated models.
