import numpy as np
from scipy.stats import spearmanr, pearsonr
from utils.convert2xyz import convert2xyz  # This should convert variables to XYZ
from utils.mat2pdb import mat2pdb  # This should save the XYZ data to PDB format

def output_3DMax(variables, str_name, wish_dist, dist):
    """
    Outputs the 3DMax results, including Spearman and Pearson correlations,
    and saves the structure as a scaled .pdb file.

    Args:
    - variables: The final variables after optimization.
    - str_name: Name of the structure file (used for output).
    - wish_dist: Target distances.
    - dist: Computed distances.
    """
    # Calculate Spearman and Pearson correlations
    spearman_corr, _ = spearmanr(wish_dist, dist)
    pearson_corr, _ = pearsonr(wish_dist, dist)

    variables = variables.flatten() if isinstance(variables, np.ndarray) and variables.ndim > 1 else variables

    # Convert variables to XYZ format for PDB output
    xyz = convert2xyz(variables, len(variables) // 3)

    # Scale the structure to increase size (to fit within a 100-unit box, as per MATLAB)
    max_abs_value = np.max(np.abs(xyz))

    scale_factor = 100 / np.max(np.abs(xyz))
    #print(f"Calculated scale factor: {scale_factor}")
    #print(f"Max absolute coordinate value used for scaling in Python: {max_abs_value}")

    xyz_scaled = xyz * scale_factor
    #print("First few scaled coordinates:\n", xyz_scaled[:5])

    # Prepare data for PDB output
    data = {
        'X': xyz_scaled[:, 0],
        'Y': xyz_scaled[:, 1],
        'Z': xyz_scaled[:, 2],
        'outfile': f"{str_name}.pdb"
    }

    # Delete any existing file and save new PDB
    try:
        mat2pdb(data)
        print(f"PDB file saved as {data['outfile']}")
    except Exception as e:
        print(f"Error while saving PDB file: {e}")

    #print(f"\nThe representative structure is {data['outfile']}")
