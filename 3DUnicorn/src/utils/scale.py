import numpy as np

def scale_coordinates_and_save(structure, size_cube=600, outfile="reconstructed.pdb", convert="no"):
    """
    Scale the reconstructed coordinates and save them to a .pdb file.

    Args:
    - structure (np.ndarray): Array of reconstructed 3D coordinates.
    - size_cube (int): Target size for scaling the coordinates.
    - outfile (str): Name of the .pdb output file.
    - convert (str): Whether to convert the coordinates to the target range.

    Returns:
    - scaled_coordinates (np.ndarray): Scaled 3D coordinates.
    """
    max_coord = np.max(structure)
    ratio = max_coord / size_cube if convert == "yes" else 1.0
    print(f"Scaling coordinates with ratio: {ratio:.4f}")

    # Apply scaling
    scaled_coordinates = structure / ratio

    # Save scaled coordinates to a text file for reference
    np.savetxt("scaled_coordinates.txt", scaled_coordinates, fmt="%.3f")
    print(f"Scaled coordinates saved to scaled_coordinates.txt.")

    # Save coordinates to a .pdb file using mat2pdb logic
    from utils.mat2pdb import mat2pdb

    input_data = {
        "X": scaled_coordinates[:, 0],
        "Y": scaled_coordinates[:, 1],
        "Z": scaled_coordinates[:, 2],
        "outfile": outfile
    }
    mat2pdb(input_data)
    print(f"PDB file saved as {outfile}.")

    return scaled_coordinates

def scale_coordinates(coordinates, size_cube):
    """
    Scale coordinates and apply smoothing.
    """
    max_val = np.max(coordinates)
    scaling_factor = size_cube / max_val if max_val != 0 else 1
    scaled_coordinates = coordinates * scaling_factor

    # Apply smoothing to coordinates
    smoothed_coordinates = np.zeros_like(scaled_coordinates)
    for i in range(1, len(scaled_coordinates)):
        smoothed_coordinates[i] = 0.8 * scaled_coordinates[i] + 0.2 * scaled_coordinates[i - 1]

    return smoothed_coordinates