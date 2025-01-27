import numpy as np
from utils.calEuclidianDist import calEuclidianDist

def gradient_chain_calc(lstCons, structure, n):
    """
    Calculate v, w, dl_dw, and dw_dv for the gradient calculation.

    Args:
    - lstCons (np.ndarray): List of constraints, where each row contains (i, j, IF, dist).
    - structure (np.ndarray): Array containing the structure's (x, y, z) coordinates.
    - n (int): Number of points.

    Returns:
    - v (float): Sum of squared differences between structure distances and IF distances.
    - w (float): Square root of (v/n), representing the distance metric.
    - dl_dw (float): Derivative of loss with respect to w.
    - dw_dv (float): Derivative of w with respect to v.
    """
    v = 0

    # Loop through existing constraints (lstCons)
    for k in range(len(lstCons)):
        #print("lstCons shape:", lstCons.shape)
        #print("Sample row from lstCons:", lstCons[0])

        i, j, IF, dist = lstCons[k]
        
        if IF <= 0:
            continue

        # Convert i and j from 1-based (MATLAB) to 0-based (Python) indices
        i = int(i) - 1
        j = int(j) - 1

        # Ensure that i and j are valid indices within the structure array
        if i * 3 + 2 >= len(structure) or j * 3 + 2 >= len(structure):
            raise IndexError(f"Index out of bounds for i={i+1}, j={j+1}. Check structure array size.")

        # Extract (x, y, z) coordinates of points i and j from the structure
        x1, y1, z1 = structure[i * 3:i * 3 + 3]  # Coordinates for point i
        x2, y2, z2 = structure[j * 3:j * 3 + 3]  # Coordinates for point j

        # Calculate Euclidean distance between the two points
        str_dist = calEuclidianDist(x1, y1, z1, x2, y2, z2)

        # Calculate squared difference between structure distance and IF distance
        z = (str_dist - dist) ** 2
        v += z

    # Calculate w
    w = np.sqrt(v / n)

    # Calculate dl_dw and dw_dv
    dl_dw = -n / w
    dw_dv = 1 / (2 * np.sqrt(n * v))

    return v, w, dl_dw, dw_dv