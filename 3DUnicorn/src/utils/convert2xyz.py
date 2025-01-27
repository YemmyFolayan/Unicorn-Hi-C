import numpy as np

def convert2xyz(variables, n):
    """
    Converts flattened variables into an n x 3 matrix for 3D coordinates.

    Args:
    - variables: A flat 1D array of length 3 * n.
    - n: Number of points.

    Returns:
    - xyz: A (n x 3) numpy array representing 3D coordinates.
    """
    # Ensure `variables` is a numpy array
    variables = np.asarray(variables, dtype=float)

    if variables.ndim != 1:
        raise ValueError("`variables` must be a flat 1D array.")

    if len(variables) != 3 * n:
        raise ValueError(f"Length of `variables` ({len(variables)}) does not match expected length (3 * {n} = {3 * n}).")

    xyz = np.zeros((n, 3))
    for i in range(n):
        xyz[i, 0] = variables[i * 3]      # x-coordinate
        xyz[i, 1] = variables[i * 3 + 1]  # y-coordinate
        xyz[i, 2] = variables[i * 3 + 2]  # z-coordinate
    return xyz