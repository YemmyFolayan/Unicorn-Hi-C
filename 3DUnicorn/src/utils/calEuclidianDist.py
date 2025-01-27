import numpy as np

def calEuclidianDist(x1, y1, z1, x2, y2, z2):
    """
    Calculate the Euclidean distance between two points in 3D space.
    
    Args:
    - x1, y1, z1 (float): Coordinates of the first point.
    - x2, y2, z2 (float): Coordinates of the second point.
    
    Returns:
    - output (float): Euclidean distance between the two points.
    """
    output = np.sqrt((x1 - x2)**2 + (y1 - y2)**2 + (z1 - z2)**2)
    return output
    