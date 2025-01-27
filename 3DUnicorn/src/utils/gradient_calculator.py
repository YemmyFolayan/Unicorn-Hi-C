import numpy as np
from utils.calEuclidianDist import calEuclidianDist
from utils.gradient_chain_calc import gradient_chain_calc

def gradient_calculator(lstCons, variables, dl_dw, dw_dv, maxIF, n):
    """
    Gradient Calculator function based on the provided MATLAB code.
    
    Args:
    - lstCons (ndarray): The list of constraints (i, j, IF, dist).
    - variables (ndarray): The current structure.
    - dl_dw (float): Gradient chain rule constant.
    - dw_dv (float): Gradient chain rule constant.
    - maxIF (float): The maximum interaction frequency.
    - n (int): The number of points in the structure.
    
    Returns:
    - change (ndarray): The calculated gradient for each point in the structure.
    - cost (float): The total cost for the current iteration.
    """
    # Initialize
    structure = variables
    change = np.zeros(len(structure))  # Gradient vector for each point
    val = 0.0  # Objective function value (used to compute cost)
    EPSILON = 1e-8  # Small constant to avoid division by zero
    
    # Calculate the chain rule derivative for the gradient calculation
    v, w, dl_dw, dw_dv = gradient_chain_calc(lstCons, structure, n)

    # Loop through the constraints
    for k in range(len(lstCons)):
        i, j, IF, dist = lstCons[k]

        if IF <= 0:
            continue
        
        # Adjust IF for adjacent points
        if abs(i - j) == 1:
            IF = 1.0 * maxIF

        # Ensure i and j are integers
        i = int(i)
        j = int(j)

        # Extract structure coordinates
        x1, y1, z1 = structure[i * 3 - 3:i * 3]
        x2, y2, z2 = structure[j * 3 - 3:j * 3]
        
        # Calculate structure distance
        str_dist = calEuclidianDist(x1, y1, z1, x2, y2, z2)
        z = str_dist - dist

        # Safeguard against divide by zero
        if str_dist < EPSILON:
            continue  # Skip this iteration if distance is zero

        # Calculate the derivative using chain rule
        tmp = dl_dw * dw_dv * 2 * (z / (str_dist + EPSILON))

        # Update objective function
        val += z ** 2

        # Update gradient values (change) using chain rule
        change[i * 3 - 3] += tmp * (x1 - x2)
        change[i * 3 - 2] += tmp * (y1 - y2)
        change[i * 3 - 1] += tmp * (z1 - z2)

        change[j * 3 - 3] += tmp * (x2 - x1)
        change[j * 3 - 2] += tmp * (y2 - y1)
        change[j * 3 - 1] += tmp * (z2 - z1)

    # Calculate cost
    cost = -(n / 2) - (n * np.log(np.sqrt(val / n)))

    # Handle NaN values in change
    change = np.nan_to_num(change, nan=0.0)

    return change, cost