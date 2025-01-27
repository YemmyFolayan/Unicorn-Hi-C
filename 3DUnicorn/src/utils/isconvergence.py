import numpy as np

def isconvergence(change, cost, NEAR_ZERO):
    """
    Check if the size of the gradient (change) is close to zero.
    
    Args:
    - change (np.ndarray): Array containing the gradient changes (derivatives).
    - cost (float): The current objective function value (cost).
    - NEAR_ZERO (float): Threshold to determine convergence.
    
    Returns:
    - converge (bool): Returns True if the gradient size is smaller than the threshold, indicating convergence.
    """
    # Calculate the sum of squared changes
    sq_change = np.square(change)
    SUM = np.sqrt(np.sum(sq_change))

    # Check if the sum is less than the convergence threshold
    if SUM < (NEAR_ZERO * abs(cost)):
        return True
    else:
        return False
