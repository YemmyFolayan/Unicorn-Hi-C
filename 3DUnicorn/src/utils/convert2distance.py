import numpy as np

def convert_to_distance(lstCons, CONVERT_FACTOR, seed=42):
    """
    Convert interaction frequencies (IF) to distances using the given conversion factor.
    
    Args:
    - lstCons (ndarray): A 2D array where each row is [x, y, IF].
    - CONVERT_FACTOR (float): Factor used to convert IF to distance.
    - seed (int): Random seed for reproducibility (default is 42).
    
    Returns:
    - lstCons (ndarray): Updated array with distances in the 4th column.
    - maxIF (float): The maximum interaction frequency.
   """
    # Set random seed to match MATLAB randomness
    np.random.seed(seed)
    
    AVG_DIST = 10.0  # An arbitrary distance

    # Step 1: Find the average IF
    avgIF = np.mean(lstCons[:, 2])  # Assuming IF is the 3rd column in lstCons

    # Step 2: Normalize IF by avgIF
    lstCons[:, 2] /= avgIF

    # Step 3: Calculate distances and track the max IF
    IFs = lstCons[:, 2]
    dist = 1.0 / (IFs ** CONVERT_FACTOR)
    avgDist = np.mean(dist)
    maxIF = np.max(IFs)

    # Step 4: Find adjacent position IFs and calculate avgAdjIF
    adjacent_mask = np.abs(lstCons[:, 0] - lstCons[:, 1]) == 1
    avgAdjIF = np.mean(IFs[adjacent_mask]) if np.any(adjacent_mask) else 0

    # Ensure maxIF is at least avgAdjIF
    maxIF = min(avgAdjIF, maxIF)

    # Step 5: Add missing adjacent pairs if none exist
    n = int(np.max(lstCons[:, :2]))  # Maximum node index
    i = 1
    while i < n:
        f = np.where((lstCons[:, 0] == i) & (lstCons[:, 1] == i + 1))[0]
        if len(f) == 0:
            # Missing adjacent pair, add it with avgAdjIF
            newCons = np.array([[i, i + 1, avgAdjIF]])
            lstCons = np.vstack([lstCons, newCons])
        else:
            # Adjust IF if it's below avgAdjIF
            if lstCons[f, 2] < avgAdjIF:
                lstCons[f, 2] = avgAdjIF
        i += 1
    
    print(f"Added missing adjacent constraints... Total constraints: {len(lstCons)}")

    # Step 6: Compute distances and store them in the 4th column
    lstCons = np.hstack([lstCons, np.zeros((lstCons.shape[0], 1))])  # Add 4th column for distance
    lstCons[:, 3] = AVG_DIST / ((lstCons[:, 2] ** CONVERT_FACTOR) * avgDist)
    maxD = np.max(lstCons[:, 3])

    print(f"Max distance is: {maxD}")

    return lstCons, maxIF
