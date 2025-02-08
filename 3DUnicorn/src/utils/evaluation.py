import numpy as np
from scipy.stats import spearmanr, pearsonr
from utils.calEuclidianDist import calEuclidianDist

def evaluate_structure(lstCons, structure, n):
    """
    Evaluate the structure by calculating RMSE, Spearman correlation, and Pearson correlation.
    
    Args:
    - lstCons (ndarray): The list of constraints, each entry is a tuple (i, j, IF, dist).
    - structure (ndarray): The optimized structure.
    - n (int): The number of points in the structure.
    
    Returns:
    - rmse (float): Root mean square error.
    - SpearmanRHO (float): Spearman correlation coefficient.
    - PearsonRHO (float): Pearson correlation coefficient.
    - WishDist_clean, Dist_clean (arrays): Cleaned target and computed distances for further use.
    """
    
    # Initialize SUM for RMSE calculation and arrays for distances
    SUM = 0.0
    Len = n * (n - 1) // 2  # Number of pairwise combinations
    Dist = np.zeros(Len)
    WishDist = np.zeros(Len)
    count = 0

    # Loop over constraints to populate Dist and WishDist
    for k in range(len(lstCons)):
        i, j, IF, dist = lstCons[k]
        i, j = int(i), int(j)  # Ensure indices are integers for slicing
        
        # Structure distance calculation between points i and j
        x1, y1, z1 = structure[i * 3 - 3: i * 3]
        x2, y2, z2 = structure[j * 3 - 3: j * 3]
        
        str_dist = calEuclidianDist(x1, y1, z1, x2, y2, z2)
        SUM += (str_dist - dist) ** 2  # Add to RMSE calculation

        # Populate Dist and WishDist if criteria are met
        if i != j and IF > 0 and count < Len:
            Dist[count] = str_dist
            WishDist[count] = dist
            count += 1

    # Calculate RMSE
    SUM /= len(lstCons)
    rmse = np.sqrt(SUM)

    # Validate distances to ensure they're usable for correlation calculations
    Dist_clean = Dist[:count]
    WishDist_clean = WishDist[:count]

    # Check for NaN or infinite values in Dist and WishDist
    if np.isnan(Dist_clean).any() or np.isinf(Dist_clean).any():
        print("Warning: Dist_clean contains NaN or infinite values")
    if np.isnan(WishDist_clean).any() or np.isinf(WishDist_clean).any():
        print("Warning: WishDist_clean contains NaN or infinite values")

    # Calculate Spearman and Pearson correlations with cleaned data
    SpearmanRHO, _ = spearmanr(Dist_clean, WishDist_clean)
    PearsonRHO, _ = pearsonr(Dist_clean, WishDist_clean)

    return rmse, SpearmanRHO, PearsonRHO, WishDist_clean, Dist_clean
