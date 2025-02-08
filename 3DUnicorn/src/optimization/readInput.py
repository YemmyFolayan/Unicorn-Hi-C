import numpy as np
import os
from utils.matrix2tuple import matrix2tuple

def read_input(INPUT_FILE, output_path):
    """
    Load and process the contact matrix from a file.
    
    Args:
    - INPUT_FILE (str): Path to the input file.
    - output_path (str): Path to the output directory.
    
    Returns:
    - lstCons (ndarray): Processed contact list.
    - n (int): Number of unique positions.
    - mapping (ndarray): Mapping of positions to absolute ids.
    """
    
    # Read the file
    filename = INPUT_FILE
    cont = np.loadtxt(filename)
    
    # Check if input is in tuple or square matrix format
    if cont.shape[1] == 3:
        print('Input is in Tuple format')
    else:
        print('Input is in Square Matrix format')
        cont = np.loadtxt(filename)
        # Convert to tuple format
        cont = matrix2tuple(cont)
        print('Conversion to Tuple Format Done successfully')
    
    pos = []
    
    # Remove zero contacts and NaN, and exclude self-interactions
    ind_greater_IF = np.where((cont[:, 2] > 0) & (~np.isnan(cont[:, 2])) & (cont[:, 0] != cont[:, 1]))
    contact = cont[ind_greater_IF]
    pos = np.hstack((contact[:, 0], contact[:, 1]))

    # lstCons is the list of filtered contacts
    lstCons = contact
    
    # Sort and get unique positions
    pos = np.unique(pos)
    
    # Number of unique positions
    n = len(pos)
    
    # Map positions to absolute ids
    mapping = np.zeros((n, 2))
    for i in range(n):
        mapping[i, 0] = pos[i]
        mapping[i, 1] = i + 1  # 1-based index, consistent with MATLAB
    
    # Save mapping to file
    fname = os.path.basename(filename).split('.')[0]
    mapname = os.path.join(output_path, f'{fname}_coordinate_mapping.txt')
    np.savetxt(mapname, mapping, fmt='%d')
    
    # Create a mapping dictionary for position to id
    Map = dict(zip(mapping[:, 0], mapping[:, 1]))
    
    # Replace positions with their mapped IDs
    for i in range(len(lstCons)):
        lstCons[i, 0] = Map[lstCons[i, 0]]
        lstCons[i, 1] = Map[lstCons[i, 1]]

    return lstCons, n, mapping
