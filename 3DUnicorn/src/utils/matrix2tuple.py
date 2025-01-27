import numpy as np

def matrix2tuple(cont):
    """
    Converts an input square matrix to a tuple format.
    
    Args:
    - cont (np.ndarray): Input square matrix (numpy array).
    
    Returns:
    - tuple_list (np.ndarray): Output list of tuples, where each tuple contains
                               (row, col, IF) corresponding to non-diagonal elements
                               of the input matrix.
    """
    len_matrix = cont.shape[0]  # Assuming a square matrix
    tuple_list = []

    # Loop through the matrix to convert it into tuple format
    for row in range(len_matrix):
        for col in range(row + 1, len_matrix):  # Avoid diagonal and duplicate entries
            IF = cont[row, col]
            tuple_list.append([row + 1, col + 1, IF])  # Use 1-based indexing like MATLAB

    return np.array(tuple_list)

# Example usage
if __name__ == "__main__":
    # Example input matrix
    cont = np.array([
        [0, 0.5, 0.8],
        [0.5, 0, 0.7],
        [0.8, 0.7, 0]
    ])
    
    tuple_result = matrix2tuple(cont)
    print("Converted Tuple Format:\n", tuple_result)