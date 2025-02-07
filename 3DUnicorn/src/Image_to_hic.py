import cv2
import numpy as np
import os

# Define static file paths
image_path = '/Users/mohanchandru/Documents/SCL_3dMax_ZSSR/3dmax_python_new/output/chr11_hr.png'
hic_output_path = '/Users/mohanchandru/Documents/SCL_3dMax_ZSSR/3DUnicorn/examples/input/chr11_hr_matrix.txt'
tuple_output_path = '/Users/mohanchandru/Documents/SCL_3dMax_ZSSR/3DUnicorn/examples/input/chr11_hr_matrix_tuple.txt'

print("connected script is executing")
# Ensure output directory exists
os.makedirs(os.path.dirname(hic_output_path), exist_ok=True)

# Load the RGB image
image = cv2.imread(image_path)

# Check the shape of the image to confirm it is an RGB image
print("Image shape: {}".format(image.shape))

# Extract one of the color channels (R, G, or B)
# Here we use the Red channel
hic_matrix = image[:, :, 0]

# Rescale the values from 0-255 to 0-1
hic_matrix_normalized = hic_matrix / 255.0

# Print the shape of the resulting Hi-C contact matrix
print("Hi-C contact matrix shape: {}".format(hic_matrix_normalized.shape))

# Expand dimensions to add batch size and channel
hic_matrix_expanded = np.expand_dims(hic_matrix_normalized, axis=0)  # Add batch dimension
hic_matrix_expanded = np.expand_dims(hic_matrix_expanded, axis=-1)  # Add channel dimension

# Print the shape of the expanded Hi-C contact matrix
print("Expanded Hi-C contact matrix shape: {}".format(hic_matrix_expanded.shape))

# Save the Hi-C contact matrix to a file
np.savetxt(hic_output_path, hic_matrix_normalized)
print("Hi-C contact matrix saved to {}".format(hic_output_path))

# Load the normalized matrix generated from the image
matrix = np.loadtxt(hic_output_path)


def convert_dense_to_tuple_debug(hic_output_path, tuple_output_path, bin_size=500000):
    with open(hic_output_path, 'r') as file:
        lines = file.readlines()

    non_zero_found = 0  # Counter for non-zero entries found

    with open(tuple_output_path, 'w') as out_file:
        for row_index, line in enumerate(lines):
            values = line.strip().split()  # Split the line into columns
            for col_index, value in enumerate(values):
                try:
                    frequency = float(value)
                    if frequency != 0:  # Only consider non-zero entries
                        position_1 = (row_index + 1) * bin_size
                        position_2 = (col_index + 1) * bin_size
                        out_file.write(f"{position_1} {position_2} {frequency}\n")
                        
                        # Debugging output for the first few non-zero values
                        if non_zero_found < 5:
                            print(f"Non-zero entry found: {position_1} {position_2} {frequency}")
                            non_zero_found += 1
                except ValueError:
                    print(f"Warning: Could not convert value '{value}' at row {row_index + 1}, col {col_index + 1}")

    #print(f"Conversion complete. Output saved to {output_file}")
    print(f"Tuples saved to {tuple_output_path}")

    print(f"Total non-zero entries found: {non_zero_found}")


convert_dense_to_tuple_debug(hic_output_path, tuple_output_path)
