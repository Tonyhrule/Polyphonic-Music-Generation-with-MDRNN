import numpy as np

# Specify the file path
file_path = "output/output.npy"

# Load the numpy array from the file
output_array = np.load(file_path)

# Print the contents of the array
print(output_array)
