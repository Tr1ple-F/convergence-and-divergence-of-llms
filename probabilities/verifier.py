import os
import numpy as np

def verify_probabilities(base_folder="."):
    all_shapes = set()
    inconsistent_paths = []

    # Walk through the directory structure
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if file == "probabilities.npy":
                file_path = os.path.join(root, file)
                # Load the numpy array and get its shape
                try:
                    arr = np.load(file_path)
                    shape = arr.shape
                    print(f"Path: {file_path}, Shape: {shape}")
                    all_shapes.add(shape)
                except Exception as e:
                    print(f"Could not load {file_path}. Error: {e}")

    # Check for inconsistencies in shapes
    if len(all_shapes) > 1:
        print("\nWarning: Inconsistent shapes found!")
        print(f"Shapes detected: {all_shapes}")
    else:
        print("\nAll files have the same shape.")

if __name__ == "__main__":
    verify_probabilities()
