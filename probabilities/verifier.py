import os

def verify_npy_sizes(directory):
    # Initialize a list to store the sizes of npy files
    sizes = []
    paths = []

    # Walk through all subdirectories and files
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file == "probabilities.npy":
                file_path = os.path.join(root, file)
                try:
                    # Get the file size in bytes
                    file_size = os.path.getsize(file_path)

                    # Store path and size for final aligned output
                    paths.append((file_path, file_size))
                    sizes.append(file_size)

                except Exception as e:
                    print(f"Error reading size of {file_path}: {e}")

    # Determine the maximum path length for alignment
    max_path_length = max(len(path) for path, _ in paths) if paths else 0

    # Print paths and sizes with fixed-length alignment
    for path, size in paths:
        print(f"{path:<{max_path_length}} Size: {size} bytes")

    # Check if all sizes are the same
    if len(set(sizes)) > 1:
        print("\nWarning: Not all .npy files have the same size!")
    else:
        print("\nAll .npy files have the same size.")

if __name__ == "__main__":
    # Set the root directory where the 'probabilities' folder is located
    root_directory = "."  # Adjust this path if needed
    verify_npy_sizes(root_directory)
