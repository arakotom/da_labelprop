import os

import os

def rename_npz_files(directory_path):
    # Walk through the directory tree
    for root, dirs, files in os.walk(directory_path):
        for filename in files:
            # Check if the file ends with .npz
            if filename.endswith('.npz'):
                # Construct the new filename
                new_filename = filename.replace('.npz', '-method-fix.npz')
                # Construct full file paths
                old_file = os.path.join(root, filename)
                new_file = os.path.join(root, new_filename)
                # Rename the file
                os.rename(old_file, new_file)
                print(f'Renamed: {old_file} to {new_file}')

# Example usage
directory_path = './results/'
rename_npz_files(directory_path)
