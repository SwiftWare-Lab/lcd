"""
Script to transfer selected matrices from 'extracted_matrices/' to 'matrices'

Requirements:
- 'extracted_matrices/' contains 'selected_matrices.txt'
- 'selected_matrices.txt' contains a relative path to a .mtx file for each of its line

prachyo 05-11-24

"""

import os
import shutil

selected_matrices_file = 'extracted_matrices/selected_matrices.txt'
source_dir = 'extracted_matrices'
target_dir = 'mtx_matrices'

os.makedirs(target_dir, exist_ok=True)

shutil.copy(selected_matrices_file, target_dir)

with open(selected_matrices_file, 'r') as f:
    selected_matrices = [line.strip() for line in f]

for matrix_path in selected_matrices:
    full_source_path = os.path.join(source_dir, matrix_path)

    if os.path.isfile(full_source_path):
        shutil.move(full_source_path, target_dir)
    else:
        print(f"File {matrix_path} not found in {source_dir}")

print("Transfer complete")