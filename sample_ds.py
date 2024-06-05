import os
import shutil
import random

# Define paths
base_path = 'results_problem_1'
output_base_path = 'results_problem_1_small_sample'

# Number of samples to select
samples_to_select = {
    'train': {'0': 50000, '1': 50000},
    'val': {'0': 1000, '1': 1000},
    'test': {'0': 1000, '1': 1000}
}

# Ensure the output base path exists
os.makedirs(output_base_path, exist_ok=True)

# Function to randomly select files and copy them to the new directory
def sample_and_copy(data_type, class_label, num_samples):
    src_folder = os.path.join(base_path, data_type, class_label)
    dest_folder = os.path.join(output_base_path, data_type, class_label)
    
    # Create destination directory if it does not exist
    os.makedirs(dest_folder, exist_ok=True)
    
    # Get all files in the source directory
    files = os.listdir(src_folder)
    
    # Randomly sample the specified number of files
    sampled_files = random.sample(files, num_samples)
    
    # Copy each sampled file to the destination directory
    for file_name in sampled_files:
        src_file_path = os.path.join(src_folder, file_name)
        dest_file_path = os.path.join(dest_folder, file_name)
        shutil.copy(src_file_path, dest_file_path)

# Process each data type (train, val, test)
for data_type in samples_to_select:
    for class_label in samples_to_select[data_type]:
        num_samples = samples_to_select[data_type][class_label]
        sample_and_copy(data_type, class_label, num_samples)
        print(class_label)
    print(data_type)

print("Sampling and copying completed.")

