import os
import shutil

problems = ['results_problem_1_sample', 'results_problem_20_sample', 'results_problem_21_sample']
categories = ['train', 'test', 'val']
labels = ['0', '1']

# Create the target directories if they don't exist
target_base = './all_problems'
for category in categories:
    for label in labels:
        os.makedirs(os.path.join(target_base, category, label), exist_ok=True)

# Function to count files in a directory
def count_files(directory):
    return sum([len(files) for _, _, files in os.walk(directory)])

# Initial counts from each problem directory
initial_counts = {category: {label: 0 for label in labels} for category in categories}
for problem in problems:
    for category in categories:
        for label in labels:
            dir_path = os.path.join('.', problem, category, label)
            initial_counts[category][label] += count_files(dir_path)

# Copy files from each problem's directories to the target directories
for problem in problems:
    for category in categories:
        for label in labels:
            source_dir = os.path.join('.', problem, category, label)
            target_dir = os.path.join(target_base, category, label)
            for filename in os.listdir(source_dir):
                source_file = os.path.join(source_dir, filename)
                target_file = os.path.join(target_dir, f"{problem}_{filename}")  # Prevent filename conflicts
                try:
                    shutil.copy(source_file, target_file)
                except Exception as e:
                    print(f"Error copying {source_file} to {target_file}: {e}")

# Final counts in the all_problems directory
final_counts = {category: {label: count_files(os.path.join(target_base, category, label)) for label in labels} for category in categories}

# Compare initial and final counts
for category in categories:
    for label in labels:
        if initial_counts[category][label] == final_counts[category][label]:
            print(f"Check passed for {category}/{label}: {initial_counts[category][label]} files.")
        else:
            print(f"Check failed for {category}")
