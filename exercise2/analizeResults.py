import pandas as pd
import matplotlib.pyplot as plt
import glob
import re

# List of CSV files
csv_files = glob.glob("results/different_sizes_*_gpu.csv")

dfs = []

# Load and process each CSV
for i, file in enumerate(csv_files, start=1):
    df = pd.read_csv(file)
    df['GPU'] = i  # Add GPU number


    # Extract resolution from Image column (assuming pattern like "image_4K" or "resized_pexels-christian-heitz_8K")
    def extract_resolution(name):
        match = re.search(r'(\d+)K', name)
        return match.group(1) + "K" if match else "Unknown"


    df['Resolution'] = df['Image'].apply(extract_resolution)

    df = df.drop(columns=['Image'])  # Remove Image column
    dfs.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)
# Set pandas to display all rows and all columns
pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)
print(merged_df)

# Save merged CSV
merged_df.to_csv("results/merged_different_sizes_results.csv", index=False)
print("Merged CSV saved as 'results/merged_different_sizes_results.csv'")

# Plot execution time vs block size for each resolution
for res in merged_df['Resolution'].unique():
    plt.figure(figsize=(8, 6))
    subset_res = merged_df[merged_df['Resolution'] == res]
    for gpu in subset_res['GPU'].unique():
        subset_gpu = subset_res[subset_res['GPU'] == gpu]
        plt.plot(subset_gpu['BlockSize'], subset_gpu['Time_ms'], marker='o', label=f'GPU {gpu}')

    plt.xlabel('Block Size')
    plt.ylabel('Execution Time [ms]')
    plt.title(f'Execution Time vs Block Size for Resolution {res}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/execution_time_{res}.png", dpi=300)
    plt.show()


# List of noise CSV files
csv_files = glob.glob("results/noise_*_gpu.csv")

dfs = []

# Load and process each CSV
for i, file in enumerate(csv_files, start=1):
    df = pd.read_csv(file)
    df['GPU'] = i  # Add GPU number


    # Extract noise level from Image column (assuming pattern like "image_4K_noise50")
    def extract_noise(name):
        match = re.search(r'noise(\d+)', name)
        return int(match.group(1)) if match else -1  # Return -1 if not found


    df['Noise'] = df['Image'].apply(extract_noise)

    df = df.drop(columns=['Image'])  # Remove Image column
    dfs.append(df)

# Concatenate all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)
print(merged_df)

# Save merged CSV
merged_df.to_csv("results/merged_noise_results.csv", index=False)
print("Merged CSV saved as 'results/merged_noise_results.csv'")

# Plot execution time vs block size for each noise level
for noise in sorted(merged_df['Noise'].unique()):
    plt.figure(figsize=(8, 6))
    subset_noise = merged_df[merged_df['Noise'] == noise]
    for gpu in subset_noise['GPU'].unique():
        subset_gpu = subset_noise[subset_noise['GPU'] == gpu]
        plt.plot(subset_gpu['BlockSize'], subset_gpu['Time_ms'], marker='o', label=f'GPU {gpu}')

    plt.xlabel('Block Size')
    plt.ylabel('Execution Time [ms]')
    plt.title(f'Execution Time vs Block Size for Noise Level {noise}')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"results/execution_time_noise{noise}.png", dpi=300)
    plt.show()
