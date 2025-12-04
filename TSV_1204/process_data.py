import pandas as pd
import matplotlib.pyplot as plt

file_path = 'tsv_5p_rawdata_sum.xlsx'
df = pd.read_excel(file_path)

# Identify blocks
blocks = []
current_block = []
block_id = 0

# The first row is already data (header is at -1 index effectively)
# But we need to handle the rows that are headers embedded in the file.

# Convert to list of dicts for easier processing
records = df.to_dict('records')

# We need to catch the start of the file as the first block
# And split whenever we see 'up_width' in 'up_width' column.

clean_data = []

current_experiment_rows = []

for i, row in enumerate(records):
    # Check if this is a header row
    if str(row['up_width']).strip() == 'up_width':
        # End of previous block
        if current_experiment_rows:
            for r in current_experiment_rows:
                r['ExperimentID'] = block_id
            clean_data.extend(current_experiment_rows)
            block_id += 1
            current_experiment_rows = []
        # This row is a header, skip it
        continue
    
    # Otherwise, it's data
    current_experiment_rows.append(row)

# Add the last block
if current_experiment_rows:
    for r in current_experiment_rows:
        r['ExperimentID'] = block_id
    clean_data.extend(current_experiment_rows)

df_clean = pd.DataFrame(clean_data)

# Convert numeric columns
cols_to_numeric = ['up_width', 'bottom_width', 'depth', 'scallps_length', 'scallps_height', 'angle']
for col in cols_to_numeric:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

print(f"Total clean rows: {len(df_clean)}")
print(f"Number of experiments (blocks): {df_clean['ExperimentID'].nunique()}")

# Analyze up_width to find 'Small' vs 'Large' cutoff
print("\nUp_width stats:")
print(df_clean['up_width'].describe())

# Let's look at the distribution of up_width
# We can print a histogram or just sorted values
print("\nSorted unique up_width values (sampled):")
unique_widths = sorted(df_clean['up_width'].dropna().unique())
print(unique_widths[:10])
print(unique_widths[-10:])

# Check if there's a clear gap
# calculate differences between sorted unique values
import numpy as np
diffs = np.diff(unique_widths)
max_gap_idx = np.argmax(diffs)
print(f"\nMax gap in up_width is {diffs[max_gap_idx]} between {unique_widths[max_gap_idx]} and {unique_widths[max_gap_idx+1]}")

# We will use this gap to separate Small and Large
cutoff = (unique_widths[max_gap_idx] + unique_widths[max_gap_idx+1]) / 2
print(f"Suggested Cutoff: {cutoff}")

# Check counts
small_count = len(df_clean[df_clean['up_width'] < cutoff])
large_count = len(df_clean[df_clean['up_width'] > cutoff])
print(f"Small samples: {small_count}")
print(f"Large samples: {large_count}")








