import pandas as pd
import numpy as np

file_path = 'tsv_5p_rawdata_sum.xlsx'
df = pd.read_excel(file_path)

# --- Same preprocessing as before ---
blocks = []
current_block = []
block_id = 0
clean_data = []
current_experiment_rows = []
records = df.to_dict('records')

for i, row in enumerate(records):
    if str(row['up_width']).strip() == 'up_width':
        if current_experiment_rows:
            for r in current_experiment_rows:
                r['ExperimentID'] = block_id
            clean_data.extend(current_experiment_rows)
            block_id += 1
            current_experiment_rows = []
        continue
    current_experiment_rows.append(row)

if current_experiment_rows:
    for r in current_experiment_rows:
        r['ExperimentID'] = block_id
    clean_data.extend(current_experiment_rows)

df_clean = pd.DataFrame(clean_data)
cols_to_numeric = ['up_width', 'bottom_width', 'depth', 'scallps_length', 'scallps_height', 'angle']
for col in cols_to_numeric:
    df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')

# --- Analysis ---
cutoff = 45.0 # Based on previous finding (gap between 42.8 and 51.12)

df_clean['Type'] = df_clean['up_width'].apply(lambda x: 'Large' if x > cutoff else 'Small')

print("Counts per block:")
counts = df_clean.groupby(['ExperimentID', 'Type']).size().unstack(fill_value=0)
print(counts)

print("\nDoes every block have at least one Large and one Small?")
has_both = (counts['Large'] > 0) & (counts['Small'] > 0)
print(f"Blocks with both types: {has_both.sum()} out of {len(counts)}")

if not has_both.all():
    print("\nBlocks missing types:")
    print(counts[~has_both])








