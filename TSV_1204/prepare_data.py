import pandas as pd
import numpy as np

file_path = 'tsv_5p_rawdata_sum.xlsx'
df = pd.read_excel(file_path)

# --- Preprocessing to clean and identify blocks ---
clean_data = []
current_experiment_rows = []
block_id = 0
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

# --- Pairing Strategy ---
# Cutoff for Small vs Large
cutoff = 45.0
pairs = []

# Group by ExperimentID
grouped = df_clean.groupby('ExperimentID')

for name, group in grouped:
    large_tsv = group[group['up_width'] > cutoff]
    small_tsvs = group[group['up_width'] <= cutoff]
    
    if len(large_tsv) != 1:
        # Skip blocks with unexpected structure (though we verified all have 1)
        continue
        
    l_row = large_tsv.iloc[0]
    
    for idx, s_row in small_tsvs.iterrows():
        pair = {
            # Inputs
            'Small_up_width': s_row['up_width'],
            'Small_depth': s_row['depth'],
            'Small_scallps_length': s_row['scallps_length'],
            'Small_scallps_height': s_row['scallps_height'],
            'Small_angle': s_row['angle'],
            'Large_up_width': l_row['up_width'], # The only Large parameter known
            
            # Outputs (Targets) - Predicting Large TSV parameters
            'Large_depth': l_row['depth'],
            'Large_scallps_length': l_row['scallps_length'],
            'Large_scallps_height': l_row['scallps_height'],
            'Large_angle': l_row['angle']
        }
        pairs.append(pair)

df_pairs = pd.DataFrame(pairs)
print(f"Generated {len(df_pairs)} pairs.")
df_pairs.to_csv('tsv_pairs_dataset.csv', index=False)








