import pandas as pd

file_path = 'tsv_5p_rawdata_sum.xlsx'
df = pd.read_excel(file_path)

print("Rows with non-null value in column 1:")
print(df[df[1].notna()])

print("\nChecking for non-numeric values in 'up_width':")
# force coerce to numeric, errors='coerce' turns non-numeric to NaN
df_numeric = df.copy()
for col in ['up_width', 'depth', 'scallps_length', 'scallps_height', 'angle']:
    df_numeric[col] = pd.to_numeric(df[col], errors='coerce')

print("\nRows where 'up_width' became NaN (indicating non-numeric original):")
print(df[df_numeric['up_width'].isna()])

# Check if there are sequential blocks
# Maybe the data is just a long list.
# If there are no identifiers, and we need to pair "small" and "large", 
# we need to know which ones belong to the same experiment.
# If the user says "mix as one database", maybe we treat them all as one pool 
# and pair every small with every large? That would generate ~ (N/2) * (N/2) pairs.
# Or maybe there are distinct groups.

print("\nTotal rows:", len(df))








