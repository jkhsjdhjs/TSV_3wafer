import pandas as pd

file_path = 'tsv_5p_rawdata_sum.xlsx'
try:
    df = pd.read_excel(file_path)
    print("Columns:", df.columns.tolist())
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nInfo:")
    print(df.info())
    print("\nValue Counts for Wafer/Experiment (if identifiable):")
    # Look for wafer identifier columns
    possible_ids = [col for col in df.columns if 'wafer' in col.lower() or 'id' in col.lower() or 'exp' in col.lower()]
    if possible_ids:
        for col in possible_ids:
            print(f"\n{col} counts:")
            print(df[col].value_counts())
    else:
        print("\nNo obvious Wafer/Experiment ID column found.")
        
except Exception as e:
    print(f"Error reading file: {e}")








