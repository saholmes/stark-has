import pandas as pd
import glob

# Load all CSVs in a folder
csv_files = glob.glob('./*.csv')  # Or download from GitHub
for file in csv_files:
    df = pd.read_csv(file)
    # Quote the schedule column (assume it's column 'schedule')
    df['schedule'] = df['schedule'].apply(lambda x: f'"{x}"' if isinstance(x, str) and '[' in str(x) else x)
    # Save back
    df.to_csv(file, index=False)
    print(f"Quoted {file}")