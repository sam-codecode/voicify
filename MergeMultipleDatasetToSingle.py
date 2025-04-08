import pandas as pd
import glob

# Assuming all your gesture CSVs are in 'datasets' folder
csv_files = glob.glob('datasets/*.csv')

df_list = [pd.read_csv(file) for file in csv_files]
full_data = pd.concat(df_list, ignore_index=True)
full_data.to_csv('full_landmark_dataset.csv', index=False)

print("✅ All datasets combined into full_landmark_dataset.csv")
