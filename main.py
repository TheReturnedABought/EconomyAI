import glob
import pandas as pd

for file_path in glob.glob('./DATA/*.csv'):
    df = file_path, pd.read_csv(file_path)
