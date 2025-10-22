import glob
import pandas as pd
from Model import Model as m
from Model import get_device as gd
from Model import save_model as sm
from Model import load_model as lm

data = []

for file_path in glob.glob('./DATA/*.csv'):
    # Read the CSV file
    df = pd.read_csv(file_path)