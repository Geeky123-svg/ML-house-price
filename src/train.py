import pandas as pd
DATA_PATH="data/raw/train.csv"
df=pd.read_csv(DATA_PATH)
print(f"Number of rows = {df.shape[0]}")
print(f"Number of columns = {df.shape[1]}")
print(df.head())
print(df["SalePrice"].describe())
print(df["SalePrice"].head())
