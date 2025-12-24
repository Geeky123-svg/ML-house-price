import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
import joblib

from data_preprocessing import prepare_data
DATA_PATH="data/raw/train.csv"
df=pd.read_csv(DATA_PATH)
print(f"Number of rows = {df.shape[0]}")
print(f"Number of columns = {df.shape[1]}")

def train_model():
    X_train, X_val, y_train, y_val, preprocessor = prepare_data("data/raw/train.csv")

    model = LinearRegression()
    model.fit(X_train, y_train)

    preds = model.predict(X_val)
    rmse = root_mean_squared_error(y_val, preds)

    print(f"Validation RMSE: {rmse:.2f}")

    joblib.dump(model, "model.joblib")
    joblib.dump(preprocessor, "preprocessor.joblib")


if __name__ == "__main__":
    train_model()