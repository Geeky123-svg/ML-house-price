import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.linear_model import Ridge
from sklearn.metrics import root_mean_squared_error
import joblib

from data_preprocessing import prepare_data
DATA_PATH="data/raw/train.csv"
df=pd.read_csv(DATA_PATH)
print(f"Number of rows = {df.shape[0]}")
print(f"Number of columns = {df.shape[1]}")

def train_model():
    X_train, X_val, y_train, y_val, preprocessor = prepare_data(
        "data/raw/train.csv"
    )

    with mlflow.start_run():

        model = Ridge(alpha=0.80, random_state=43)
        model.fit(X_train, y_train)

        preds = model.predict(X_val)
        rmse = root_mean_squared_error(y_val, preds)


        mlflow.log_param("model_type", "Ridge")
        mlflow.log_param("alpha", 0.80)
        mlflow.log_metric("rmse", rmse)

        joblib.dump(model, "model.joblib")
        joblib.dump(preprocessor, "preprocessor.joblib")

        mlflow.log_artifact("model.joblib")
        mlflow.log_artifact("preprocessor.joblib")
        mlflow.log_param("target_transform", "log1p")


        print(f"Validation RMSE (Ridge): {rmse:.2f}")


if __name__ == "__main__":
    train_model()