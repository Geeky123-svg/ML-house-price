import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def load_data(path):
    return pd.read_csv(path)

def pre_processor(X):
    num_features=X.select_dtypes(include=["int64","float64"]).columns
    obj_features=X.select_dtypes(include=["object"]).columns
    num_pipeline=Pipeline(steps=[
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="constant", fill_value="None")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])
    preprocessor=ColumnTransformer(
        transformers=[
            ("num",num_pipeline,num_features),
            ("obj",cat_pipeline,obj_features)
        ]
    )
    return preprocessor

def prepare_data(data_path):
    df = load_data(data_path)

    x = df.drop("SalePrice", axis=1)
    y = df["SalePrice"]

    X_train, X_val, y_train, y_val = train_test_split(
        x, y, test_size=0.2, random_state=42
    )

    preprocessor = pre_processor(X_train)

    X_train_transformed = preprocessor.fit_transform(X_train)
    X_val_transformed = preprocessor.transform(X_val)

    return X_train_transformed, X_val_transformed, y_train, y_val, preprocessor