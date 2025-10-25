# model_linear_and_families.py
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import config
import numpy as np

def get_preprocessor(num_features, cat_features):
    return ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), num_features),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("ohe", OneHotEncoder(handle_unknown='ignore', sparse_output=False))]), cat_features)
    ])

def run_model(model, X_train, y_train, X_val, y_val, preprocessor, name, degree=None):
    if degree:
        pipe = Pipeline([
            ("pre", preprocessor),
            ("poly", PolynomialFeatures(degree=degree, include_bias=False)),
            ("model", model)
        ])
    else:
        pipe = Pipeline([("pre", preprocessor), ("model", model)])

    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"{name} RMSE: {rmse:.2f}")
    return rmse, pipe

if __name__ == "__main__":
    df = pd.read_csv(config.DATA_TRAIN_PATH)
    test_df = pd.read_csv(config.DATA_TEST_PATH)
    y = df[config.TARGET]
    X = df.drop(columns=[config.ID_COL, config.TARGET])
    num_features = X.select_dtypes(include=["float", "int"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = get_preprocessor(num_features, cat_features)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)

    scores = {}
    for Model, name in [(LinearRegression(), "Linear Regression"),
                        (Ridge(), "Ridge"), (Lasso(), "Lasso"), (ElasticNet(), "ElasticNet")]:
        rmse, _ = run_model(Model, X_train, y_train, X_val, y_val, preprocessor, name)
        scores[name] = rmse
    # Polynomial Regression
    for degree in [2]:
        rmse, _ = run_model(LinearRegression(), X_train, y_train, X_val, y_val, preprocessor, f"Poly Regression d={degree}", degree)
        scores[f"Poly Regression d={degree}"] = rmse

    print("All Linear/Polynomial Scores:", scores)
