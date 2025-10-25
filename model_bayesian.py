# model_bayesian.py
import pandas as pd
from sklearn.linear_model import BayesianRidge
from model_linear_and_families import get_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config
import numpy as np

if __name__ == "__main__":
    df = pd.read_csv(config.DATA_TRAIN_PATH)
    y = df[config.TARGET]
    X = df.drop(columns=[config.ID_COL, config.TARGET])
    num_features = X.select_dtypes(include=["float", "int"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = get_preprocessor(num_features, cat_features)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)

    model = BayesianRidge()
    pipe = Pipeline([("pre", preprocessor), ("model", model)])
    pipe.fit(X_train, y_train)
    val_pred = pipe.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, val_pred))
    print(f"Bayesian Ridge RMSE: {rmse}")
