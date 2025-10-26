# model_tree_ensemble.py
import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
import xgboost as xgb
from model_linear_and_families import get_preprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config
import numpy as np
from sklearn.pipeline import Pipeline


if __name__ == "__main__":
    df = pd.read_csv(config.DATA_TRAIN_PATH)
    y = df[config.TARGET]
    X = df.drop(columns=[config.ID_COL, config.TARGET])
    num_features = X.select_dtypes(include=["float", "int"]).columns.tolist()
    cat_features = X.select_dtypes(include=["object"]).columns.tolist()
    preprocessor = get_preprocessor(num_features, cat_features)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=config.TEST_SIZE, random_state=config.SEED)

    models = [
        (DecisionTreeRegressor(random_state=config.SEED), "Decision Tree"),
        (RandomForestRegressor(n_estimators=100, n_jobs=config.NUM_JOBS, random_state=config.SEED), "Random Forest"),
        (AdaBoostRegressor(n_estimators=100, random_state=config.SEED), "AdaBoost"),
        (xgb.XGBRegressor(n_estimators=100, n_jobs=config.NUM_JOBS, random_state=config.SEED, tree_method='hist'), "XGBoost")
    ]
    scores = {}
    for model, name in models:
        pipe = Pipeline([("pre", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        val_pred = pipe.predict(X_val)
        rmse = np.sqrt(mean_squared_error(y_val, val_pred))
        print(f"{name} RMSE:", rmse)
        scores[name] = rmse
    print("Tree/Ensemble Scores:", scores)
