# data_prep.py
import pandas as pd
from sklearn.model_selection import train_test_split
import config

def load_data():
    df = pd.read_csv(config.DATA_TRAIN_PATH)
    test_df = pd.read_csv(config.DATA_TEST_PATH)
    return df, test_df

def basic_info(df):
    print(df.info())
    print(df.head())
    print(df.describe())
    print(df.isnull().sum().sort_values(ascending=False).head(20))

def train_val_split(df):
    X = df.drop([config.TARGET, config.ID_COL], axis=1)
    y = df[config.TARGET]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=config.TEST_SIZE, random_state=config.SEED
    )
    return X_train, X_val, y_train, y_val

if __name__ == "__main__":
    df, test_df = load_data()
    basic_info(df)
    X_train, X_val, y_train, y_val = train_val_split(df)
    print("Train/Val Split:", X_train.shape, X_val.shape)
