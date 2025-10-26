import os
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures, RobustScaler
from sklearn.impute import SimpleImputer

SEED = 42

folder = os.path.dirname(__file__)
TRAIN_PATH = os.path.join(folder, "train.csv")
TEST_PATH = os.path.join(folder, "test.csv")
OUT_PATH = os.path.join(folder, "poly_submission.csv")

def feature_engineering(df):
    df = df.copy()
    df['TotalBaths'] = df['FullBaths'] + 0.5*df['HalfBaths'] + df['BasementFullBaths'] + 0.5*df['BasementHalfBaths']
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = (df['YearSold'] - df['RenovationYear']).clip(lower=0)
    df['QualityCondition'] = df['OverallQuality'] * df['OverallCondition']
    df['FacilityScore'] = ((df['SwimmingPoolArea'] > 0).astype(int) + 
        (df['ParkingArea'] > 0).astype(int) + (df['BasementTotalSF'] > 0).astype(int) +
        (df['UpperFloorArea'] > 0).astype(int) + ((df['TerraceArea'] + df['OpenVerandaArea'] + df['EnclosedVerandaArea'] +
            df['SeasonalPorchArea'] + df['ScreenPorchArea']) > 0).astype(int))
    return df

train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

X = train_df.drop(['Id', 'HotelValue'], axis=1)
y = np.log1p(train_df['HotelValue'])
X_test = test_df.drop(['Id'], axis=1)
test_ids = test_df['Id']

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
X_encoded = pd.get_dummies(X, columns=categorical_features, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=False)
X_encoded, X_test_encoded = X_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X_encoded)
X_test_imputed = imputer.transform(X_test_encoded)
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)
X_test_poly = poly.transform(X_test_scaled)

model = LinearRegression()
model.fit(X_poly, y)
preds_log = model.predict(X_test_poly)
preds = np.expm1(preds_log)
pd.DataFrame({'Id': test_ids, 'HotelValue': preds}).to_csv(OUT_PATH, index=False)
print(f"Polynomial Regression (degree 2) submission written to {OUT_PATH}")
