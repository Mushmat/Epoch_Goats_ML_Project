import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

SEED = 42
np.random.seed(SEED)
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "best_kaggle_submission.csv"

print("="*80)
print("BEST PIPELINE FOR HOTEL VALUE SUBMISSION")
print("="*80)

# 1. LOAD DATA
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 2. FEATURE ENGINEERING (Keep strong features)
def advanced_feature_engineering(df):
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

train_df = advanced_feature_engineering(train_df)
test_df = advanced_feature_engineering(test_df)

# 3. OUTLIER REMOVAL -- use 0.03/0.97 quantiles
def remove_outliers_advanced(df, target_col='HotelValue'):
    before = len(df)
    Q1 = df[target_col].quantile(0.01)
    Q3 = df[target_col].quantile(0.99)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    after = len(df_clean)
    print(f"  Removed {before-after} outliers ({100*(before-after)/before:.2f}%)")
    return df_clean

train_df = remove_outliers_advanced(train_df)

# 4. PREPARE DATA
print("\n[4/7] Preparing data...")
X_train = train_df.drop(['Id', 'HotelValue'], axis=1)
y_train = train_df['HotelValue']
X_test = test_df.drop(['Id'], axis=1)
test_ids = test_df['Id']
y_train_log = np.log1p(y_train)
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=False)
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_test_imputed = imputer.transform(X_test_encoded)
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

# 5. TRAIN MODELS (validation split is for diagnostics; use all data for submission)
print("\n[5/7] Training base models and reporting validation RMSE (for diagnostics)...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_log, test_size=0.15, random_state=SEED)
models = {}

ridge = Ridge(alpha=15, random_state=SEED)
ridge.fit(X_tr, y_tr)
models['Ridge'] = {'model': ridge, 'rmse': np.sqrt(mean_squared_error(y_val, ridge.predict(X_val))), 'weight': 0.15}

lasso = Lasso(alpha=0.0005, max_iter=10000, random_state=SEED)
lasso.fit(X_tr, y_tr)
models['Lasso'] = {'model': lasso, 'rmse': np.sqrt(mean_squared_error(y_val, lasso.predict(X_val))), 'weight': 0.10}

elastic = ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000, random_state=SEED)
elastic.fit(X_tr, y_tr)
models['ElasticNet'] = {'model': elastic, 'rmse': np.sqrt(mean_squared_error(y_val, elastic.predict(X_val))), 'weight': 0.10}

bayesian = BayesianRidge(compute_score=True)
bayesian.fit(X_tr, y_tr)
models['BayesianRidge'] = {'model': bayesian, 'rmse': np.sqrt(mean_squared_error(y_val, bayesian.predict(X_val))), 'weight': 0.15}

xgb_model = xgb.XGBRegressor(
    n_estimators=1000, max_depth=3, learning_rate=0.03, subsample=0.8,
    colsample_bytree=0.8, min_child_weight=3, gamma=0.1,
    reg_alpha=0.1, reg_lambda=1, random_state=SEED, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
models['XGBoost'] = {'model': xgb_model, 'rmse': np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val))), 'weight': 0.30}

gb_model = GradientBoostingRegressor(
    n_estimators=500, max_depth=4, learning_rate=0.03, subsample=0.8,
    min_samples_split=5, min_samples_leaf=2, random_state=SEED
)
gb_model.fit(X_tr, y_tr)
models['GradientBoosting'] = {'model': gb_model, 'rmse': np.sqrt(mean_squared_error(y_val, gb_model.predict(X_val))), 'weight': 0.20}

print("\nModel Performance:")
for name, info in sorted(models.items(), key=lambda x: x[1]['rmse']):
    print(f"{name:20s} RMSE: {info['rmse']:.6f}  Weight: {info['weight']:.2f}")

# 6. ENSEMBLE - retrain models on ALL processed training data, then blend
print("\n[6/7] Creating weighted ensemble with all training data...")
weights = {
    'Ridge': 0.13, 'Lasso': 0.12, 'ElasticNet': 0.12,
    'BayesianRidge': 0.13, 'XGBoost': 0.19, 'GradientBoosting': 0.31
}
for info in models.values():
    info['model'].fit(X_train_scaled, y_train_log)
ensemble_preds_log = np.zeros(len(X_test_scaled))
for name, info in models.items():
    ensemble_preds_log += info['model'].predict(X_test_scaled) * weights.get(name, info['weight'])
final_preds = np.expm1(ensemble_preds_log)
final_preds = np.clip(final_preds, y_train.min(), y_train.max() * 0.97)

# 7. WRITE SUBMISSION
print("\n[7/7] Generating submission file...")
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSUBMISSION SAVED: {SUBMISSION_PATH}")
print("="*80)
print("\nBest Ensemble Details:")
print(f"  - 6 models, weighted linear blend")
print(f"  - Weights: GBM 27%, XGB 21%, Ridge 16%, Bayesian 15%, Lasso 11.5%, ElasticNet 9.5%")
print(f"  - Outlier removal: 3–97% quantile clipped")
print("="*80)
