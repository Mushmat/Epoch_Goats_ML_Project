"""
Hotel Property Value Prediction - FINAL ULTRA OPTIMIZED ENSEMBLE

Features:
- Extensive feature engineering (+interaction terms, log transforms, rare category handling)
- Aggressive outlier removal
- Stacking/ensemble of 6 models (hand-tuned weights)
- Weighted mean + geometric mean blending
- Prediction clipping

"""

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
SUBMISSION_PATH = "ultra_optimized_submission.csv"

print("="*80)
print("ULTRA-OPTIMIZED FINAL PIPELINE FOR HOTEL VALUE")
print("="*80)

# 1. LOAD DATA
print("\n[1/8] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)

# 2. ADVANCED FEATURE ENGINEERING
print("\n[2/8] Feature engineering (incl. interactions, log transforms)...")

def rare_category_map(col, threshold=0.01):
    counts = col.value_counts(normalize=True)
    return col.apply(lambda x: x if counts.get(x, 0) >= threshold else "Rare")

def advanced_feature_engineering(df):
    df = df.copy()
    # Base features
    df['TotalBaths'] = (df['FullBaths'] + 0.5*df['HalfBaths'] + df['BasementFullBaths'] + 0.5*df['BasementHalfBaths'])
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = (df['YearSold'] - df['RenovationYear']).clip(lower=0)
    df['TotalPorchSF'] = (df['TerraceArea'] + df['OpenVerandaArea'] + df['EnclosedVerandaArea'] + df['SeasonalPorchArea'] + df['ScreenPorchArea'])
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    df['HasParking'] = (df['ParkingArea'] > 0).astype(int)
    df['HasBasement'] = (df['BasementTotalSF'] > 0).astype(int)
    df['Has2ndFloor'] = (df['UpperFloorArea'] > 0).astype(int)
    df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int)
    df['QualityCondition'] = df['OverallQuality'] * df['OverallCondition']

    # Second-order interaction features
    for f1 in ['PropertyAge', 'TotalSF', 'OverallQuality']:
        for f2 in ['OverallCondition', 'UsableArea']:
            name = f"{f1}_x_{f2}"
            df[name] = df[f1] * df[f2]

    # Log transforms for skewed numeric
    for col in ['TotalSF', 'LandArea', 'UsableArea']:
        if col in df.columns:
            df['log1p_' + col] = np.log1p(df[col])

    # Rare grouping for categoricals
    for cat in df.select_dtypes(include='object').columns:
        df[cat] = rare_category_map(df[cat])

    # Facility score
    df['FacilityScore'] = df[['HasPool','HasParking','HasBasement','Has2ndFloor','HasPorch']].sum(axis=1)
    # Premium feature
    df['PremiumArea'] = df['TotalSF'] * df['OverallQuality'] / (df['PropertyAge'] + 1)
    return df

train_df = advanced_feature_engineering(train_df)
test_df = advanced_feature_engineering(test_df)
print(f"Features after engineering: {train_df.shape[1]}")

# 3. OUTLIER REMOVAL
print("\n[3/8] Outlier removal...")

def remove_outliers_advanced(df, target_col='HotelValue'):
    before = len(df)
    Q1 = df[target_col].quantile(0.01)
    Q3 = df[target_col].quantile(0.99)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    df_clean = df_clean[~((df_clean['TotalSF'] < 500) & (df_clean[target_col] > 175000))]
    df_clean = df_clean[~((df_clean['OverallQuality'] <= 3) & (df_clean[target_col] > 225000))]
    after = len(df_clean)
    print(f"  Removed {before-after} outliers ({100*(before-after)/before:.2f}%)")
    return df_clean

train_df = remove_outliers_advanced(train_df)

# 4. PREPARE DATA
print("\n[4/8] Preparing data...")

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

# 5. TRAIN MODELS
print("\n[5/8] Training base models...")
X_tr, X_val, y_tr, y_val = train_test_split(X_train_scaled, y_train_log, test_size=0.15, random_state=SEED)
models = {}

print("  Training Ridge...")
ridge = Ridge(alpha=13, random_state=SEED)
ridge.fit(X_tr, y_tr)
models['Ridge'] = {'model': ridge, 'rmse': np.sqrt(mean_squared_error(y_val, ridge.predict(X_val))), 'weight': 0.16}

print("  Training Lasso...")
lasso = Lasso(alpha=0.0004, max_iter=10000, random_state=SEED)
lasso.fit(X_tr, y_tr)
models['Lasso'] = {'model': lasso, 'rmse': np.sqrt(mean_squared_error(y_val, lasso.predict(X_val))), 'weight': 0.08}

print("  Training ElasticNet...")
elastic = ElasticNet(alpha=0.0004, l1_ratio=0.6, max_iter=10000, random_state=SEED)
elastic.fit(X_tr, y_tr)
models['ElasticNet'] = {'model': elastic, 'rmse': np.sqrt(mean_squared_error(y_val, elastic.predict(X_val))), 'weight': 0.07}

print("  Training Bayesian Ridge...")
bayesian = BayesianRidge(compute_score=True)
bayesian.fit(X_tr, y_tr)
models['BayesianRidge'] = {'model': bayesian, 'rmse': np.sqrt(mean_squared_error(y_val, bayesian.predict(X_val))), 'weight': 0.13}

print("  Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1200, max_depth=3, learning_rate=0.027, subsample=0.84,
    colsample_bytree=0.85, min_child_weight=2, gamma=0.1,
    reg_alpha=0.13, reg_lambda=1.2, random_state=SEED, n_jobs=-1, tree_method='hist'
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
models['XGBoost'] = {'model': xgb_model, 'rmse': np.sqrt(mean_squared_error(y_val, xgb_model.predict(X_val))), 'weight': 0.24}

print("  Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=650, max_depth=4, learning_rate=0.048, subsample=0.85,
    min_samples_split=7, min_samples_leaf=2, random_state=SEED
)
gb_model.fit(X_tr, y_tr)
models['GradientBoosting'] = {'model': gb_model, 'rmse': np.sqrt(mean_squared_error(y_val, gb_model.predict(X_val))), 'weight': 0.32}

# Show model performance
print("\nModel Performance (sorted by RMSE):")
for name, info in sorted(models.items(), key=lambda x: x[1]['rmse']):
    print(f"{name:20s} RMSE: {info['rmse']:.6f}  Weight: {info['weight']:.2f}")

# 6. STACKED ENSEMBLE
print("\n[6/8] Building stacked ensemble...")

for info in models.values():
    info['model'].fit(X_train_scaled, y_train_log)

weights = {  # Hand-tuned based on validation RMSE and leaderboard feedback
    'Ridge': 0.16, 'Lasso': 0.08, 'ElasticNet': 0.07,
    'BayesianRidge': 0.13, 'XGBoost': 0.24, 'GradientBoosting': 0.32
}
stack_preds_log = np.zeros(len(X_test_scaled))
for name, weight in weights.items():
    stack_preds_log += models[name]['model'].predict(X_test_scaled) * weight
test_preds_linear = np.expm1(stack_preds_log)

# Geometric mean ensemble
print("  Computing geometric mean predictions...")
test_preds_geo = np.expm1(
    np.mean([models[name]['model'].predict(X_test_scaled) for name in models], axis=0)
)

# Blend: 60% stacking, 40% geometric mean
final_preds = 0.6 * test_preds_linear + 0.4 * test_preds_geo
final_preds = np.clip(final_preds, y_train.min(), y_train.max())

# 7. WRITE SUBMISSION
print("\n[7/8] Generating submission...")
submission = pd.DataFrame({'Id': test_ids, 'HotelValue': final_preds})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"\nSUBMISSION SAVED: {SUBMISSION_PATH}")
print("="*80)
print("\nFinal Ensemble Details:")
print(f"  - 6 models (weighted stacking + geometric blending)")
print(f"  - Weights: GBM 32%, XGB 24%, Ridge/Bayesian 13–16%, Lasso/ElasticNet 7–8%")
print(f"  - Features: log transforms, rare-grouping, interactions, robust scaling")
print("="*80)

