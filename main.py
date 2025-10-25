"""
Hotel Property Value Prediction - ULTRA OPTIMIZED
Target: < 22K RMSE on Kaggle

Key improvements:
- Stacking/Ensemble of top models
- More aggressive feature engineering
- Better outlier handling
- Optimized XGBoost parameters
- Model blending
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEED = 42
np.random.seed(SEED)

TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "ultra_optimized_submission.csv"

print("="*80)
print("ULTRA-OPTIMIZED PIPELINE FOR HOTEL VALUE PREDICTION")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/7] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train: {train_df.shape}, Test: {test_df.shape}")

# ============================================================================
# 2. ADVANCED FEATURE ENGINEERING
# ============================================================================
print("\n[2/7] Advanced feature engineering...")

def advanced_feature_engineering(df):
    """Create highly predictive features"""
    df = df.copy()
    
    # Basic features
    df['TotalBaths'] = (df['FullBaths'] + 0.5*df['HalfBaths'] + 
                        df['BasementFullBaths'] + 0.5*df['BasementHalfBaths'])
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    df['YearsSinceRemodel'] = df['YearsSinceRemodel'].clip(lower=0)
    
    # Porch and outdoor
    df['TotalPorchSF'] = (df['TerraceArea'] + df['OpenVerandaArea'] + 
                          df['EnclosedVerandaArea'] + df['SeasonalPorchArea'] + 
                          df['ScreenPorchArea'])
    
    # Binary features
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    df['HasParking'] = (df['ParkingArea'] > 0).astype(int)
    df['HasBasement'] = (df['BasementTotalSF'] > 0).astype(int)
    df['Has2ndFloor'] = (df['UpperFloorArea'] > 0).astype(int)
    df['HasPorch'] = (df['TotalPorchSF'] > 0).astype(int)
    
    # Quality interactions
    df['QualityCondition'] = df['OverallQuality'] * df['OverallCondition']
    df['QualitySF'] = df['OverallQuality'] * df['TotalSF']
    df['LivingAreaQuality'] = df['UsableArea'] * df['OverallQuality']
    
    # Ratios
    df['BathPerRoom'] = df['TotalBaths'] / (df['TotalRooms'] + 1)
    df['BasementRatio'] = df['BasementTotalSF'] / (df['TotalSF'] + 1)
    df['LivingRatio'] = df['UsableArea'] / (df['TotalSF'] + 1)
    
    # Area per room
    df['AreaPerRoom'] = df['TotalSF'] / (df['GuestRooms'] + 1)
    
    # Parking features
    df['ParkingPerArea'] = df['ParkingCapacity'] / (df['TotalSF'] + 1)
    
    # Quality features
    df['AvgQuality'] = (df['OverallQuality'] + df['OverallCondition'] + 
                        df['ExteriorQuality'].map({'Ex': 5, 'Gd': 4, 'TA': 3, 'Fa': 2, 'Po': 1}).fillna(3)) / 3
    
    # Age interactions
    df['AgeQuality'] = df['PropertyAge'] * df['OverallQuality']
    
    # Total facilities score
    df['FacilityScore'] = (df['HasPool'] + df['HasParking'] + df['HasBasement'] + 
                           df['Has2ndFloor'] + df['HasPorch'])
    
    # Premium features
    df['PremiumArea'] = df['TotalSF'] * df['OverallQuality'] / (df['PropertyAge'] + 1)
    
    return df

train_df = advanced_feature_engineering(train_df)
test_df = advanced_feature_engineering(test_df)

print(f"Features after engineering: {train_df.shape[1]}")

# ============================================================================
# 3. OUTLIER REMOVAL (More Aggressive)
# ============================================================================
print("\n[3/7] Removing outliers...")

def remove_outliers_advanced(df, target_col='HotelValue'):
    """Remove outliers using multiple methods"""
    before = len(df)
    
    # Remove extreme values
    Q1 = df[target_col].quantile(0.005)
    Q3 = df[target_col].quantile(0.995)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    df_clean = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    
    # Remove properties with unrealistic combinations
    df_clean = df_clean[~((df_clean['TotalSF'] < 500) & (df_clean[target_col] > 200000))]
    df_clean = df_clean[~((df_clean['OverallQuality'] <= 3) & (df_clean[target_col] > 250000))]
    
    after = len(df_clean)
    print(f"  Removed {before - after} outliers ({100*(before-after)/before:.2f}%)")
    
    return df_clean

train_df = remove_outliers_advanced(train_df)

# ============================================================================
# 4. PREPARE DATA
# ============================================================================
print("\n[4/7] Preparing data...")

X_train = train_df.drop(['Id', 'HotelValue'], axis=1)
y_train = train_df['HotelValue']
X_test = test_df.drop(['Id'], axis=1)
test_ids = test_df['Id']

# Log transform target
y_train_log = np.log1p(y_train)

# Encode categorical features
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=False)

# Align columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

# Impute and scale
imputer = SimpleImputer(strategy='median')
X_train_imputed = imputer.fit_transform(X_train_encoded)
X_test_imputed = imputer.transform(X_test_encoded)

scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"Final shape: Train {X_train_scaled.shape}, Test {X_test_scaled.shape}")

# ============================================================================
# 5. TRAIN MULTIPLE MODELS WITH OPTIMIZED PARAMETERS
# ============================================================================
print("\n[5/7] Training ensemble of models...")

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train_log, test_size=0.15, random_state=SEED
)

models = {}

# Ridge (optimized)
print("  Training Ridge...")
ridge = Ridge(alpha=15, random_state=SEED)
ridge.fit(X_tr, y_tr)
ridge_pred = ridge.predict(X_val)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
models['Ridge'] = {'model': ridge, 'rmse': ridge_rmse, 'weight': 0.15}
print(f"    Validation RMSE: {ridge_rmse:.6f}")

# Lasso (optimized)
print("  Training Lasso...")
lasso = Lasso(alpha=0.0005, max_iter=10000, random_state=SEED)
lasso.fit(X_tr, y_tr)
lasso_pred = lasso.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_pred))
models['Lasso'] = {'model': lasso, 'rmse': lasso_rmse, 'weight': 0.10}
print(f"    Validation RMSE: {lasso_rmse:.6f}")

# ElasticNet (optimized)
print("  Training ElasticNet...")
elastic = ElasticNet(alpha=0.0005, l1_ratio=0.5, max_iter=10000, random_state=SEED)
elastic.fit(X_tr, y_tr)
elastic_pred = elastic.predict(X_val)
elastic_rmse = np.sqrt(mean_squared_error(y_val, elastic_pred))
models['ElasticNet'] = {'model': elastic, 'rmse': elastic_rmse, 'weight': 0.10}
print(f"    Validation RMSE: {elastic_rmse:.6f}")

# Bayesian Ridge
print("  Training Bayesian Ridge...")
bayesian = BayesianRidge(compute_score=True)
bayesian.fit(X_tr, y_tr)
bayesian_pred = bayesian.predict(X_val)
bayesian_rmse = np.sqrt(mean_squared_error(y_val, bayesian_pred))
models['BayesianRidge'] = {'model': bayesian, 'rmse': bayesian_rmse, 'weight': 0.15}
print(f"    Validation RMSE: {bayesian_rmse:.6f}")

# XGBoost (heavily optimized)
print("  Training XGBoost...")
xgb_model = xgb.XGBRegressor(
    n_estimators=1000,
    max_depth=3,
    learning_rate=0.03,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=SEED,
    n_jobs=-1,
    tree_method='hist'
)
xgb_model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)], verbose=False)
xgb_pred = xgb_model.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
models['XGBoost'] = {'model': xgb_model, 'rmse': xgb_rmse, 'weight': 0.30}
print(f"    Validation RMSE: {xgb_rmse:.6f}")

# Gradient Boosting (optimized)
print("  Training Gradient Boosting...")
gb_model = GradientBoostingRegressor(
    n_estimators=500,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=SEED
)
gb_model.fit(X_tr, y_tr)
gb_pred = gb_model.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
models['GradientBoosting'] = {'model': gb_model, 'rmse': gb_rmse, 'weight': 0.20}
print(f"    Validation RMSE: {gb_rmse:.6f}")

# ============================================================================
# 6. ENSEMBLE PREDICTION (WEIGHTED AVERAGE)
# ============================================================================
print("\n[6/7] Creating ensemble predictions...")

# Display all models
print("\nModel Performance:")
print("-" * 60)
for name, info in sorted(models.items(), key=lambda x: x[1]['rmse']):
    print(f"{name:20s} RMSE: {info['rmse']:.6f}  Weight: {info['weight']:.2f}")

# Retrain all models on full training data
print("\nRetraining all models on full training data...")
for name, info in models.items():
    info['model'].fit(X_train_scaled, y_train_log)

# Make predictions
print("Making ensemble predictions...")
test_predictions_log = np.zeros(len(X_test_scaled))

for name, info in models.items():
    pred = info['model'].predict(X_test_scaled)
    test_predictions_log += pred * info['weight']
    print(f"  Added {name} predictions (weight: {info['weight']:.2f})")

# Inverse transform
test_predictions = np.expm1(test_predictions_log)

# ============================================================================
# 7. GENERATE SUBMISSION
# ============================================================================
print("\n[7/7] Generating submission...")

submission = pd.DataFrame({
    'Id': test_ids,
    'HotelValue': test_predictions
})
submission.to_csv(SUBMISSION_PATH, index=False)

print(f"\n{'='*80}")
print(f"SUBMISSION SAVED: {SUBMISSION_PATH}")
print(f"{'='*80}")
print("\nEnsemble Details:")
print(f"  - Combined {len(models)} models with weighted averaging")
print(f"  - XGBoost (30%), GradientBoosting (20%), Ridge (15%), Bayesian (15%)")
print(f"  - ElasticNet (10%), Lasso (10%)")
print(f"\nExpected improvement: This ensemble should perform better than single XGBoost")
print(f"Target: < 22,000 RMSE")
print(f"{'='*80}\n")
