"""
Hotel Property Value Prediction - Optimized Pipeline
This script includes:
- Log transformation of target
- Advanced feature engineering
- Outlier removal
- Hyperparameter tuning for multiple models
- Cross-validation
- Submission generation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
from sklearn.metrics import mean_squared_error, make_scorer
import joblib
import warnings
warnings.filterwarnings('ignore')

# Configuration
SEED = 42
np.random.seed(SEED)

# File paths
TRAIN_PATH = "train.csv"
TEST_PATH = "test.csv"
SUBMISSION_PATH = "optimized_submission.csv"
BEST_MODEL_PATH = "best_optimized_model.pkl"

print("="*80)
print("HOTEL PROPERTY VALUE PREDICTION - OPTIMIZED PIPELINE")
print("="*80)

# ============================================================================
# 1. LOAD DATA
# ============================================================================
print("\n[1/8] Loading data...")
train_df = pd.read_csv(TRAIN_PATH)
test_df = pd.read_csv(TEST_PATH)
print(f"Train shape: {train_df.shape}, Test shape: {test_df.shape}")

# ============================================================================
# 2. FEATURE ENGINEERING
# ============================================================================
print("\n[2/8] Feature engineering...")

def feature_engineering(df):
    """Create new features from existing ones"""
    df = df.copy()
    
    # Total bathrooms
    df['TotalBaths'] = (df['FullBaths'] + 0.5 * df['HalfBaths'] + 
                        df['BasementFullBaths'] + 0.5 * df['BasementHalfBaths'])
    
    # Total square footage
    df['TotalSF'] = df['BasementTotalSF'] + df['GroundFloorArea'] + df['UpperFloorArea']
    
    # Age of property
    df['PropertyAge'] = df['YearSold'] - df['ConstructionYear']
    df['YearsSinceRemodel'] = df['YearSold'] - df['RenovationYear']
    df['YearsSinceRemodel'] = df['YearsSinceRemodel'].apply(lambda x: 0 if x < 0 else x)
    
    # Total porch area
    df['TotalPorchSF'] = (df['TerraceArea'] + df['OpenVerandaArea'] + 
                          df['EnclosedVerandaArea'] + df['SeasonalPorchArea'] + 
                          df['ScreenPorchArea'])
    
    # Has pool
    df['HasPool'] = (df['SwimmingPoolArea'] > 0).astype(int)
    
    # Has garage/parking
    df['HasParking'] = (df['ParkingArea'] > 0).astype(int)
    
    # Overall quality * condition
    df['QualityCondition'] = df['OverallQuality'] * df['OverallCondition']
    
    # Rooms per area ratio
    df['RoomsPerArea'] = df['TotalRooms'] / (df['TotalSF'] + 1)
    
    # Basement ratio
    df['BasementRatio'] = df['BasementTotalSF'] / (df['TotalSF'] + 1)
    
    # Living area quality
    df['LivingAreaQuality'] = df['UsableArea'] * df['OverallQuality']
    
    return df

train_df = feature_engineering(train_df)
test_df = feature_engineering(test_df)

print(f"Features after engineering: {train_df.shape[1]}")

# ============================================================================
# 3. OUTLIER REMOVAL
# ============================================================================
print("\n[3/8] Removing outliers...")

def remove_outliers(df, target_col='HotelValue'):
    """Remove outliers based on IQR method"""
    Q1 = df[target_col].quantile(0.01)
    Q3 = df[target_col].quantile(0.99)
    IQR = Q3 - Q1
    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR
    
    before = len(df)
    df_clean = df[(df[target_col] >= lower) & (df[target_col] <= upper)]
    after = len(df_clean)
    print(f"  Removed {before - after} outliers ({100*(before-after)/before:.2f}%)")
    
    return df_clean

train_df = remove_outliers(train_df)

# ============================================================================
# 4. PREPARE TARGET (LOG TRANSFORMATION)
# ============================================================================
print("\n[4/8] Preparing target with log transformation...")

# Separate features and target
X_train = train_df.drop(['Id', 'HotelValue'], axis=1)
y_train = train_df['HotelValue']
X_test = test_df.drop(['Id'], axis=1)
test_ids = test_df['Id']

# Log transform target
y_train_log = np.log1p(y_train)
print(f"  Target transformed: mean={y_train.mean():.0f} -> log_mean={y_train_log.mean():.2f}")

# ============================================================================
# 5. PREPROCESSING PIPELINE
# ============================================================================
print("\n[5/8] Building preprocessing pipeline...")

# Identify feature types
numeric_features = X_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
categorical_features = X_train.select_dtypes(include=['object']).columns.tolist()

print(f"  Numeric features: {len(numeric_features)}")
print(f"  Categorical features: {len(categorical_features)}")

# Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', RobustScaler())  # RobustScaler is better for outliers
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', pd.get_dummies)  # Will use pd.get_dummies later for simplicity
])

# Combine preprocessing steps
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', SimpleImputer(strategy='constant', fill_value='Missing'), categorical_features)
    ])

# ============================================================================
# 6. HANDLE CATEGORICAL FEATURES
# ============================================================================
print("\n[6/8] Encoding categorical features...")

# One-hot encode categorical features
X_train_encoded = pd.get_dummies(X_train, columns=categorical_features, drop_first=False)
X_test_encoded = pd.get_dummies(X_test, columns=categorical_features, drop_first=False)

# Align train and test columns
X_train_encoded, X_test_encoded = X_train_encoded.align(X_test_encoded, join='left', axis=1, fill_value=0)

print(f"  Features after encoding: {X_train_encoded.shape[1]}")

# Handle any remaining missing values
imputer = SimpleImputer(strategy='median')
X_train_imputed = pd.DataFrame(
    imputer.fit_transform(X_train_encoded),
    columns=X_train_encoded.columns
)
X_test_imputed = pd.DataFrame(
    imputer.transform(X_test_encoded),
    columns=X_test_encoded.columns
)

# Scale features
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train_imputed)
X_test_scaled = scaler.transform(X_test_imputed)

print(f"  Final training shape: {X_train_scaled.shape}")

# ============================================================================
# 7. MODEL TRAINING WITH HYPERPARAMETER TUNING
# ============================================================================
print("\n[7/8] Training models with hyperparameter tuning...")

# Define RMSE scorer
rmse_scorer = make_scorer(lambda y, y_pred: np.sqrt(mean_squared_error(y, y_pred)), 
                          greater_is_better=False)

# Split for validation
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train_scaled, y_train_log, test_size=0.2, random_state=SEED
)

models_results = {}

# -------------------------
# Ridge Regression
# -------------------------
print("\n  [1] Ridge Regression with GridSearch...")
ridge_params = {'alpha': [0.1, 1, 5, 10, 50, 100, 500]}
ridge_grid = GridSearchCV(Ridge(random_state=SEED), ridge_params, 
                          cv=5, scoring=rmse_scorer, n_jobs=-1)
ridge_grid.fit(X_tr, y_tr)
ridge_pred = ridge_grid.predict(X_val)
ridge_rmse = np.sqrt(mean_squared_error(y_val, ridge_pred))
models_results['Ridge'] = {'model': ridge_grid, 'rmse': ridge_rmse, 
                           'best_params': ridge_grid.best_params_}
print(f"     Best params: {ridge_grid.best_params_}")
print(f"     Validation RMSE: {ridge_rmse:.4f}")

# -------------------------
# Lasso Regression
# -------------------------
print("\n  [2] Lasso Regression with GridSearch...")
lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1, 10]}
lasso_grid = GridSearchCV(Lasso(random_state=SEED, max_iter=10000), lasso_params,
                          cv=5, scoring=rmse_scorer, n_jobs=-1)
lasso_grid.fit(X_tr, y_tr)
lasso_pred = lasso_grid.predict(X_val)
lasso_rmse = np.sqrt(mean_squared_error(y_val, lasso_pred))
models_results['Lasso'] = {'model': lasso_grid, 'rmse': lasso_rmse,
                           'best_params': lasso_grid.best_params_}
print(f"     Best params: {lasso_grid.best_params_}")
print(f"     Validation RMSE: {lasso_rmse:.4f}")

# -------------------------
# ElasticNet
# -------------------------
print("\n  [3] ElasticNet with GridSearch...")
elastic_params = {'alpha': [0.001, 0.01, 0.1, 1], 'l1_ratio': [0.1, 0.5, 0.9]}
elastic_grid = GridSearchCV(ElasticNet(random_state=SEED, max_iter=10000), elastic_params,
                            cv=5, scoring=rmse_scorer, n_jobs=-1)
elastic_grid.fit(X_tr, y_tr)
elastic_pred = elastic_grid.predict(X_val)
elastic_rmse = np.sqrt(mean_squared_error(y_val, elastic_pred))
models_results['ElasticNet'] = {'model': elastic_grid, 'rmse': elastic_rmse,
                                'best_params': elastic_grid.best_params_}
print(f"     Best params: {elastic_grid.best_params_}")
print(f"     Validation RMSE: {elastic_rmse:.4f}")

# -------------------------
# Bayesian Ridge
# -------------------------
print("\n  [4] Bayesian Ridge...")
bayesian = BayesianRidge(compute_score=True)
bayesian.fit(X_tr, y_tr)
bayesian_pred = bayesian.predict(X_val)
bayesian_rmse = np.sqrt(mean_squared_error(y_val, bayesian_pred))
models_results['BayesianRidge'] = {'model': bayesian, 'rmse': bayesian_rmse}
print(f"     Validation RMSE: {bayesian_rmse:.4f}")

# -------------------------
# Random Forest
# -------------------------
print("\n  [5] Random Forest with GridSearch...")
rf_params = {
    'n_estimators': [200, 500],
    'max_depth': [10, 20, None],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
rf_grid = GridSearchCV(
    RandomForestRegressor(random_state=SEED, n_jobs=-1),
    rf_params, cv=3, scoring=rmse_scorer, n_jobs=-1, verbose=0
)
rf_grid.fit(X_tr, y_tr)
rf_pred = rf_grid.predict(X_val)
rf_rmse = np.sqrt(mean_squared_error(y_val, rf_pred))
models_results['RandomForest'] = {'model': rf_grid, 'rmse': rf_rmse,
                                  'best_params': rf_grid.best_params_}
print(f"     Best params: {rf_grid.best_params_}")
print(f"     Validation RMSE: {rf_rmse:.4f}")

# -------------------------
# XGBoost
# -------------------------
print("\n  [6] XGBoost with GridSearch...")
xgb_params = {
    'n_estimators': [200, 500, 1000],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'subsample': [0.8, 1.0],
    'colsample_bytree': [0.8, 1.0]
}
xgb_grid = GridSearchCV(
    xgb.XGBRegressor(random_state=SEED, n_jobs=-1, tree_method='hist'),
    xgb_params, cv=3, scoring=rmse_scorer, n_jobs=-1, verbose=0
)
xgb_grid.fit(X_tr, y_tr)
xgb_pred = xgb_grid.predict(X_val)
xgb_rmse = np.sqrt(mean_squared_error(y_val, xgb_pred))
models_results['XGBoost'] = {'model': xgb_grid, 'rmse': xgb_rmse,
                             'best_params': xgb_grid.best_params_}
print(f"     Best params: {xgb_grid.best_params_}")
print(f"     Validation RMSE: {xgb_rmse:.4f}")

# -------------------------
# Gradient Boosting
# -------------------------
print("\n  [7] Gradient Boosting...")
gb_params = {
    'n_estimators': [200, 500],
    'max_depth': [3, 5],
    'learning_rate': [0.01, 0.05, 0.1]
}
gb_grid = GridSearchCV(
    GradientBoostingRegressor(random_state=SEED),
    gb_params, cv=3, scoring=rmse_scorer, n_jobs=-1, verbose=0
)
gb_grid.fit(X_tr, y_tr)
gb_pred = gb_grid.predict(X_val)
gb_rmse = np.sqrt(mean_squared_error(y_val, gb_pred))
models_results['GradientBoosting'] = {'model': gb_grid, 'rmse': gb_rmse,
                                      'best_params': gb_grid.best_params_}
print(f"     Best params: {gb_grid.best_params_}")
print(f"     Validation RMSE: {gb_rmse:.4f}")

# ============================================================================
# 8. MODEL COMPARISON AND SELECTION
# ============================================================================
print("\n[8/8] Model comparison and final selection...")
print("\n" + "="*80)
print("MODEL COMPARISON (Validation RMSE on Log-Transformed Target)")
print("="*80)

# Sort models by RMSE
sorted_models = sorted(models_results.items(), key=lambda x: x[1]['rmse'])

for rank, (name, results) in enumerate(sorted_models, 1):
    print(f"{rank}. {name:20s} - RMSE: {results['rmse']:.6f}")
    if 'best_params' in results:
        print(f"   Best params: {results['best_params']}")

# Select best model
best_model_name = sorted_models[0][0]
best_model_obj = sorted_models[0][1]['model']
best_rmse = sorted_models[0][1]['rmse']

print(f"\n{'='*80}")
print(f"BEST MODEL: {best_model_name} with Validation RMSE: {best_rmse:.6f}")
print(f"{'='*80}")

# ============================================================================
# 9. RETRAIN ON FULL DATA AND PREDICT
# ============================================================================
print(f"\nRetraining {best_model_name} on full training data...")
best_model_obj.fit(X_train_scaled, y_train_log)

print("Making predictions on test set...")
test_predictions_log = best_model_obj.predict(X_test_scaled)

# Inverse transform predictions
test_predictions = np.expm1(test_predictions_log)

# ============================================================================
# 10. GENERATE SUBMISSION
# ============================================================================
print("\nGenerating submission file...")
submission = pd.DataFrame({
    'Id': test_ids,
    'HotelValue': test_predictions
})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission saved to: {SUBMISSION_PATH}")

# Save best model
joblib.dump({
    'model': best_model_obj,
    'scaler': scaler,
    'imputer': imputer,
    'feature_columns': X_train_encoded.columns.tolist()
}, BEST_MODEL_PATH)
print(f"Best model saved to: {BEST_MODEL_PATH}")

print("\n" + "="*80)
print("PIPELINE COMPLETE!")
print("="*80)
print(f"\nNext steps:")
print(f"1. Submit '{SUBMISSION_PATH}' to Kaggle")
print(f"2. Document all models and scores in your report")
print(f"3. The best model ({best_model_name}) is saved for future use")
print("\n" + "="*80)
