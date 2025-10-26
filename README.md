# Hotel Property Value Prediction

## Project Overview

This project builds regression models to predict hotel property values using tabular data with mixed feature types. The goal was to minimize root mean squared error (RMSE) on a Kaggle-style test set.

---

## Dataset

- `train.csv`: Property features + target column `HotelValue`.
- `test.csv`: Property features only, predictions submitted for leaderboard evaluation.
- Target distribution was highly right-skewed, so a logarithmic transform was applied.

---

## Feature Engineering

- Created aggregate features like total baths (`TotalBaths`), total floor area (`TotalSF`), property age, years since remodel, and a facility score aggregating pool, parking, basement, veranda presence.
- Combined `OverallQuality` and `OverallCondition` into `QualityCondition`.
- Missing values imputed with median for numerical and "Missing" for categoricals.
- Categorical variables one-hot encoded and aligned between train and test.
- Robust scaling applied to numerical features.

---

## Models Tried and Scores Achieved

| Model               | Key Parameters                         | Public RMSE |
|---------------------|--------------------------------------|-------------|
| Linear Regression    | Default                             | **18,183**  |
| Ridge               | alpha=15                           | 22,686      |
| Lasso               | alpha=0.0005, max_iter=10000       | 19,841      |
| ElasticNet          | alpha=0.0005, l1_ratio=0.5          | 19,431      |
| Bayesian Ridge      | Default                            | 22,500      |
| Polynomial Regression| degree=2                           | 249,315     |
| K Nearest Neighbors | n_neighbors=7                      | 51,307      |
| Random Forest       | 300 trees, max_depth=18            | 32,381      |
| Decision Tree       | max_depth=14                       | 40,972      |
| AdaBoost            | Decision tree base, 250 estimators  | 29,094      |
| XGBoost             | 1000 estimators, max_depth=3       | 29,207      |
| Gradient Boosting   | 500 estimators, max_depth=4         | 26,201      |

---

## Experimental Observations

- **Linear Regression was best**, indicating the problem is largely linear in nature after good feature engineering.
- Regularized linear models (Ridge, Lasso, ElasticNet) performed comparably but did not surpass Linear Regression.
- Ensemble methods and more complex models (XGBoost, GBM, AdaBoost, Random Forest) showed lower performance, likely due to noise and weak nonlinearities.
- Polynomial regression severely overfitted.
- KNN and decision trees performed poorly, likely due to high dimensionality and sparse categorical features.
- Outlier removal experiments slightly degraded the Linear model performance, suggesting outliers carried meaningful signal.
- Blending or stacking failed to outperform Linear Regression.

---

## Final Submission

- Generated from the Linear Regression model trained on all training data, predicting on test data after preprocessing and log-transform inversions.
- Submission file saved as `linear_submission.csv`.

---

## How to Run

- Ensure Python 3.11+ is installed.  
- Install dependencies:

`pip install numpy pandas scikit-learn xgboost`

- Place `train.csv` and `test.csv` in the working directory with the scripts.
- Run the desired model script, e.g.:

`python model_linear.py`


- Output will be saved in the script folder as `[model]_submission.csv`.

---

## Future Work

- Adding more feature combinations and domain knowledge.
- Trying robust or quantile regression.
- Hyperparameter tuning with automated tools.
- External data enrichment (location, market indexes).
- Advanced ensembling or neural network models.

---

This README provides a complete guide to your project code, experiments, and results and will help evaluators and collaborators understand your work clearly.



