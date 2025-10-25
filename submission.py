# submission.py
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
import config

def make_submission(model_path, test_path, submit_path="my_submission.csv"):
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)
    preds = model.predict(test_df.drop(columns=[config.ID_COL]))
    submission = pd.read_csv(config.SUBMISSION_SAMPLE)
    submission[config.TARGET] = preds
    submission.to_csv(submit_path, index=False)
    print("Submission saved:", submit_path)

if __name__ == "__main__":
    make_submission(config.BEST_MODEL_PATH, config.DATA_TEST_PATH)
