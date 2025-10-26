import pandas as pd
import joblib
import config

def make_submission(model_path, test_path, submit_path="my_submission.csv"):
    test_df = pd.read_csv(test_path)
    model = joblib.load(model_path)
    X_test = test_df.drop(columns=[config.ID_COL])
    preds = model.predict(X_test)
    submission = pd.DataFrame({
        config.ID_COL: test_df[config.ID_COL],
        config.TARGET: preds
    })
    submission.to_csv(submit_path, index=False)
    print("Submission saved:", submit_path)

if __name__ == "__main__":
    make_submission(config.BEST_MODEL_PATH, config.DATA_TEST_PATH)
