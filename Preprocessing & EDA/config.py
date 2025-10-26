# config.py
DATA_TRAIN_PATH = "train.csv"
DATA_TEST_PATH = "test.csv"
SEED = 42
TEST_SIZE = 0.2
SUBMISSION_SAMPLE = "sample_submission.csv"
BEST_MODEL_PATH = "best_model.pkl"
NUM_JOBS = -1  # all CPUs

# Main categorical/numerical split and target name
TARGET = "HotelValue"
ID_COL = "Id"
