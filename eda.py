# eda.py
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import config

def plot_target_distribution(df):
    plt.figure(figsize=(8,5))
    sns.histplot(df[config.TARGET], bins=50, kde=True)
    plt.title("Distribution of HotelValue")
    plt.show()

def missing_value_summary(df):
    missing = df.isnull().sum()
    print("Missing Values (top 20):")
    print(missing[missing > 0].sort_values(ascending=False).head(20))

if __name__ == "__main__":
    df = pd.read_csv(config.DATA_TRAIN_PATH)
    plot_target_distribution(df)
    missing_value_summary(df)
