# data/check_data.py
import pandas as pd


def main():
    df = pd.read_csv("data/amazon_reviews.csv")
    print(df.shape)
    print(df.isnull().sum())
    print(df['rating'].describe())


if __name__ == "__main__":
    main()