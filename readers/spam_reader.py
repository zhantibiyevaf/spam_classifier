from pathlib import Path

import pandas as pd


DATASET_PATH = Path(__file__).resolve().parent.parent / "data" / "spam_email_dataset.csv"


def read_spam_dataset(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Load the spam dataset into a DataFrame."""
    return pd.read_csv(dataset_path)


def load_spam_data(dataset_path: Path = DATASET_PATH) -> pd.DataFrame:
    """Backward-compatible wrapper used by the training script."""
    return read_spam_dataset(dataset_path)


def main() -> None:
    df = read_spam_dataset()
    print(df.head())
    print(df.columns.tolist())
    print(df.info())


if __name__ == "__main__":
    main()
