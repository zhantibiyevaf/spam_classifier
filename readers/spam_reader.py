import pandas as pd

def load_spam_data(path: str) -> pd.DataFrame:
    """
    Load the spam email dataset from a CSV file.

    Args:
        path (str): The file path to the CSV dataset.

    Returns:
        pd.DataFrame: A DataFrame containing the loaded dataset.
    """
    df = pd.read_csv(path)
    return df