import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from readers.spam_reader import read_spam_dataset

def main():
    df = read_spam_dataset()
    print(df.head())
    print(df.columns)
    print(df.info())

if __name__ == "__main__":
    main()
