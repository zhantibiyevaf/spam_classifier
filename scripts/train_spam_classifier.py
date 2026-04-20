import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from readers.spam_reader import load_spam_data
from features.spam_features import num_features
from classifiers.reusable_classifier import ReusableClassifier


def main():
    # Load the dataset, select numeric features, and train the classifier.
    data_path = PROJECT_ROOT / 'data' / 'spam_email_dataset.csv'

    df = load_spam_data(data_path)
    features, labels = num_features(df)

    for model_type in ["logistic_regression", "random_forest"]:
        classifier = ReusableClassifier(model_type=model_type)
        accuracy = classifier.assess(features, labels)
        
        print('Spam classifier trained succesfully!')
        print(f"{model_type} accuracy: {accuracy:.2f}")
   

if __name__ == "__main__":
    main()
