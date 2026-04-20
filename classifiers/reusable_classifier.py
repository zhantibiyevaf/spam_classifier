from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class ReusableClassifier:
    def __init__(self, model_type='logistic_regression', test_size=0.2, random_state=42):
        self.model_type = model_type
        self.test_size = test_size
        self.random_state = random_state
        self.model = self._build_model()

    def _build_model(self):
        if self.model_type == 'logistic_regression':
            return LogisticRegression(max_iter=1000, random_state=self.random_state)
        if self.model_type == 'random_forest':
            return RandomForestClassifier(random_state=self.random_state)
        raise ValueError(f"Unsupported model_type: {self.model_type}")

    def assess(self, features, labels):
        X_train, X_test, y_train, y_test = train_test_split(
            features,
            labels,
            test_size=self.test_size,
            random_state=self.random_state,
            stratify=labels,
        )
        self.model.fit(X_train, y_train)
        return self.model.score(X_test, y_test)
