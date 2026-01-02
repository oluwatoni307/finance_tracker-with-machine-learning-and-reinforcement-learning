import pandas as pd
import numpy as np
from sklearn.calibration import CalibratedClassifierCV, LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from pathlib import Path


class TransactionClassifier:
    """ML classifier for automatic transaction categorization"""

    def __init__(self):
        """Initialize vectorizer and model"""
        # TF-IDF: Converts text to numerical features
        self.vectorizer = TfidfVectorizer(
            analyzer="char", ngram_range=(3, 5), min_df=2, lowercase=True
        )

        # Naive Bayes: Probabilistic classifier
        self.model = CalibratedClassifierCV(
            LinearSVC(),
            method="sigmoid"
        )

        self.categories = None  # Will store category list
        self.is_trained = False

    def load_training_data(self, filepath="data/raw/labeled_transactions.csv"):
        """Load manually labeled examples"""
        print(f"📂 Loading training data from {filepath}")

        df = pd.read_csv(filepath)

        # Validate required columns exist
        if "Description" not in df.columns or "Category" not in df.columns:
            raise ValueError("CSV must have 'Description' and 'Category' columns")

        # Clean descriptions (lowercase, strip whitespace)
        df["Description"] = df["Description"].str.lower().str.strip()

        # Get unique categories
        self.categories = sorted(df["Category"].unique())

        print(f"✅ Loaded {len(df)} labeled examples")
        print(f"📊 Categories: {', '.join(self.categories)}")
        print(f"📈 Examples per category:")
        print(df["Category"].value_counts())

        return df

    def split_data(self, df, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        X = df["Description"]  # Features (text)
        y = df["Category"]  # Labels (categories)

        # Split: 80% training, 20% testing
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,  # 20% for testing
            random_state=random_state,  # Reproducible split
            stratify=y,  # Keep same category ratio in train/test
        )

        print(f"\n📊 Data Split:")
        print(f"   Training: {len(X_train)} examples")
        print(f"   Testing: {len(X_test)} examples")

        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        """Train the classifier"""
        print("\n🎓 Training classifier...")

        # Step 1: Convert text to TF-IDF features
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"   TF-IDF vocabulary size: {len(self.vectorizer.vocabulary_)}")

        # Step 2: Train Naive Bayes model
        self.model.fit(X_train_tfidf, y_train)

        self.is_trained = True
        print("✅ Training complete!")

        return self

    def evaluate(self, X_test, y_test):
        """Evaluate model on test set"""
        print("\n📊 Evaluating model on test set...")

        # Convert test text to TF-IDF features (using SAME vocabulary from training)
        X_test_tfidf = self.vectorizer.transform(X_test)

        # Make predictions
        y_pred = self.model.predict(X_test_tfidf)

        # Calculate accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"\n🎯 Overall Accuracy: {accuracy:.2%}")

        # Detailed metrics per category
        print("\n📈 Per-Category Performance:")
        report = classification_report(y_test, y_pred, zero_division=0)
        print(report)

        # Confusion matrix
        print("\n🔍 Confusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=self.categories)
        self._print_confusion_matrix(cm)

        return accuracy

    def _print_confusion_matrix(self, cm):
        """Pretty print confusion matrix"""
        # Header
        print("\n" + " " * 15 + "PREDICTED")
        print(" " * 10, end="")
        for cat in self.categories:
            print(f"{cat[:8]:>10}", end="")
        print("\n" + "ACTUAL")

        # Rows
        for i, cat in enumerate(self.categories):
            print(f"{cat[:10]:>10}", end="")
            for j in range(len(self.categories)):
                print(f"{cm[i][j]:>10}", end="")
            print()

    def predict(self, descriptions):
        """Predict categories for new transaction descriptions"""
        if not self.is_trained:
            raise ValueError("Model must be trained before making predictions")

        # Handle single string input
        if isinstance(descriptions, str):
            descriptions = [descriptions]

        # Clean descriptions
        descriptions_clean = [desc.lower().strip() for desc in descriptions]

        # Convert to TF-IDF features
        X_tfidf = self.vectorizer.transform(descriptions_clean)

        # Predict
        predictions = self.model.predict(X_tfidf)

        # Get prediction probabilities
        probabilities = self.model.predict_proba(X_tfidf)

        # Format results
        results = []
        for desc, pred, probs in zip(descriptions, predictions, probabilities):
            confidence = probs.max()
            results.append(
                {"description": desc, "category": pred, "confidence": confidence}
            )

        return results

    def predict_single(self, description):
        """Predict category for a single description"""
        result = self.predict([description])[0]
        return result["category"], result["confidence"]

    def save_model(self, filepath="data/models/transaction_classifier.pkl"):
        """Save trained model and vectorizer"""
        if not self.is_trained:
            raise ValueError("Cannot save untrained model")

        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            "vectorizer": self.vectorizer,
            "model": self.model,
            "categories": self.categories,
        }

        with open(filepath, "wb") as f:
            pickle.dump(model_data, f)

        print(f"\n💾 Model saved to: {filepath}")

    @classmethod
    def load_model(cls, filepath="data/models/transaction_classifier.pkl"):
        """Load a trained model"""
        print(f"📂 Loading model from {filepath}")

        with open(filepath, "rb") as f:
            model_data = pickle.load(f)

        # Create new classifier and restore trained components
        classifier = cls()
        classifier.vectorizer = model_data["vectorizer"]
        classifier.model = model_data["model"]
        classifier.categories = model_data["categories"]
        classifier.is_trained = True

        print("✅ Model loaded successfully")
        return classifier


# Complete training pipeline
if __name__ == "__main__":
    print("🚀 Day 3: ML Classification Training\n")

    # Initialize classifier
    classifier = TransactionClassifier()

    # Load training data
    df = classifier.load_training_data("data/raw/labeled_transactions_balanced.csv")

    # Split into train/test
    X_train, X_test, y_train, y_test = classifier.split_data(df)

    # Train model
    classifier.train(X_train, y_train)

    # Evaluate performance
    accuracy = classifier.evaluate(X_test, y_test)

    # Save model
    classifier.save_model()

    # Test on some examples
    print("\n" + "=" * 50)
    print("🧪 TESTING ON NEW EXAMPLES")
    print("=" * 50)

    test_descriptions = [
        "whole foods market",
        "uber ride",
        "netflix subscription",
        "shell gas station",
        "doctor appointment",
    ]

    for desc in test_descriptions:
        category, confidence = classifier.predict_single(desc)
        print(f"{desc:30} → {category:15} (confidence: {confidence:.2%})")

    print("\n✅ Day 3 Complete! Classifier trained and saved.")
