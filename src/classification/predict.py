import pandas as pd
from train import TransactionClassifier


def categorize_transactions(input_csv, output_csv):
    """Apply trained classifier to real transaction data"""

    print(f"📂 Loading transactions from {input_csv}")
    df = pd.read_csv(input_csv)

    # Load trained model
    classifier = TransactionClassifier.load_model()

    print(f"\n🤖 Categorizing {len(df)} transactions...")

    # Filter spending transactions
    spending_mask = df["transaction_type"] == "spending"
    spending_df = df[spending_mask].copy()

    # Predict
    predictions = classifier.predict(spending_df["Description"].tolist())

    # Attach predictions
    spending_df["predicted_category"] = [p["category"] for p in predictions]
    spending_df["confidence"] = [p["confidence"] for p in predictions]

    # Merge back into original dataframe
    df.loc[spending_mask, "predicted_category"] = spending_df["predicted_category"]
    df.loc[spending_mask, "confidence"] = spending_df["confidence"]

    # Save results
    df.to_csv(output_csv, index=False)
    print(f"✅ Categorized transactions saved to {output_csv}")

    # Category distribution
    print("\n📊 Category Distribution:")
    print(df["predicted_category"].value_counts(dropna=False))

    # Low-confidence predictions
    low_conf = df[(df["confidence"] < 0.6) & df["confidence"].notna()]

    if not low_conf.empty:
        print(f"\n⚠️  {len(low_conf)} low-confidence predictions (< 60%):")
        print(low_conf[["Description", "predicted_category", "confidence"]].head(10))

    return df


if __name__ == "__main__":
    categorize_transactions(
        input_csv="data/processed/transactions_clean.csv",
        output_csv="data/processed/transactions_categorized.csv"
    )
