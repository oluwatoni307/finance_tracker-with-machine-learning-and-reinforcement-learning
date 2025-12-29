import pandas as pd
import re
from datetime import datetime

class TransactionCleaner:
    """Cleans and enriches transaction data for analysis"""
    
    def __init__(self, df):
        """
        Args:
            df: DataFrame from TransactionIngestion.load()
        """
        self.df = df.copy()  # Work on copy to preserve original
        self.stats = {
            'duplicates_removed': 0,
            'descriptions_normalized': 0,
            'features_added': 0
        }
        
        
    
    def get_stats(self):
        """Return cleaning statistics"""
        return self.stats
    
    def remove_duplicates(self):
        """Remove exact duplicate transactions (e.g., from double imports)"""
        initial_count = len(self.df)
        
        # Drop duplicates based on Date + Amount + Description combination
        self.df = self.df.drop_duplicates(
            subset=['Date', 'Amount', 'Description'],
            keep='first'  # Keep first occurrence, remove subsequent ones
        )
        
        self.stats['duplicates_removed'] = initial_count - len(self.df)
        
        if self.stats['duplicates_removed'] > 0:
            print(f"🗑️  Removed {self.stats['duplicates_removed']} duplicate transactions")
        
        return self  # Return self for method chaining
    def normalize_descriptions(self):
        """Clean up merchant names for consistency"""
        
        # Convert to lowercase
        self.df['Description'] = self.df['Description'].str.lower()
        
        # Remove leading/trailing whitespace
        self.df['Description'] = self.df['Description'].str.strip()
        
        # Remove store numbers: "walmart #1234" → "walmart"
        self.df['Description'] = self.df['Description'].str.replace(
            r'\s*#\d+', '', regex=True
        )
        
        # Remove "store" suffix: "target store" → "target"
        self.df['Description'] = self.df['Description'].str.replace(
            r'\s+store\s*$', '', regex=True
        )
        
        # Condense multiple spaces to single space
        self.df['Description'] = self.df['Description'].str.replace(
            r'\s+', ' ', regex=True
        )
        
        self.stats['descriptions_normalized'] = len(self.df)
        print(f"✨ Normalized {self.stats['descriptions_normalized']} merchant names")
        
        return 
    
    def add_time_features(self):
        """Extract useful time-based features from Date column"""
        
        # Ensure Date is datetime type (should already be from ingestion)
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Extract day of week (0=Monday, 6=Sunday)
        self.df['day_of_week'] = self.df['Date'].dt.dayofweek
        
        # Extract day name (for readability)
        self.df['day_name'] = self.df['Date'].dt.day_name()
        
        # Is it a weekend? (Saturday=5, Sunday=6)
        self.df['is_weekend'] = self.df['day_of_week'].isin([5, 6])
        
        # Extract month (1-12)
        self.df['month'] = self.df['Date'].dt.month
        
        # Extract month name
        self.df['month_name'] = self.df['Date'].dt.month_name()
        
        # Day of month (1-31) - useful for payday detection later
        self.df['day_of_month'] = self.df['Date'].dt.day
        
        self.stats['features_added'] = 5
        print(f"📅 Added {self.stats['features_added']} time-based features")
        
        return 
    
    
    def add_spending_categories(self):
        """Categorize transactions as income vs spending"""
        
        # Create spending/income flag
        self.df['transaction_type'] = self.df['Amount'].apply(
            lambda x: 'income' if x > 0 else 'spending'
        )
        
        # Absolute amount (for easier analysis)
        self.df['amount_abs'] = self.df['Amount'].abs()
        
        print(f"💰 Categorized transactions: "
              f"{(self.df['transaction_type'] == 'income').sum()} income, "
              f"{(self.df['transaction_type'] == 'spending').sum()} spending")
        
        return self

    def clean_all(self):
        """Run complete cleaning pipeline"""
        print("\n" + "="*50)
        print("🧹 STARTING DATA CLEANING PIPELINE")
        print("="*50 + "\n")
        
        self.remove_duplicates()
        self.normalize_descriptions()
        self.add_time_features()
        self.add_spending_categories()
        
        print("\n" + "="*50)
        print("✅ CLEANING COMPLETE")
        print("="*50)
        print(f"\nFinal dataset: {len(self.df)} transactions")
        print(f"Date range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        
        return self.df
    
    def save_cleaned(self, output_path='data/processed/transactions_clean.csv'):
        """Save cleaned DataFrame to CSV"""
        from pathlib import Path
        
        # Create processed directory if it doesn't exist
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Save to CSV
        self.df.to_csv(output_path, index=False)
        print(f"\n💾 Saved cleaned data to: {output_path}")
        
        return 
    
    
# Test the cleaning pipeline
if __name__ == "__main__":
    import sys
    sys.path.append('.')  # Add current directory to path
    
    from src.data_processing.ingestion import TransactionIngestion
    
    print("🚀 Day 2: Data Cleaning Pipeline\n")
    
    # Step 1: Load raw data
    print("Step 1: Loading raw transactions...")
    ingestion = TransactionIngestion("data/raw/transactions.csv")
    raw_df = ingestion.load()
    
    # Step 2: Clean the data
    print("\nStep 2: Cleaning data...")
    cleaner = TransactionCleaner(raw_df)
    cleaned_df = cleaner.clean_all()
    
    # Step 3: Show sample of cleaned data
    print("\n" + "="*50)
    print("📊 SAMPLE OF CLEANED DATA")
    print("="*50)
    print(cleaned_df[['Date', 'Description', 'Amount', 'day_name', 'is_weekend', 'transaction_type']].head(10))
    
    # Step 4: Save cleaned data
    cleaner.save_cleaned()
    
    # Step 5: Show statistics
    print("\n" + "="*50)
    print("📈 CLEANING STATISTICS")
    print("="*50)
    stats = cleaner.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n✅ Day 2 Complete! Cleaned data saved to data/processed/")