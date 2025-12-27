import pandas as pd
from pathlib import Path
from datetime import datetime

class TransactionIngestion:
    """Loads and validates raw transaction CSV files"""
    
    def __init__(self, filepath):
        self.filepath = Path(filepath)
        self.df = None
        
    def load(self):
        """Load CSV and perform basic validation"""
        print(f"📂 Loading transactions from {self.filepath}")
        
        # Load CSV
        self.df = pd.read_csv(self.filepath)
        
        # Validation checks
        required_columns = ['Date', 'Description', 'Amount']
        missing = set(required_columns) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Convert date column to datetime
        self.df['Date'] = pd.to_datetime(self.df['Date'])
        
        # Convert amount to float (in case it's string)
        self.df['Amount'] = pd.to_numeric(self.df['Amount'], errors='coerce')
        
        # Remove rows with missing critical data
        initial_count = len(self.df)
        self.df = self.df.dropna(subset=['Date', 'Amount'])
        dropped = initial_count - len(self.df)
        
        if dropped > 0:
            print(f"⚠️  Dropped {dropped} rows with missing data")
        
        print(f"✅ Loaded {len(self.df)} valid transactions")
        return self.df
    
    def basic_stats(self):
        """Print exploratory data analysis"""
        if self.df is None:
            raise ValueError("Must call .load() first")
        
        print("\n" + "="*50)
        print("BASIC STATISTICS")
        print("="*50)
        
        # Date range
        print(f"📅 Date Range: {self.df['Date'].min().date()} to {self.df['Date'].max().date()}")
        
        # Spending vs Income
        spending = self.df[self.df['Amount'] < 0]['Amount'].sum()
        income = self.df[self.df['Amount'] > 0]['Amount'].sum()
        print(f"💸 Total Spending: ${abs(spending):,.2f}")
        print(f"💰 Total Income: ${income:,.2f}")
        print(f"📊 Net: ${income + spending:,.2f}")
        
        # Transaction counts
        print(f"🔢 Total Transactions: {len(self.df)}")
        
        # Top spending descriptions
        print("\n🏆 Top 5 Most Frequent Merchants:")
        top_merchants = self.df['Description'].value_counts().head(5)
        for merchant, count in top_merchants.items():
            print(f"   {merchant}: {count} transactions")
        
        return self.df.describe()

# Test the ingestion
if __name__ == "__main__":
    ingestion = TransactionIngestion("data/raw/transactions.csv")
    df = ingestion.load()
    stats = ingestion.basic_stats()
    
    print("\n📋 First 10 transactions:")
    print(df.head(10))