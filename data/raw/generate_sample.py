import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Generate 3 months of fake transactions
np.random.seed(42)
start_date = datetime(2024, 10, 1)
dates = [start_date + timedelta(days=i) for i in range(90)]

# Simulate realistic transactions
descriptions = [
    "Walmart Supercenter", "Shell Gas Station", "Starbucks", 
    "Amazon.com", "Netflix Subscription", "Electric Bill",
    "Target Store", "McDonald's", "Uber Ride", "Gym Membership"
]

data = []
for date in dates:
    # Random number of transactions per day (0-3)
    n_transactions = np.random.choice([0, 1, 2, 3], p=[0.3, 0.4, 0.2, 0.1])
    for _ in range(n_transactions):
        desc = np.random.choice(descriptions)
        amount = -round(np.random.uniform(5, 150), 2)  # Negative = spending
        data.append([date.strftime('%Y-%m-%d'), desc, amount])

# Add monthly income
for month in [10, 11, 12]:
    data.append([f'2024-{month:02d}-01', 'Salary Deposit', 3000.00])

df = pd.DataFrame(data, columns=['Date', 'Description', 'Amount'])
df = df.sort_values('Date').reset_index(drop=True)
df.to_csv('data/raw/transactions.csv', index=False)
print(f"✅ Generated {len(df)} transactions")