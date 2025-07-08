import pandas as pd
import numpy as np

# Create a date range from 2015-01-01 to 2025-06-30
date_range = pd.date_range(start='2015-01-01', end='2025-06-30', freq='D')

# Create a DataFrame with the date range
df = pd.DataFrame(date_range, columns=['Date'])

# Set a base interest rate of around 4% yearly
base_interest_rate = 4.04

# Generate a random interest rate that generally hovers around the base rate
df['Customer Savings Interest Rate'] = base_interest_rate + np.random.normal(0, 0.5, len(date_range))

# Ensure the interest rate doesn't go negative
df['Customer Savings Interest Rate'] = df['Customer Savings Interest Rate'].clip(lower=0)

# Create a linear trend for daily savings and spending
n = len(date_range)
trend = np.linspace(0, 1, n)

# Set initial and final values for daily savings and spending
initial_daily_savings = 0
final_daily_savings = 18000
initial_daily_spending = 5000
final_daily_spending = 50000

# Calculate daily savings and spending with a linear increase
df['Daily Savings'] = initial_daily_savings + trend * final_daily_savings
df['Daily Spending'] = initial_daily_spending + trend * (final_daily_spending - initial_daily_spending)

# Save the DataFrame to a CSV file in the specified path
file_path = r"C:\Users\manik\Downloads\synthetic_customer_data.csv"
df.to_csv(file_path, index=False)

file_path
