import pandas as pd
import numpy as np
from datetime import datetime

# Set random seed for reproducibility
np.random.seed(42)

# Define date range
start_date = datetime(2015, 5, 5)
end_date = datetime(2025, 4, 30)

# Generate date range with daily frequency
date_range = pd.date_range(start=start_date, end=end_date, freq='D')
num_days = len(date_range)

# Initialize DataFrame
data = pd.DataFrame(index=date_range, columns=[
    'Daily_Spending',
    'Savings_Investment_Value',
    'Total_Portfolio_Value',
    'Portfolio_Return_YTD',
    'Investment_Goal',
    'Risk_Tolerance'
])

# Base daily spending value with inflation
base_daily_spending = 50
inflation_rate = 0.02  # 2% yearly inflation
data['Daily_Spending'] = base_daily_spending * (1 + inflation_rate) ** (np.arange(num_days) / 365)
data['Daily_Spending'] = data['Daily_Spending'] + np.random.uniform(-10, 10, num_days)
data['Daily_Spending'] = data['Daily_Spending'].round(2)

# Simulate daily savings with a chance of zero savings
monthly_savings = 1000  # Assume a monthly savings goal
daily_savings_prob = 0.3  # Probability of saving on a given day
daily_savings_amount = monthly_savings * np.random.uniform(0.2, 0.3, num_days) / 30
data['Savings_Investment_Value'] = np.where(np.random.rand(num_days) < daily_savings_prob, daily_savings_amount, 0)
data['Savings_Investment_Value'] = data['Savings_Investment_Value'].round(2)

# Simulate Portfolio Return YTD
data['Portfolio_Return_YTD'] = np.random.uniform(-0.005, 0.01, num_days).round(4)

# Simulate a significant negative period from March 2020 to March 2021
negative_period = (data.index >= '2020-03-01') & (data.index <= '2021-03-01')
data.loc[negative_period, 'Portfolio_Return_YTD'] = np.random.uniform(-0.02, -0.005, negative_period.sum()).round(4)

# Calculate Total Portfolio Value based on Portfolio_Return_YTD and Savings_Investment_Value
data['Total_Portfolio_Value'] = 0.0
initial_investment = 0.0

for i in range(1, num_days):
    # Calculate the portfolio value based on the return
    data.loc[data.index[i], 'Total_Portfolio_Value'] = data.loc[data.index[i-1], 'Total_Portfolio_Value'] * (1 + data.loc[data.index[i], 'Portfolio_Return_YTD'])

    # Add daily savings to the portfolio
    data.loc[data.index[i], 'Total_Portfolio_Value'] += data.loc[data.index[i], 'Savings_Investment_Value']

data['Total_Portfolio_Value'] = data['Total_Portfolio_Value'].round(2)

# Set constant values for Investment Goal and Risk Tolerance
data['Investment_Goal'] = 'Wealth Accumulation'
data['Risk_Tolerance'] = 'Medium'

# Reset index to make 'Date' a column
data.reset_index(inplace=True)
data.rename(columns={'index': 'Date'}, inplace=True)

# Save to CSV in the specified path
file_path = r"C:\Users\manik\Downloads\synthetic_customer_dataset_daily.csv"
data.to_csv(file_path, index=False)

print("Synthetic customer dataset with daily values and inflation effect created and saved successfully.")
