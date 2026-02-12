#Strategic Customer Retention: A Multi-Factor Churn & CLV Analysis

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# 1. Load the Dataset
# Replace 'your_data.csv' with your actual file path
try:
    df = pd.read_csv('Telco-Customer-Churn.csv')
    print("Dataset loaded successfully!")
except FileNotFoundError:
    print("Error: Please ensure the CSV file is in the same directory.")

# 2. Data Cleaning (The Pandas Phase)
# Convert TotalCharges to numeric (handles errors caused by empty strings)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)

# Convert Churn from 'Yes'/'No' to 1/0 for mathematical analysis
df['Churn_Numeric'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

# 3. High-Level Visualization (The Matplotlib Phase)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# --- Plot 1: Distribution of Tenure vs Churn ---
# Shows how long people stay before they quit
ax1.hist(df[df['Churn'] == 'No']['tenure'], bins=30, alpha=0.5, label='Stayed', color='blue')
ax1.hist(df[df['Churn'] == 'Yes']['tenure'], bins=30, alpha=0.5, label='Churned', color='red')
ax1.set_xlabel('Tenure (Months)')
ax1.set_ylabel('Number of Customers')
ax1.set_title('Customer Loyalty vs. Churn')
ax1.legend()

# --- Plot 2: Contract Type Impact ---
# This identifies the "Business Problem"
contract_churn = df.groupby('Contract')['Churn_Numeric'].mean() * 100
ax2.bar(contract_churn.index, contract_churn.values, color=['#ff9999','#66b3ff','#99ff99'])
ax2.set_ylabel('Churn Rate (%)')
ax2.set_title('Which Contract Type has the Highest Churn?')

plt.tight_layout()
plt.show()

# 4. Logical Insight: Identifying High-Value At-Risk Customers
# Criteria: Paying > $80/month but on a Month-to-Month contract
high_value_at_risk = df[(df['MonthlyCharges'] > 80) & (df['Contract'] == 'Month-to-month')]

print(f"\n--- Analysis Summary ---")
print(f"Total High-Value At-Risk Customers: {len(high_value_at_risk)}")
print(f"Average Monthly Charge of Churned Users: ${df[df['Churn_Numeric'] == 1]['MonthlyCharges'].mean():.2f}")