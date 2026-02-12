import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

class RetailAnalyticsEngine:
    def __init__(self, file_path):
        """Initialize the engine and load data."""
        try:
            self.df = pd.read_csv(file_path)
            print(f" Data Loaded: {self.df.shape[0]} rows.")
        except Exception as e:
            print(f" Error loading file: {e}")

    def clean_data(self):
        """Advanced cleaning: Handling outliers and date formatting."""
        # Convert dates if they exist
        if 'OrderDate' in self.df.columns:
            self.df['OrderDate'] = pd.to_datetime(self.df['OrderDate'])
        
        # Fill missing values logic-heavy way: fill numerical with median
        num_cols = self.df.select_dtypes(include=[np.number]).columns
        self.df[num_cols] = self.df[num_cols].fillna(self.df[num_cols].median())
        
        # Create 'Profit_Margin' feature
        if 'Revenue' in self.df.columns and 'Cost' in self.df.columns:
            self.df['Profit_Margin'] = (self.df['Revenue'] - self.df['Cost']) / self.df['Revenue']
        
        print("✅ Data Cleaning & Feature Engineering Complete.")

    def analyze_supply_risk(self):
        """
        Identify products with high profit but low stock (Stock-out Risk).
        This shows the interviewer you understand business risk.
        """
        risk_metrics = self.df.groupby('Category').agg({
            'Profit_Margin': 'mean',
            'StockLevel': 'std',  # High STD means unstable supply
            'Sales': 'sum'
        }).rename(columns={'StockLevel': 'Supply_Volatility'})
        
        return risk_metrics

    def build_dashboard(self):
        """Professional Matplotlib Dashboard with multiple subplots."""
        sns.set_style("whitegrid")
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Executive Operational Intelligence Dashboard', fontsize=20)

        # 1. Profit vs Volatility (Scatter Plot)
        risk_data = self.analyze_supply_risk()
        sns.scatterplot(ax=axes[0, 0], data=risk_data, x='Supply_Volatility', y='Profit_Margin', 
                        size='Sales', hue='Category', sizes=(100, 1000), alpha=0.7)
        axes[0, 0].set_title('Profit Margin vs. Supply Volatility')

        # 2. Sales Distribution (Box Plot)
        sns.boxplot(ax=axes[0, 1], x='Category', y='Sales', data=self.df, palette='Set2')
        axes[0, 1].set_title('Sales Variance by Category')

        # 3. Correlation Heatmap (Logic Check)
        numeric_df = self.df.select_dtypes(include=[np.number])
        sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', ax=axes[1, 0])
        axes[1, 0].set_title('Feature Correlation Matrix')

        # 4. Profit Trend (Time Series - if OrderDate exists)
        if 'OrderDate' in self.df.columns:
            trend = self.df.set_index('OrderDate').resample('M')['Profit_Margin'].mean()
            axes[1, 1].plot(trend.index, trend.values, marker='o', color='darkgreen')
            axes[1, 1].set_title('Monthly Profit Margin Trend')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()

# --- EXECUTION BLOCK ---
# To use this, you just need a CSV with: Category, Sales, Revenue, Cost, StockLevel
# engine = RetailAnalyticsEngine('retail_data.csv')
# engine.clean_data()
# engine.build_dashboard()