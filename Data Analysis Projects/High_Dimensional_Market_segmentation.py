import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.preprocessing import MinMaxScaler

class PricingOptimizer:
    def __init__(self, n_days=365):
        """Generates synthetic time-series data with seasonality."""
        np.random.seed(42)
        dates = pd.date_range(start='2025-01-01', periods=n_days)
        
        # Logic: Base demand + Seasonality (Weekend spikes) + Noise
        base_demand = 100
        seasonality = 20 * np.sin(2 * np.pi * dates.dayofweek / 7)
        noise = np.random.normal(0, 5, n_days)
        
        self.df = pd.DataFrame({
            'Date': dates,
            'Demand': (base_demand + seasonality + noise).astype(int),
            'Price': np.random.uniform(10, 20, n_days)
        })
        self.df.set_index('Date', inplace=True)

    def forecast_demand(self, days_to_predict=14):
        """Uses Triple Exponential Smoothing (Holt-Winters) for forecasting."""
        # This handles both trend and seasonality
        model = ExponentialSmoothing(self.df['Demand'], trend='add', seasonal='add', seasonal_periods=7)
        model_fit = model.fit()
        self.forecast = model_fit.forecast(days_to_predict)
        print("✅ Demand Forecast Generated.")

    def dynamic_pricing_logic(self):
        """
        The Logic-Heavy Part: 
        If Predicted Demand > Average, Increase Price by 15%.
        If Predicted Demand < Average, Decrease Price by 10%.
        """
        avg_demand = self.df['Demand'].mean()
        self.suggested_prices = self.forecast.apply(
            lambda x: 20 * 1.15 if x > avg_demand else 20 * 0.90
        )
        return self.suggested_prices

    def plot_intelligence_dashboard(self):
        """Professional Visualization of Predictions vs Pricing."""
        plt.figure(figsize=(14, 7))
        
        # Plot Historical Demand
        plt.plot(self.df.index[-30:], self.df['Demand'].tail(30), label='Historical Demand', color='gray', alpha=0.5)
        
        # Plot Forecasted Demand
        plt.plot(self.forecast.index, self.forecast.values, label='Predicted Demand', color='blue', marker='o')
        
        # Secondary axis for Price
        ax2 = plt.gca().twinx()
        ax2.step(self.suggested_prices.index, self.suggested_prices.values, where='post', label='Optimized Price', color='red', linestyle='--')
        ax2.set_ylabel('Suggested Price ($)')
        
        plt.title('Aegis: Forecast-Driven Dynamic Pricing Strategy', fontsize=16)
        plt.legend(loc='upper left')
        plt.show()

# EXECUTION
aegis = PricingOptimizer()
aegis.forecast_demand()
aegis.dynamic_pricing_logic()
aegis.plot_intelligence_dashboard()
