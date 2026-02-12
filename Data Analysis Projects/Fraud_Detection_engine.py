import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve

class SentinelFraudEngine:
    def __init__(self, data_path=None):
        """Initialize and generate/load data."""
        if data_path:
            self.df = pd.read_csv(data_path)
        else:
            self._generate_mock_data()
        
        self.model = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
        self.scaler = StandardScaler()

    def _generate_mock_data(self):
        """Generates a synthetic dataset if no CSV is provided."""
        np.random.seed(42)
        n_samples = 5000
        # Logic: Fraud (1) is 5% of data, Legit (0) is 95%
        data = {
            'TransactionAmount': np.random.exponential(scale=100, size=n_samples),
            'IsInternational': np.random.choice([0, 1], size=n_samples, p=[0.9, 0.1]),
            'FrequencyPerDay': np.random.poisson(lam=3, size=n_samples),
            'TimeOfDay': np.random.randint(0, 24, size=n_samples),
            'IsFraud': np.random.choice([0, 1], size=n_samples, p=[0.95, 0.05])
        }
        self.df = pd.DataFrame(data)
        # Inject logic: High amount + International + Night time = Higher Fraud likelihood
        self.df.loc[(self.df['TransactionAmount'] > 250) & (self.df['IsInternational'] == 1), 'IsFraud'] = 1

    def preprocess_and_train(self):
        """Prepare data and train the ensemble model."""
        X = self.df.drop('IsFraud', axis=1)
        y = self.df['IsFraud']

        # Logic: Feature Scaling (Crucial for financial data)
        X_scaled = self.scaler.fit_transform(X)
        
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, stratify=y)
        
        print("🚀 Training Sentinel Fraud Engine...")
        self.model.fit(X_train, y_train)
        self.y_test, self.y_pred = y_test, self.model.predict(X_test)
        self.y_probs = self.model.predict_proba(X_test)[:, 1]

    def plot_performance_dashboard(self):
        """Generates a professional diagnostic dashboard."""
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle('Sentinel Fraud Engine: Diagnostic Report', fontsize=16)

        # 1. Confusion Matrix: Shows True Positives vs False Alarms
        cm = confusion_matrix(self.y_test, self.y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0])
        axes[0].set_title('Confusion Matrix')
        axes[0].set_xlabel('Predicted')
        axes[0].set_ylabel('Actual')
        

        # 2. Precision-Recall Curve: Essential for imbalanced datasets
        precision, recall, _ = precision_recall_curve(self.y_test, self.y_probs)
        axes[1].plot(recall, precision, color='crimson', lw=2)
        axes[1].set_title('Precision-Recall Curve')
        axes[1].set_xlabel('Recall')
        axes[1].set_ylabel('Precision')
        

        # 3. Feature Importance: Which factors drive fraud?
        importances = self.model.feature_importances_
        features = self.df.drop('IsFraud', axis=1).columns
        indices = np.argsort(importances)
        axes[2].barh(range(len(indices)), importances[indices], color='forestgreen', align='center')
        axes[2].set_yticks(range(len(indices)), [features[i] for i in indices])
        axes[2].set_title('Key Fraud Indicators')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show()
        
        print("\n--- Model Classification Report ---")
        print(classification_report(self.y_test, self.y_pred))

# --- EXECUTION ---
sentinel = SentinelFraudEngine()
sentinel.preprocess_and_train()
sentinel.plot_performance_dashboard()
