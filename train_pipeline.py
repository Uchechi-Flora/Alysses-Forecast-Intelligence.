import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# 1. Load Data
df = pd.read_excel('skincare_business_dataset_updated.xlsx')
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month

# 2. Financial Metrics & Heuristic Target Labeling
df['Revenue'] = df['Units_Sold'] * df['Selling_Price']
df['COGS'] = df['Units_Sold'] * df['Cost_Price']
df['Total_Cost'] = df['COGS'] + df['Delivery_Costs'] + df['Ads_Spend']
df['Profit'] = df['Revenue'] - df['Total_Cost']
df['Margin'] = df['Profit'] / (df['Revenue'] + 1)

# Establish risk thresholds from the 2025 baseline data
margin_25th = df['Margin'].quantile(0.25)
brand_avg_returns = df['Returns'].mean()

# Mark rows as High Risk (1) if they fail 2 or more conditions simultaneously
cond_low_margin = df['Margin'] <= margin_25th
cond_high_returns = df['Returns'] > brand_avg_returns
cond_low_demand = df['Units_Sold'] < df['Units_Sold'].median()

df['Is_High_Risk'] = ((cond_low_margin.astype(int) + cond_high_returns.astype(int) + cond_low_demand.astype(int)) >= 2).astype(int)

# Train Classifier Model for Discontinuation Risk
features_list = ['Units_Sold', 'Margin', 'Returns', 'Delivery_Costs']
X_cls = df[features_list]
y_cls = df['Is_High_Risk']

scaler = StandardScaler()
X_cls_scaled = scaler.fit_transform(X_cls)

clf_model = RandomForestClassifier(n_estimators=100, random_state=42)
clf_model.fit(X_cls_scaled, y_cls)

# 3. Compile Demand Tables & Pre-Calculations for UI Speed
demand_matrix = df.groupby(['Product_Name', 'Customer_Region', 'Month'])['Units_Sold'].sum().reset_index()

# Pack everything away for deployment
artifacts = {
    'classifier': clf_model,
    'scaler': scaler,
    'raw_data': df,
    'demand_matrix': demand_matrix,
    'features_list': features_list
}

joblib.dump(artifacts, 'alysses_production_artifacts.pkl')
print("🎉 Success: Machine learning models trained and exported as 'alysses_production_artifacts.pkl'")