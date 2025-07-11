import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# 1. Load & clean data
df = pd.read_excel('stockdata.xlsx')

# Drop index and completely empty columns
df = df.drop(columns=[col for col in df.columns if 'Unnamed' in col and df[col].nunique() <= 1])

# Ensure Date column exists and convert to datetime
# (replace 'Date' with the actual name if it's different)
df = df.rename(columns={'Unnamed: 0': 'Index', 'BA': 'BA', 'Nividia': 'Nvidia', ' NOK': 'NOK'})
if 'Date' not in df.columns:
    # if your date ended up under a different header, adjust here
    df['Date'] = pd.to_datetime(df.iloc[:, 0])
else:
    df['Date'] = pd.to_datetime(df['Date'])

df = df.set_index('Date').sort_index()

# 2. Feature & target selection
# Example: predict Nvidia close price (column "Nvidia") from other symbols
target = 'Nvidia'
features = [col for col in df.columns if col != target]

X = df[features]
y = df[target]

# 3. Train/test split (time-based)
split = int(len(df) * 0.8)
X_train, X_test = X.iloc[:split], X.iloc[split:]
y_train, y_test = y.iloc[:split], y.iloc[split:]

# 4. Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 5. Evaluate
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
print(f"Test MSE: {mse:.4f}")

# 6. Save model
joblib.dump(model, 'stock_price_rf_model.pkl')
print("Model saved to 'stock_price_rf_model.pkl'")
