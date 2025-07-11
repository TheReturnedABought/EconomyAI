import os
import sys
import difflib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import joblib

# --- User configuration ---
DATE_FORMAT = '%d/%m/%Y'        # format of your 'Date' column
DATA_PATH = 'stockdata.csv'     # path to your input file (.csv, .xls, .xlsx)
MODEL_PATH = 'stock_price_rf_model.pkl'
TRAIN_SPLIT = 0.8               # fraction for time-based train/test split
CONTINUE_TRAINING = True        # if True, load existing model and retrain on full data
# -------------------------------

# Utility: load and preprocess data
def load_data(path, date_format):
    ext = os.path.splitext(path)[1].lower()
    if ext == '.csv':
        df = pd.read_csv(path, header=0)
    elif ext in ('.xls', '.xlsx', '.xlsm'):
        df = pd.read_excel(path, engine='openpyxl', header=0)
    else:
        raise ValueError(f"Unsupported file extension '{ext}' in {path}.")

    # detect multi-header
    unnamed = sum(str(c).startswith('Unnamed') for c in df.columns)
    if unnamed > len(df.columns) * 0.5:
        if ext == '.csv':
            df = pd.read_csv(path, header=[0,1])
        else:
            df = pd.read_excel(path, header=[0,1], engine='openpyxl')
        df.columns = [f"{t}_{m}".strip('_') for t, m in df.columns]

    # clean column names
    df.columns = [str(col).lstrip('\u00A0\u202F').strip() for col in df.columns]
    df = df.dropna(axis=1, how='all')

    # find date column
    date_cols = [c for c in df.columns if 'date' in c.lower()]
    if not date_cols:
        raise KeyError(f"No 'Date' column found; available columns: {df.columns.tolist()}")
    df = df.rename(columns={date_cols[0]: 'Date'})
    df['Date_raw'] = df['Date']
    df['Date'] = pd.to_datetime(df['Date_raw'], format=date_format, errors='coerce')
    df = df.dropna(subset=['Date']).set_index('Date').sort_index()
    return df

# Simulation & training for one target column
def run_for_target(df, target_col, continue_training=False, model_path=None):
    # prepare feature list
    features = [c for c in df.columns if c not in (target_col, 'Date_raw')]

    # clean data: strip symbols and convert
    df_clean = df.copy()
    df_clean[features] = df_clean[features].replace(r'[\$,]', '', regex=True)
    df_clean[target_col] = df_clean[target_col].replace(r'[\$,]', '', regex=True)
    X_all = df_clean[features].apply(pd.to_numeric, errors='coerce')
    y_all = pd.to_numeric(df_clean[target_col], errors='coerce')
    data = pd.concat([X_all, y_all], axis=1).dropna()
    X_all = data[features]
    y_all = data[target_col]

    # split into train/test
    split_idx = int(len(X_all) * TRAIN_SPLIT)
    X_train, X_test = X_all.iloc[:split_idx], X_all.iloc[split_idx:]
    y_train, y_test = y_all.iloc[:split_idx], y_all.iloc[split_idx:]

    # load or initialize model
    if continue_training and model_path and os.path.exists(model_path):
        print(f"Loading existing model from {model_path} and retraining...")
        model = joblib.load(model_path)
        model.fit(X_train, y_train)
    else:
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

    # make predictions
    preds = model.predict(X_test)

    # simulate trading strategy
    profit = 0.0
    y_test_vals = list(y_test)
    for i in range(len(preds) - 1):
        today_price = y_test_vals[i]
        tomorrow_price = y_test_vals[i+1]
        if preds[i] > today_price:
            profit += (tomorrow_price - today_price)

    # calculate buy-and-hold
    buy_hold = y_test_vals[-1] - y_test_vals[0] if len(y_test_vals) > 1 else 0.0

    # save updated model
    if model_path:
        joblib.dump(model, model_path)
        print(f"Model saved to {model_path}")

    return profit, buy_hold

# Main execution
if __name__ == '__main__':
    df = load_data(DATA_PATH, DATE_FORMAT)
    targets = [c for c in df.columns if c.endswith('_High')]
    summary = {}

    for tgt in targets:
        print(f"Processing {tgt}...")
        prof, bh = run_for_target(df, tgt, CONTINUE_TRAINING, MODEL_PATH)
        summary[tgt] = {'strategy_profit': prof, 'buy_hold_profit': bh}

    # print per-index summary
    print("\nSummary per index:")
    total_strategy = 0.0
    total_hold = 0.0
    for idx, res in summary.items():
        sp = res['strategy_profit']
        bh = res['buy_hold_profit']
        total_strategy += sp
        total_hold += bh
        print(f"{idx}: strategy gain = ${sp:.2f}, buy-hold gain = ${bh:.2f}")

    # print overall totals
    print("\nOverall totals:")
    print(f"Total strategy gain across all indices: ${total_strategy:.2f}")
    print(f"Total buy-hold gain across all indices: ${total_hold:.2f}")
