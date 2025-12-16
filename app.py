# app.py
import os, math
from datetime import timedelta
from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf

# Config / filenames (edit if needed)
DATA_PATH = "data/Turmeric.xlsx"        # optional: dataset used to populate dropdowns
MODEL_PATH = "gru_minmax_model.h5"      # or .keras
SCALER_PATH = "scaler_minmax.save"
SEQ_LEN = 12
FEATURES = ['min_price','max_price','month','week','min_r4','max_r4','min_pct_1','max_pct_1']

app = Flask(__name__)

# --------- Load model & scaler ---------
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found at {MODEL_PATH}")

if not os.path.exists(SCALER_PATH):
    raise FileNotFoundError(f"Scaler file not found at {SCALER_PATH}")

# load Keras model
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

scaler = joblib.load(SCALER_PATH)

# --------- Load dataset for dropdowns (optional) ---------
if os.path.exists(DATA_PATH):
    df = pd.read_excel(DATA_PATH)
    df.columns = df.columns.str.strip()
    df['Price Date'] = pd.to_datetime(df['Price Date'], errors='coerce')
    for c in ['District Name','Commodity','Variety','Grade']:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip()
    districts = sorted(df['District Name'].dropna().unique().tolist())
    commodities = sorted(df['Commodity'].dropna().unique().tolist())
    varieties = sorted(df['Variety'].dropna().unique().tolist())
    grades = sorted(df['Grade'].dropna().unique().tolist())
else:
    # fallback: minimal dropdowns
    df = None
    districts = []
    commodities = []
    varieties = []
    grades = []

# --------- Helper functions ---------
def build_ts_for_combo(df_local, district, commodity, variety=None, grade=None, freq='W'):
    sub = df_local[(df_local['District Name'] == district) & (df_local['Commodity'] == commodity)].copy()
    if variety:
        sub = sub[sub['Variety'] == variety]
    if grade:
        sub = sub[sub['Grade'] == grade]
    if len(sub) == 0:
        return None
    ts = sub[['Price Date','Min Price (Rs./Quintal)','Max Price (Rs./Quintal)']].rename(
        columns={'Min Price (Rs./Quintal)':'min_price','Max Price (Rs./Quintal)':'max_price'})
    ts = ts.set_index('Price Date').sort_index()
    ts_res = ts.resample(freq).mean().interpolate(limit_direction='both').fillna(method='bfill').fillna(method='ffill')
    return ts_res

def add_time_features(df_ts):
    df = df_ts.copy()
    df['month'] = df.index.month
    df['week'] = df.index.isocalendar().week.astype(int)
    df['min_r4'] = df['min_price'].rolling(window=4, min_periods=1).mean()
    df['max_r4'] = df['max_price'].rolling(window=4, min_periods=1).mean()
    df['min_pct_1'] = df['min_price'].pct_change().fillna(0)
    df['max_pct_1'] = df['max_price'].pct_change().fillna(0)
    df = df.fillna(method='bfill').fillna(method='ffill')
    return df

def inverse_target(scaled_vals, scaler_local):
    n = scaled_vals.shape[0]
    dummy = np.zeros((n, len(FEATURES)))
    dummy[:, 0:2] = scaled_vals
    dummy_df = pd.DataFrame(dummy, columns=FEATURES)
    inv = scaler_local.inverse_transform(dummy_df)
    return inv[:,0], inv[:,1]

def forecast_to_date_simple(df_feat, scaler_local, model_local, target_date):
    last_date = df_feat.index[-1]
    if isinstance(target_date, str):
        target_date = pd.to_datetime(target_date)
    days_ahead = (target_date - last_date).days
    if days_ahead < 0:
        # return last known
        return float(df_feat['min_price'].iloc[-1]), float(df_feat['max_price'].iloc[-1])
    steps_ahead = math.ceil(days_ahead / 7) if days_ahead>0 else 1

    last_window_df = df_feat.iloc[-SEQ_LEN:].copy()
    last_window_scaled = scaler_local.transform(last_window_df[FEATURES])
    window = last_window_scaled.copy()
    history_min = list(last_window_df['min_price'].values)
    history_max = list(last_window_df['max_price'].values)
    preds_scaled = []

    for step in range(steps_ahead):
        x = window.reshape(1, window.shape[0], window.shape[1])
        pred = model_local.predict(x)[0]
        preds_scaled.append(pred)
        pmin, pmax = inverse_target(pred.reshape(1,2), scaler_local)
        pmin, pmax = float(pmin[0]), float(pmax[0])
        history_min.append(pmin); history_max.append(pmax)
        next_date = last_date + timedelta(days=(step+1)*7)
        next_month = next_date.month
        next_week = int(next_date.isocalendar()[1])
        min_r4 = float(np.mean(history_min[-4:]))
        max_r4 = float(np.mean(history_max[-4:]))
        min_pct_1 = (pmin / history_min[-2] - 1) if len(history_min)>=2 and history_min[-2]!=0 else 0
        max_pct_1 = (pmax / history_max[-2] - 1) if len(history_max)>=2 and history_max[-2]!=0 else 0
        next_row = np.array([pmin, pmax, next_month, next_week, min_r4, max_r4, min_pct_1, max_pct_1]).reshape(1,-1)
        next_row_df = pd.DataFrame(next_row, columns=FEATURES)
        next_row_scaled = scaler_local.transform(next_row_df)
        window = np.vstack([window[1:], next_row_scaled[0]])

    final_scaled = np.array(preds_scaled[-1]).reshape(-1,2)
    final_min, final_max = inverse_target(final_scaled, scaler_local)
    return float(final_min[0]), float(final_max[0])

# --------- Routes ---------
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", districts=districts, commodities=commodities, varieties=varieties, grades=grades)

@app.route("/predict", methods=["POST"])
def predict():
    district = request.form.get("district")
    commodity = request.form.get("commodity")
    variety = request.form.get("variety") or None
    grade = request.form.get("grade") or None
    target_date = request.form.get("date")

    if df is None:
        return "<h3>Dataset not found</h3><a href='/'>Go Back</a>"

    ts = build_ts_for_combo(df, district, commodity, variety, grade)
    if ts is None:
        return "<h3>No data found for selected inputs</h3><a href='/'>Go Back</a>"

    df_feat = add_time_features(ts)

    if len(df_feat) < SEQ_LEN + 1:
        return "<h3>Not enough data for prediction</h3><a href='/'>Go Back</a>"

    pred_min, pred_max = forecast_to_date_simple(
        df_feat, scaler, model, target_date
    )

    # THIS IS THE KEY FIX
    return render_template(
        "result.html",
        district=district,
        commodity=commodity,
        variety=variety,
        grade=grade,
        date=target_date,
        min_price=round(pred_min, 2),
        max_price=round(pred_max, 2)
    )


if __name__ == "__main__":
    from datetime import timedelta
    import math
    app.run(debug=True, host="0.0.0.0", port=5000)
