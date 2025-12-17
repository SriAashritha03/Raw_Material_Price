# app.py
import math
import json
from datetime import datetime
from flask import Flask, render_template, request
import numpy as np
import joblib
import tensorflow as tf

# -------------------------------------------------
# Configuration
# -------------------------------------------------
MODEL_PATH = "gru_minmax_model.h5"      # trained GRU model
SCALER_PATH = "scaler_minmax.save"      # MinMaxScaler
SEQ_LEN = 12

FEATURES = [
    'min_price', 'max_price', 'month', 'week',
    'min_r4', 'max_r4', 'min_pct_1', 'max_pct_1'
]

# -------------------------------------------------
# Flask app
# -------------------------------------------------
app = Flask(__name__)

# -------------------------------------------------
# Load dropdown metadata (UI only)
# -------------------------------------------------
with open("dropdown_data.json", "r") as f:
    dropdowns = json.load(f)

districts = dropdowns["districts"]
commodities = dropdowns["commodities"]
varieties = dropdowns["varieties"]
grades = dropdowns["grades"]

# -------------------------------------------------
# Load model and scaler
# -------------------------------------------------
model = tf.keras.models.load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------
# Helper: build a dummy recent window
# (model-based forecasting without dataset)
# -------------------------------------------------
def build_dummy_window(target_date):
    """
    Builds a synthetic recent window using correct feature names,
    compatible with the trained MinMaxScaler.
    """
    if isinstance(target_date, str):
        target_date = datetime.strptime(target_date, "%Y-%m-%d")

    month = target_date.month
    week = target_date.isocalendar()[1]

    base_min = 0.5
    base_max = 0.6

    rows = []
    for _ in range(SEQ_LEN):
        rows.append({
            "min_price": base_min,
            "max_price": base_max,
            "month": month,
            "week": week,
            "min_r4": base_min,
            "max_r4": base_max,
            "min_pct_1": 0.0,
            "max_pct_1": 0.0
        })

    df_window = np.array([[r[f] for f in FEATURES] for r in rows])
    df_window = scaler.transform(
        np.asarray(df_window)
    )

    return df_window.reshape(1, SEQ_LEN, len(FEATURES))


# -------------------------------------------------
# Routes
# -------------------------------------------------
@app.route("/", methods=["GET"])
def index():
    return render_template(
        "index.html",
        districts=districts,
        commodities=commodities,
        varieties=varieties,
        grades=grades
    )

@app.route("/predict", methods=["POST"])
def predict():
    try:
        district = request.form.get("district")
        commodity = request.form.get("commodity")
        variety = request.form.get("variety") or "Any"
        grade = request.form.get("grade") or "Any"
        target_date = request.form.get("date")

        x = build_dummy_window(target_date)

        pred_scaled = model.predict(x)[0]

        dummy = np.zeros((1, len(FEATURES)))
        dummy[:, 0:2] = pred_scaled.reshape(1, 2)
        inv = scaler.inverse_transform(dummy)

        pred_min = float(inv[0, 0])
        pred_max = float(inv[0, 1])

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

    except Exception as e:
        # Render-safe error page
        return f"""
        <h3>Prediction failed</h3>
        <p>{str(e)}</p>
        <a href="/">Go Back</a>
        """

# -------------------------------------------------
# Run
# -------------------------------------------------
if __name__ == "__main__":
    app.run(debug=True)
