"""Tata Steel Machine Failure Prediction - Streamlit app.

Deployment: keep this file, requirements.txt and
`tata_steel_failure_xgb_model.pkl` together in the repo root.
"""
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="Machine Failure Prediction", page_icon="⚙️", layout="centered")

# Model path is relative to this file so it works locally and on Streamlit Cloud.
MODEL_PATH = Path(__file__).parent / "tata_steel_failure_xgb_model.pkl"


@st.cache_resource
def load_model_artifacts(path: Path):
    """Load the model + preprocessing artifacts saved by the notebook."""
    if not path.exists():
        st.error(
            f"Model file `{path.name}` not found next to app.py.\n\n"
            "Download it from your Google Drive (saved by the notebook's "
            "'Save the File' cell) and add it to the repository root."
        )
        st.stop()
    return joblib.load(path)


artifacts = load_model_artifacts(MODEL_PATH)
model = artifacts["model"]
scaler = artifacts["scaler"]
numeric_features = artifacts["numeric_features"]
log_transform_cols = artifacts["log_transform_cols"]
feature_columns = artifacts["feature_columns"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Must mirror the feature engineering used during training."""
    df = df.copy()
    df["power"] = df["torque"] * df["rotational_speed"]
    df["temp_diff"] = df["process_temperature"] - df["air_temperature"]
    df["wear_torque"] = df["tool_wear"] * df["torque"]
    df["torque_per_speed"] = df["torque"] / df["rotational_speed"]
    return df


def predict(inputs: dict, machine_type: str):
    df = pd.DataFrame(
        {
            **{k: [float(v)] for k, v in inputs.items()},
            "Type_L": [int(machine_type == "L")],
            "Type_M": [int(machine_type == "M")],
        }
    )
    df = engineer_features(df)
    for col in log_transform_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    df[numeric_features] = scaler.transform(df[numeric_features])
    X = df[feature_columns]
    return int(model.predict(X)[0]), float(model.predict_proba(X)[:, 1][0])


# ------------------------------- UI -----------------------------------------
st.title("⚙️ Tata Steel Machine Failure Prediction")
st.markdown(
    "Enter the machine's current sensor readings and click **Predict** "
    "to estimate the probability of a machine failure."
)

with st.sidebar:
    st.header("About")
    st.write(
        "A tuned **XGBoost** classifier trained on the Tata Steel "
        "predictive-maintenance dataset flags machines at risk of failure "
        "from sensor readings."
    )
    st.caption(f"Model artifact: `{MODEL_PATH.name}`")

with st.form("prediction_form"):
    st.subheader("Sensor Readings")
    c1, c2 = st.columns(2)
    with c1:
        air_temperature = st.number_input("Air Temperature [K]", 290.0, 310.0, 300.0, 0.1)
        process_temperature = st.number_input("Process Temperature [K]", 300.0, 320.0, 310.0, 0.1)
        rotational_speed = st.number_input("Rotational Speed [rpm]", 1000, 3000, 1500, 10)
    with c2:
        torque = st.number_input("Torque [Nm]", 0.0, 100.0, 40.0, 0.1)
        tool_wear = st.number_input("Tool Wear [min]", 0, 300, 100, 1)
        machine_type = st.selectbox("Machine Type", ["L", "M", "H"])
    submitted = st.form_submit_button("Predict Machine Failure", use_container_width=True)

if submitted:
    try:
        pred, proba = predict(
            {
                "air_temperature": air_temperature,
                "process_temperature": process_temperature,
                "rotational_speed": rotational_speed,
                "torque": torque,
                "tool_wear": tool_wear,
            },
            machine_type,
        )
    except Exception as exc:  # surface a readable error instead of a traceback
        st.error(f"Prediction failed: {exc}")
        st.stop()

    st.subheader("Prediction Result")
    if pred == 1:
        st.error(f"⚠️ Machine is **predicted to FAIL** (failure probability: {proba:.2%})")
        st.warning("Immediate inspection / maintenance action recommended.")
    else:
        st.success(f"✅ Machine is **predicted to be NORMAL** (failure probability: {proba:.2%})")
        st.info("No immediate failure detected.")
    st.progress(min(max(proba, 0.0), 1.0))
