import os

import requests
import streamlit as st

# Set this to your deployed API's base URL (e.g. via Streamlit "Secrets" as
# API_URL, or an environment variable). Falls back to localhost for local dev.
API_URL = st.secrets.get("API_URL", os.environ.get("API_URL", "http://localhost:8000"))

st.set_page_config(page_title="Customer Churn Prediction", page_icon="📉")
st.title("Customer Churn Prediction App")
st.subheader("Based on Telecom Dataset")
st.caption(f"Connected to prediction API: {API_URL}")


@st.cache_data(ttl=300)
def get_features():
    resp = requests.get(f"{API_URL}/features", timeout=10)
    resp.raise_for_status()
    return resp.json()


try:
    features = get_features()
except Exception as e:
    st.error(
        f"Could not reach the prediction API at `{API_URL}`.\n\n"
        f"Details: {e}\n\n"
        "Set the correct API_URL in Streamlit secrets or the API_URL "
        "environment variable, and make sure the API is running."
    )
    st.stop()

# --- Categorical inputs ------------------------------------------------------
st.subheader("Categorical Features")
categorical_input_vals = {}
for feat in features["categorical"]:
    categorical_input_vals[feat["name"]] = st.selectbox(
        feat["name"], feat["options"], key=feat["name"]
    )

# --- Numerical inputs ---------------------------------------------------------
st.subheader("Numerical Features")
numerical_input_vals = {}
for col in features["numerical"]:
    numerical_input_vals[col] = st.number_input(col, key=col, value=0.0)

# --- Predict -------------------------------------------------------------------
if st.button("Predict"):
    payload = {**categorical_input_vals, **numerical_input_vals}
    if "SENIORCITIZEN" in payload:
        payload["SENIORCITIZEN"] = int(payload["SENIORCITIZEN"])

    try:
        resp = requests.post(f"{API_URL}/predict", json=payload, timeout=15)
        resp.raise_for_status()
        result = resp.json()
    except requests.exceptions.HTTPError:
        st.error(f"API rejected the request: {resp.text}")
    except Exception as e:
        st.error(f"Prediction request failed: {e}")
    else:
        prediction = result["prediction"]
        churn_prob = result.get("churn_probability")
        translation_dict = {"Yes": "Expected", "No": "Not Expected"}
        prediction_translate = translation_dict.get(prediction, prediction)
        st.write(
            f"The Prediction is **{prediction}**, hence customer is "
            f"**{prediction_translate}** to churn."
        )
        if churn_prob is not None:
            st.write(f"Churn probability: **{churn_prob:.1%}**")
