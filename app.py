import streamlit as st
import pandas as pd
import shap
from xgboost import XGBClassifier
from src.preprocess import preprocess

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Intelligence Engine",
    layout="wide"
)

st.title("📊 Customer Churn Intelligence Engine")
st.caption("Predict customer churn and understand the drivers using Explainable AI")

# -----------------------------
# Load Data & Train Model
# -----------------------------
@st.cache_resource
def load_model_and_data():
    df = preprocess("data/telco_churn.csv")

    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    model = XGBClassifier(
        eval_metric="logloss",
        max_depth=4,
        n_estimators=150,
        learning_rate=0.1,
        random_state=42
    )
    model.fit(X, y)

    return model, X

model, X = load_model_and_data()

# -----------------------------
# Load SHAP Explainer (Fast)
# -----------------------------
@st.cache_resource
def load_explainer(model):
    return shap.TreeExplainer(model)

explainer = load_explainer(model)

# -----------------------------
# Sidebar Inputs
# -----------------------------
st.sidebar.header("Customer Information")

tenure = st.sidebar.slider("Tenure (months)", 0, 72, 12)
monthly_charges = st.sidebar.slider("Monthly Charges", 20, 150, 70)
total_charges = st.sidebar.slider("Total Charges", 0, 10000, 1000)

contract = st.sidebar.selectbox(
    "Contract Type",
    ["Month-to-month", "One year", "Two year"]
)

internet_service = st.sidebar.selectbox(
    "Internet Service",
    ["DSL", "Fiber optic", "No"]
)

payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer", "Credit card"]
)

# -----------------------------
# Build Input DataFrame
# -----------------------------
input_df = pd.DataFrame(columns=X.columns)
input_df.loc[0] = 0

input_df["tenure"] = tenure
input_df["MonthlyCharges"] = monthly_charges
input_df["TotalCharges"] = total_charges

def set_one_hot(prefix, value):
    col = f"{prefix}_{value}"
    if col in input_df.columns:
        input_df[col] = 1

set_one_hot("Contract", contract)
set_one_hot("InternetService", internet_service)
set_one_hot("PaymentMethod", payment_method)

input_df = input_df.fillna(0)

# -----------------------------
# Prediction & Explanation
# -----------------------------
if st.button("🔮 Predict Churn Risk"):
    probability = model.predict_proba(input_df)[0][1]

    st.subheader("Prediction Result")

    if probability >= 0.5:
        st.error(f"⚠ High Churn Risk — Probability: {probability:.2f}")
    else:
        st.success(f"✅ Low Churn Risk — Probability: {probability:.2f}")

    st.subheader("Why this prediction?")

    shap_values = explainer.shap_values(input_df)

    shap_df = (
        pd.DataFrame({
            "Feature": input_df.columns,
            "Impact": shap_values[0]
        })
        .assign(abs_impact=lambda x: x["Impact"].abs())
        .sort_values("abs_impact", ascending=False)
        .head(8)
    )

    st.bar_chart(
        shap_df.set_index("Feature")["Impact"]
    )
