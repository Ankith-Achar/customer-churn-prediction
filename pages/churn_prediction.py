import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_churn_model.pkl"

# ---------------------------------------------------------------------------
# Compact styling (navy / blue palette). UI only - no effect on prediction.
# ---------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --navy: #0B1F3A;
        --blue: #1F6FEB;
        --blue-soft: #EAF2FF;
        --border: #D6E0F0;
        --radius: 10px;
    }
    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 1.2rem;
        max-width: 1250px;
    }
    h1 {
        font-size: 1.7rem !important;
        margin-top: 0.5rem !important;
        padding-top: 0.25rem !important;
        color: var(--navy);
        padding: 0 0 0.2rem 0 !important;
    }
    h3 {
        font-size: 1.05rem !important;
        color: var(--navy);
        padding: 0.2rem 0 0.2rem 0 !important;
    }
    [data-testid="stForm"] {
        border: 1px solid var(--border);
        border-radius: var(--radius);
        padding: 0.9rem 1rem 0.8rem 1rem;
    }
    [data-testid="stForm"] [data-testid="stVerticalBlock"] {
        gap: 0.55rem;
    }
    [data-testid="stWidgetLabel"] p {
        font-size: 0.8rem;
        margin-bottom: 0;
    }
    .section-title {
        font-size: 0.82rem;
        font-weight: 600;
        color: var(--navy);
        background: var(--blue-soft);
        border-left: 3px solid var(--blue);
        border-radius: 4px;
        padding: 0.25rem 0.6rem;
        margin-bottom: 0.2rem;
    }
    [data-testid="stFormSubmitButton"] button {
        background-color: var(--blue);
        color: #FFFFFF;
        border: none;
        border-radius: var(--radius);
        font-weight: 600;
        padding: 0.35rem 1rem;
    }
    [data-testid="stFormSubmitButton"] button:hover {
        background-color: var(--navy);
        color: #FFFFFF;
    }
    [data-testid="stMetricValue"] {
        font-size: 1.6rem;
        color: var(--navy);
    }
    [data-testid="stMetricLabel"] p {
        font-size: 0.8rem;
    }
    [data-testid="stAlert"] {
        border-radius: var(--radius);
        padding: 0.6rem 0.9rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎯 Customer Churn Prediction")
st.caption("Enter the customer's information to predict churn.")

if not MODEL_PATH.exists():
    st.error(
        "Model file not found. Make sure "
        "best_churn_model.pkl is inside the models folder."
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()

with st.form("churn_prediction_form"):

    st.subheader("Customer Details")
    col_profile, col_services, col_billing = st.columns([1, 2, 1.2])

    # ---- Profile & account --------------------------------------------
    with col_profile:
        st.markdown(
            '<div class="section-title">👤 Profile</div>',
            unsafe_allow_html=True
        )
        gender = st.selectbox("Gender", ["Female", "Male"])
        senior = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)

    # ---- Services -------------------------------------------------------
    with col_services:
        st.markdown(
            '<div class="section-title">📡 Services</div>',
            unsafe_allow_html=True
        )
        svc_left, svc_right = st.columns(2)

        with svc_left:
            phone = st.selectbox("Phone Service", ["Yes", "No"])
            multiple_lines = st.selectbox(
                "Multiple Lines",
                ["No", "Yes", "No phone service"]
            )
            internet = st.selectbox(
                "Internet Service",
                ["DSL", "Fiber optic", "No"]
            )
            security = st.selectbox(
                "Online Security",
                ["No", "Yes", "No internet service"]
            )
            backup = st.selectbox(
                "Online Backup",
                ["No", "Yes", "No internet service"]
            )

        with svc_right:
            protection = st.selectbox(
                "Device Protection",
                ["No", "Yes", "No internet service"]
            )
            support = st.selectbox(
                "Tech Support",
                ["No", "Yes", "No internet service"]
            )
            streaming_tv = st.selectbox(
                "Streaming TV",
                ["No", "Yes", "No internet service"]
            )
            streaming_movies = st.selectbox(
                "Streaming Movies",
                ["No", "Yes", "No internet service"]
            )

    # ---- Contract & billing --------------------------------------------
    with col_billing:
        st.markdown(
            '<div class="section-title">💳 Contract & Billing</div>',
            unsafe_allow_html=True
        )
        contract = st.selectbox(
            "Contract",
            ["Month-to-month", "One year", "Two year"]
        )
        paperless = st.selectbox(
            "Paperless Billing", ["Yes", "No"]
        )
        payment = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )
        monthly_charges = st.number_input(
            "Monthly Charges",
            min_value=0.0,
            value=70.0
        )
        total_charges = st.number_input(
            "Total Charges",
            min_value=0.0,
            value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn",
        use_container_width=True
    )

if submitted:
    customer = pd.DataFrame([{
        "gender": gender,
        "SeniorCitizen": senior,
        "Partner": partner,
        "Dependents": dependents,
        "tenure": tenure,
        "PhoneService": phone,
        "MultipleLines": multiple_lines,
        "InternetService": internet,
        "OnlineSecurity": security,
        "OnlineBackup": backup,
        "DeviceProtection": protection,
        "TechSupport": support,
        "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies,
        "Contract": contract,
        "PaperlessBilling": paperless,
        "PaymentMethod": payment,
        "MonthlyCharges": monthly_charges,
        "TotalCharges": total_charges
    }])

    try:
        prediction = model.predict(customer)[0]

        st.subheader("Prediction Result")

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                st.error(
                    "The model's class labels do not include 1. "
                    "Check how Churn was encoded during training."
                )
                st.stop()

            churn_probability = float(
                probabilities[classes.index(1)]
            )

            left, middle, right = st.columns([1.3, 1, 2])

            with left:
                if prediction == 1:
                    st.error("Customer may churn")
                else:
                    st.success("Customer may stay")

            with middle:
                st.metric(
                    "Churn Probability",
                    f"{churn_probability:.1%}"
                )

            with right:
                st.caption("Churn probability gauge")
                st.progress(
                    max(0.0, min(1.0, churn_probability))
                )

        else:
            if prediction == 1:
                st.error("Customer may churn")
            else:
                st.success("Customer may stay")

            st.info("This model does not provide churn probabilities.")

    except Exception as e:
        st.error(
            f"Prediction failed: {e}. Check that the saved model "
            "expects these 19 input features."
        )
