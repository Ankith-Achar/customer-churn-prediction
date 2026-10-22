```python
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Churn Prediction",
    page_icon="🎯",
    layout="wide"
)

ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "models" / "best_churn_model.pkl"
DATA_DIR = ROOT / "data"

FEATURES = [
    "gender", "SeniorCitizen", "Partner", "Dependents", "tenure",
    "PhoneService", "MultipleLines", "InternetService",
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies", "Contract",
    "PaperlessBilling", "PaymentMethod", "MonthlyCharges",
    "TotalCharges"
]

# ---------------------------------------------------------
# Styling
# ---------------------------------------------------------
st.markdown("""
<style>
:root {
    --navy: #0B1F3A;
    --blue: #1F6FEB;
    --blue-soft: #EAF2FF;
    --border: #D6E0F0;
}
.block-container {
    padding-top: 2rem;
    padding-bottom: 1.5rem;
    max-width: 1250px;
}
h1 {
    color: var(--navy);
    font-size: 1.7rem !important;
}
h2, h3 {
    color: var(--navy);
}
[data-testid="stForm"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 0.9rem 1rem;
}
.section-title {
    font-size: 0.82rem;
    font-weight: 600;
    color: var(--navy);
    background: var(--blue-soft);
    border-left: 3px solid var(--blue);
    border-radius: 4px;
    padding: 0.3rem 0.6rem;
    margin-bottom: 0.3rem;
}
[data-testid="stMetricValue"] {
    color: var(--navy);
    font-size: 1.6rem;
}
</style>
""", unsafe_allow_html=True)

st.title("🎯 Customer Churn Prediction")
st.caption(
    "Predict customer churn and understand which features influence the result."
)

# ---------------------------------------------------------
# Load saved model
# ---------------------------------------------------------
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


def get_classifier(estimator):
    """Return the final classifier from a supported sklearn Pipeline."""
    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.values())
        if steps:
            return steps[-1]
    return estimator


def get_feature_importance(estimator, input_columns):
    """
    Extract importance from the fitted model.
    For a Pipeline, use transformed feature names when available.
    """
    classifier = get_classifier(estimator)

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(
            classifier.feature_importances_
        ).ravel()
    elif hasattr(classifier, "coef_"):
        coefficients = np.asarray(classifier.coef_)
        values = np.mean(np.abs(coefficients), axis=0)
    else:
        return None

    names = None

    if hasattr(estimator, "feature_names_in_"):
        names = list(estimator.feature_names_in_)

    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = list(step.get_feature_names_out())
                except Exception:
                    pass

    if names is None or len(names) != len(values):
        if len(values) == len(input_columns):
            names = list(input_columns)
        else:
            names = [f"Encoded feature {i + 1}" for i in range(len(values))]

    result = pd.DataFrame({
        "Feature": names,
        "Importance": values
    })

    return result.sort_values(
        "Importance", ascending=False
    ).head(12)


def find_dataset():
    """Find the project's Telco CSV if it exists."""
    candidates = [
        DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ]

    for path in candidates:
        if path.exists():
            return path

    return None


def get_shap_explanation(estimator, customer):
    """
    Try a model-agnostic SHAP explanation using real customer records
    as background data. Return None with a message if unavailable.
    """
    dataset_path = find_dataset()

    if dataset_path is None:
        return None, (
            "The Telco dataset CSV was not found. Place the original CSV "
            "inside the data folder to enable individual SHAP explanations."
        )

    try:
        import shap

        background = pd.read_csv(dataset_path)

        # The original training dataset uses these same feature names.
        if not set(FEATURES).issubset(background.columns):
            return None, (
                "The CSV does not contain all expected model features. "
                "Individual SHAP explanations cannot be generated safely."
            )

        background = background[FEATURES].copy()

        # Match input data types expected by the model.
        for column in ["SeniorCitizen", "tenure"]:
            background[column] = pd.to_numeric(
                background[column], errors="coerce"
            )

        for column in ["MonthlyCharges", "TotalCharges"]:
            background[column] = pd.to_numeric(
                background[column], errors="coerce"
            )

        background = background.dropna()

        if background.empty:
            return None, "No usable background records were found."

        background = background.sample(
            n=min(40, len(background)), random_state=42
        )

        # Model-agnostic SHAP works through the saved estimator's
        # predict_proba interface, including supported Pipelines.
        if not hasattr(estimator, "predict_proba"):
            return None, (
                "This model does not provide predict_proba, so the "
                "current probability-based SHAP explanation is unavailable."
            )

        explainer = shap.Explainer(
            lambda data: estimator.predict_proba(
                pd.DataFrame(data, columns=FEATURES)
            )[:, list(estimator.classes_).index(1)],
            background,
            algorithm="permutation"
        )

        explanation = explainer(
            customer[FEATURES],
            max_evals=2 * len(FEATURES) + 1
        )

        values = np.asarray(explanation.values)

        if values.ndim == 2:
            values = values[0]

        if len(values) != len(FEATURES):
            return None, "SHAP returned an unexpected feature shape."

        result = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP value": values
        })

        result["Absolute impact"] = result["SHAP value"].abs()
        result = result.sort_values(
            "Absolute impact", ascending=False
        ).head(10)

        return result, None

    except ImportError:
        return None, (
            "SHAP is not installed in this environment. Install it with "
            "`pip install shap` to enable individual explanations."
        )
    except Exception as e:
        return None, f"SHAP explanation could not be generated: {e}"


# ---------------------------------------------------------
# Customer form
# ---------------------------------------------------------
with st.form("churn_prediction_form"):

    st.subheader("Customer Details")
    col_profile, col_services, col_billing = st.columns([1, 2, 1.2])

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
            "Monthly Charges", min_value=0.0, value=70.0
        )
        total_charges = st.number_input(
            "Total Charges", min_value=0.0, value=840.0
        )

    submitted = st.form_submit_button(
        "Predict Churn", use_container_width=True
    )


# ---------------------------------------------------------
# Prediction and visual explanations
# ---------------------------------------------------------
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
    }])[FEATURES]

    try:
        prediction = model.predict(customer)[0]

        st.divider()
        st.subheader("Prediction Result")

        churn_probability = None

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

            result_col, probability_col = st.columns([1, 1])

            with result_col:
                if prediction == 1:
                    st.error("⚠️ Customer may churn")
                else:
                    st.success("✅ Customer may stay")

            with probability_col:
                st.metric(
                    "Estimated churn probability",
                    f"{churn_probability:.1%}"
                )
                st.progress(
                    max(0.0, min(1.0, churn_probability))
                )
                st.caption(
                    "This is the model's estimated probability, "
                    "not a guarantee of future behaviour."
                )
        else:
            if prediction == 1:
                st.error("⚠️ Customer may churn")
            else:
                st.success("✅ Customer may stay")

            st.info(
                "This model does not provide churn probabilities."
            )

        # -------------------------------------------------
        # Overall feature importance
        # -------------------------------------------------
        st.divider()
        st.subheader("What influences churn overall?")
        st.caption(
            "This chart shows global model importance, when the saved "
            "classifier exposes a supported importance measure. It does "
            "not explain the current customer individually."
        )

        importance = get_feature_importance(model, FEATURES)

        if importance is not None and not importance.empty:
            importance_chart = importance.set_index("Feature")
            st.bar_chart(importance_chart["Importance"])
        else:
            st.info(
                "The saved model does not expose a compatible built-in "
                "feature importance or coefficient measure. A separate "
                "model-agnostic importance analysis would be needed."
            )

        # -------------------------------------------------
        # Individual SHAP explanation
        # -------------------------------------------------
        st.divider()
        st.subheader("Why did the model make this prediction?")
        st.caption(
            "SHAP estimates how each feature moves the model score "
            "relative to a background set of customers. Positive values "
            "push towards churn; negative values push away from churn."
        )

        if st.button(
            "Explain this prediction with SHAP",
            key="explain_prediction"
        ):
            with st.spinner("Calculating the explanation..."):
                shap_result, shap_error = get_shap_explanation(
                    model, customer
                )

            if shap_error:
                st.warning(shap_error)
            elif shap_result is not None:
                chart_data = shap_result.set_index("Feature")[
                    "SHAP value"
                ].sort_values()

                st.bar_chart(chart_data)

                st.markdown("**Largest feature contributions**")
                st.dataframe(
                    shap_result[["Feature", "SHAP value"]],
                    use_container_width=True,
                    hide_index=True
                )

                st.caption(
                    "These values explain the model score relative to "
                    "the selected background records. They are not "
                    "causal effects."
                )

    except Exception as e:
        st.error(
            f"Prediction or explanation failed: {e}. Check that the "
            "saved model expects these 19 input features."
        )
```
