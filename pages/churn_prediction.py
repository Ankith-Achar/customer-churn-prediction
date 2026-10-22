
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
# Load model
# ---------------------------------------------------------
if not MODEL_PATH.exists():
    st.error(
        f"Model file not found: {MODEL_PATH}"
    )
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
except Exception as e:
    st.error(f"Could not load the model: {e}")
    st.stop()


# ---------------------------------------------------------
# Find dataset
# ---------------------------------------------------------
def find_dataset():
    possible_paths = [
        DATA_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
        ROOT / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
    ]

    for path in possible_paths:
        if path.exists():
            return path

    # Also accept a suffixed copy, such as the downloaded (8).csv.
    if DATA_DIR.exists():
        matches = list(DATA_DIR.glob("*Telco*Customer*Churn*.csv"))
        if matches:
            return matches[0]

    return None


# ---------------------------------------------------------
# Global feature importance
# ---------------------------------------------------------
def get_feature_importance(estimator, input_columns):
    classifier = estimator

    if hasattr(estimator, "named_steps"):
        steps = list(estimator.named_steps.values())
        if steps:
            classifier = steps[-1]

    if hasattr(classifier, "feature_importances_"):
        values = np.asarray(
            classifier.feature_importances_
        ).ravel()
    elif hasattr(classifier, "coef_"):
        values = np.asarray(classifier.coef_)
        if values.ndim > 1:
            values = np.mean(np.abs(values), axis=0)
        else:
            values = np.abs(values)
    else:
        return None

    names = None

    # Retrieve transformed feature names when possible.
    if hasattr(estimator, "named_steps"):
        for step in estimator.named_steps.values():
            if hasattr(step, "get_feature_names_out"):
                try:
                    names = list(step.get_feature_names_out())
                except Exception:
                    pass

    if names is None or len(names) != len(values):
        if hasattr(estimator, "feature_names_in_"):
            candidate_names = list(estimator.feature_names_in_)
            if len(candidate_names) == len(values):
                names = candidate_names

    if names is None or len(names) != len(values):
        if len(input_columns) == len(values):
            names = list(input_columns)
        else:
            names = [
                f"Encoded feature {i + 1}"
                for i in range(len(values))
            ]

    result = pd.DataFrame({
        "Feature": names,
        "Importance": values
    })

    return result.sort_values(
        "Importance", ascending=False
    ).head(12)


# ---------------------------------------------------------
# Individual SHAP explanation
# ---------------------------------------------------------

def get_shap_explanation(estimator, customer):
    dataset_path = find_dataset()

    if dataset_path is None:
        return None, "Telco dataset CSV not found in the data folder."

    if not hasattr(estimator, "predict_proba"):
        return None, "The saved model does not support predict_proba."

    try:
        import shap

        background = pd.read_csv(dataset_path)

        missing = [c for c in FEATURES if c not in background.columns]
        if missing:
            return None, "Missing dataset columns: " + ", ".join(missing)

        background = background[FEATURES].copy()

        numeric_columns = [
            "SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"
        ]
        categorical_columns = [
            c for c in FEATURES if c not in numeric_columns
        ]

        for col in numeric_columns:
            background[col] = pd.to_numeric(
                background[col], errors="coerce"
            )

        background = background.dropna().reset_index(drop=True)

        if background.empty:
            return None, "No usable rows found in the dataset."

        background = background.sample(
            n=min(30, len(background)), random_state=42
        )

        # Build a consistent category mapping from the dataset and input.
        category_maps = {}

        for col in categorical_columns:
            categories = list(
                background[col].dropna().astype(str).unique()
            )

            input_value = str(customer.iloc[0][col])
            if input_value not in categories:
                categories.append(input_value)

            category_maps[col] = {
                value: i for i, value in enumerate(categories)
            }

        def encode_frame(frame):
            result = frame[FEATURES].copy()

            for col in categorical_columns:
                result[col] = (
                    result[col].astype(str)
                    .map(category_maps[col])
                    .astype(float)
                )

            for col in numeric_columns:
                result[col] = pd.to_numeric(
                    result[col], errors="coerce"
                )

            return result.astype(float)

        def decode_frame(array):
            result = pd.DataFrame(array, columns=FEATURES)

            for col in categorical_columns:
                reverse_map = {
                    float(i): value
                    for value, i in category_maps[col].items()
                }

                def decode_value(value):
                    number = float(value)
                    nearest = min(
                        reverse_map,
                        key=lambda key: abs(key - number)
                    )
                    return reverse_map[nearest]

                result[col] = result[col].map(decode_value)

            for col in numeric_columns:
                result[col] = pd.to_numeric(
                    result[col], errors="coerce"
                )

            return result[FEATURES]

        background_encoded = encode_frame(background)
        customer_encoded = encode_frame(customer)

        classes = list(estimator.classes_)
        if 1 not in classes:
            return None, "Model class 1 for churn was not found."

        churn_index = classes.index(1)

        def predict_probability(array):
            decoded = decode_frame(array)
            probabilities = estimator.predict_proba(decoded)
            return probabilities[:, churn_index]

        masker = shap.maskers.Independent(background_encoded)

        explainer = shap.PermutationExplainer(
            predict_probability,
            masker
        )

        explanation = explainer(
            customer_encoded,
            max_evals=2 * len(FEATURES) + 1
        )

        values = np.asarray(explanation.values).reshape(-1)

        if len(values) != len(FEATURES):
            return None, "Unexpected SHAP output shape."

        result = pd.DataFrame({
            "Feature": FEATURES,
            "SHAP value": values
        })

        result["Absolute impact"] = result["SHAP value"].abs()

        return (
            result.sort_values(
                "Absolute impact", ascending=False
            ).head(10),
            None
        )

    except Exception as e:
        return None, f"{type(e).__name__}: {e}"


# ---------------------------------------------------------
# Customer input form
# ---------------------------------------------------------
with st.form("churn_prediction_form"):

    st.subheader("Customer Details")

    col_profile, col_services, col_billing = st.columns(
        [1, 2, 1.2]
    )

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


# ---------------------------------------------------------
# Predict once and store the customer for subsequent reruns
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

        st.session_state["churn_customer"] = customer
        st.session_state["churn_prediction"] = prediction
        st.session_state["churn_prediction_error"] = None
        st.session_state["shap_result"] = None
        st.session_state["shap_error"] = None

        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(customer)[0]
            classes = list(model.classes_)

            if 1 not in classes:
                st.session_state["churn_prediction_error"] = (
                    "The model's class labels do not include 1. "
                    "Check the target encoding used during training."
                )
            else:
                st.session_state["churn_probability"] = float(
                    probabilities[classes.index(1)]
                )
        else:
            st.session_state["churn_probability"] = None

    except Exception as e:
        st.session_state["churn_prediction_error"] = (
            f"Prediction failed: {type(e).__name__}: {e}"
        )


# ---------------------------------------------------------
# Display stored prediction and explanations
# This remains visible after the SHAP button is clicked.
# ---------------------------------------------------------
if st.session_state.get("churn_prediction_error"):
    st.error(st.session_state["churn_prediction_error"])

elif "churn_customer" in st.session_state:
    customer = st.session_state["churn_customer"]
    prediction = st.session_state["churn_prediction"]
    churn_probability = st.session_state.get("churn_probability")

    st.divider()
    st.subheader("Prediction Result")

    result_col, probability_col = st.columns([1, 1])

    with result_col:
        if prediction == 1:
            st.error("⚠️ Customer may churn")
        else:
            st.success("✅ Customer may stay")

    with probability_col:
        if churn_probability is not None:
            st.metric(
                "Estimated churn probability",
                f"{churn_probability:.1%}"
            )
            st.progress(
                max(0.0, min(1.0, churn_probability))
            )
        else:
            st.info("This model does not provide churn probabilities.")

    st.caption(
        "The prediction is an estimate from the trained model, "
        "not a guarantee of future customer behaviour."
    )

    # -----------------------------------------------------
    # Global feature importance
    # -----------------------------------------------------
    st.divider()
    st.subheader("What influences churn overall?")

    importance = get_feature_importance(model, FEATURES)

    if importance is not None and not importance.empty:
        importance_chart = importance.set_index("Feature")[
            "Importance"
        ]

        st.bar_chart(importance_chart)

        st.caption(
            "This is overall model importance, not an explanation "
            "of this individual customer's prediction."
        )
    else:
        st.info(
            "This model does not expose a supported built-in feature "
            "importance measure. SHAP can still be attempted below."
        )

    # -----------------------------------------------------
    # Individual SHAP explanation
    # -----------------------------------------------------
    st.divider()
    st.subheader("Why did the model make this prediction?")

    st.markdown(
        "- **Positive SHAP value:** pushes the model score towards churn.\n"
        "- **Negative SHAP value:** pushes the model score away from churn.\n"
        "- Larger absolute values indicate a stronger contribution "
        "to the model score."
    )

    if st.button(
        "Explain this prediction with SHAP",
        key="explain_prediction"
    ):
        st.session_state["shap_result"] = None
        st.session_state["shap_error"] = None

        with st.spinner(
            "Calculating SHAP explanation. This may take a little time..."
        ):
            shap_result, shap_error = get_shap_explanation(
                model, customer
            )

        st.session_state["shap_result"] = shap_result
        st.session_state["shap_error"] = shap_error

    # Read stored explanation after every rerun.
    if st.session_state.get("shap_error"):
        st.warning(st.session_state["shap_error"])

    shap_result = st.session_state.get("shap_result")

    if shap_result is not None:
        chart_data = shap_result.set_index("Feature")[
            "SHAP value"
        ].sort_values()

        st.markdown("**Feature contributions for this customer**")
        st.bar_chart(chart_data)

        st.dataframe(
            shap_result[["Feature", "SHAP value"]],
            use_container_width=True,
            hide_index=True
        )

        st.caption(
            "SHAP values explain the model score relative to the "
            "background customers. They describe model behaviour, "
            "not causal effects."
        )
