
import streamlit as st

st.set_page_config(
    page_title="Telco Churn | Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- CUSTOM STYLE ----------
st.markdown("""
<style>
    .stApp {
        background-color: #F3F6FA;
    }

    .block-container {
        padding-top: 2.2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    .hero {
        background: linear-gradient(120deg, #14243A, #244D73);
        padding: 2.2rem;
        border-radius: 18px;
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        color: white;
        font-size: 2.35rem;
        margin-bottom: 0.5rem;
    }

    .hero p {
        color: #E1EAF5;
        font-size: 1.05rem;
        margin-bottom: 0;
    }

    .section-title {
        color: #14243A;
        font-size: 1.45rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.4rem;
    }

    .nav-card {
        background-color: white;
        padding: 1.5rem;
        border: 1px solid #E0E7EF;
        border-radius: 16px;
        min-height: 210px;
    }

    .nav-card h3 {
        color: #14243A;
        margin-top: 0.4rem;
    }

    .nav-card p {
        color: #526176;
        line-height: 1.6;
    }

    .icon-circle {
        background-color: #E7F0FC;
        width: 48px;
        height: 48px;
        border-radius: 12px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 25px;
    }

    div.stButton > button,
    div.stLinkButton > a {
        border-radius: 10px;
        font-weight: 600;
        min-height: 44px;
    }

    [data-testid="stMetric"] {
        background-color: white;
        border: 1px solid #E0E7EF;
        padding: 1rem;
        border-radius: 12px;
    }

    footer {
        visibility: hidden;
    }
</style>
""", unsafe_allow_html=True)


# ---------- SIDEBAR ----------
with st.sidebar:
    st.markdown("## 📊 Telco Analytics")
    st.caption("Machine Learning Project")
    st.divider()
    st.markdown("### Welcome")
    st.write("Explore customer churn analysis and predictions.")
    st.divider()
    st.caption("Phase 3 · Model Deployment")


# ---------- HERO SECTION ----------
st.markdown("""
<div class="hero">
    <h1>Telco Customer Churn</h1>
    <p>
        Understand customer behaviour, explore machine learning
        results, and estimate the likelihood of customer churn.
    </p>
</div>
""", unsafe_allow_html=True)


# ---------- OVERVIEW ----------
st.markdown(
    '<div class="section-title">Project Overview</div>',
    unsafe_allow_html=True
)

st.write(
    "This application brings together the analysis of customer "
    "churn and an interactive prediction tool in one place."
)

# ---------- NAVIGATION CARDS ----------
st.markdown(
    '<div class="section-title">Explore the Application</div>',
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="nav-card">
        <div class="icon-circle">📈</div>
        <h3>Project Details</h3>
        <p>
            Explore the dataset, exploratory data analysis,
            preprocessing methodology, machine learning models,
            and evaluation results.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.page_link(
        "pages/project_details.py",
        label="Explore Project  →",
        icon="📊",
        use_container_width=True
    )

with col2:
    st.markdown("""
    <div class="nav-card">
        <div class="icon-circle">🎯</div>
        <h3>Churn Prediction</h3>
        <p>
            Enter customer details, run the trained model,
            and view the predicted churn outcome and probability.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.write("")
    st.page_link(
        "pages/churn_prediction.py",
        label="Predict Churn  →",
        icon="🔍",
        use_container_width=True
    )


# ---------- FOOTER ----------
st.divider()

left, right = st.columns([3, 1])

with left:
    st.caption("Telco Customer Churn Prediction · Machine Learning Lab")

with right:
    st.caption("Phase 3 Deployment")
