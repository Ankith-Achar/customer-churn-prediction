"""
Project Details page - Telco Customer Churn Prediction (MCA ML Lab mini-project).

Documents the work in Telco_Churn_ML_Project.ipynb (the source of truth):
  Q1 dataset & metadata -> Q3 EDA -> Q4 preprocessing -> Q5 model training
  -> Q6 evaluation -> Answer 6 conclusion.

Every number shown here is computed from the dataset CSV or from
data/model_results.csv. Nothing about results is hard-coded.

Required files (relative to the project root):
  data/WA_Fn-UseC_-Telco-Customer-Churn.csv   (a "(8)" suffixed copy also works)
  data/model_results.csv                       (optional, see Section F)
"""
from __future__ import annotations

import itertools
import re
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

try:  # Plotly is preferred when installed.
    import plotly.express as px
    import plotly.graph_objects as go

    PLOTLY = True
except ImportError:
    PLOTLY = False

try:  # Matplotlib is the fallback renderer.
    from matplotlib.figure import Figure

    MPL = True
except ImportError:
    MPL = False

# st.set_page_config must be the first Streamlit command on the page.
st.set_page_config(
    page_title="Project Details | Telco Churn",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================================================
# 1. Paths, constants and notebook-derived content
# =============================================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
RESULTS_PATH = DATA_DIR / "model_results.csv"
NOTEBOOK_NAME = "Telco_Churn_ML_Project.ipynb"

NAVY, BLUE, MID_BLUE = "#14243A", "#2F6DB5", "#5B8FCB"
RETAINED, CHURN = "#5B8FCB", "#C8553D"
CHART_H = 360
LABELS = {"No": "No Churn", "Yes": "Churn"}  # notebook wording
LABEL_ORDER = ["No Churn", "Churn"]
COLOR_MAP = {"No Churn": RETAINED, "Churn": CHURN}
CATEGORY_PALETTE = [BLUE, CHURN, "#7A8CA5", "#E0A458", "#3C8D7D"]

RANDOM_STATE = 42  # RANDOM_STATE in notebook cell 0
NUMERICAL_COLS = ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"]

# Notebook Q1 - feature dictionary (feature, notebook dtype, category, description)
FEATURE_DICTIONARY = [
    ("customerID", "object", "Identifier", "Unique customer key - drop before modeling"),
    ("gender", "object", "Demographic", "Male / Female"),
    ("SeniorCitizen", "int64", "Demographic", "1 = Senior (65+), 0 = Not Senior"),
    ("Partner", "object", "Demographic", "Has a partner (Yes/No)"),
    ("Dependents", "object", "Demographic", "Has dependents (Yes/No)"),
    ("tenure", "int64", "Account", "Months customer has been with the company"),
    ("PhoneService", "object", "Service", "Has phone service (Yes/No)"),
    ("MultipleLines", "object", "Service", "Multiple phone lines (Yes / No / No phone service)"),
    ("InternetService", "object", "Service", "DSL / Fiber optic / No internet"),
    ("OnlineSecurity", "object", "Service", "Online security add-on"),
    ("OnlineBackup", "object", "Service", "Online backup add-on"),
    ("DeviceProtection", "object", "Service", "Device protection add-on"),
    ("TechSupport", "object", "Service", "Tech support add-on"),
    ("StreamingTV", "object", "Service", "Streaming TV subscription"),
    ("StreamingMovies", "object", "Service", "Streaming movies subscription"),
    ("Contract", "object", "Account", "Month-to-month / One year / Two year contract"),
    ("PaperlessBilling", "object", "Account", "Uses paperless billing (Yes/No)"),
    ("PaymentMethod", "object", "Account", "Electronic check / Mailed check / Bank transfer / Credit card"),
    ("MonthlyCharges", "float64", "Account", "Current monthly bill amount (USD)"),
    ("TotalCharges", "object (needs fix)", "Account",
     "Total amount charged - contains blank strings (needs cleaning)"),
    ("Churn", "object (TARGET)", "Target", "Did customer leave in last month? Yes=1 / No=0"),
]

# Notebook Q4 - preprocessing decisions (step, action, reason)
PREPROCESSING_STEPS = [
    ("1", "TotalCharges: blanks coerced to NaN",
     "Customers with tenure = 0 have no billing history, so TotalCharges is blank. "
     "pd.to_numeric(errors='coerce') converts the column to numbers."),
    ("2", "Drop customerID",
     "Non-informative unique key; the notebook lists it as a data-leakage risk if used as a feature."),
    ("3", "Encode target: Yes -> 1, No -> 0",
     "Numeric labels make Churn = 1 the positive class for precision, recall, F1 and ROC-AUC."),
    ("4", "Median imputation for numerical features",
     "Robust to outliers compared with the mean; it only has an effect where values are missing."),
    ("5", "StandardScaler for numerical features",
     "Needed by distance- and margin-based models (KNN, SVM, Logistic Regression)."),
    ("6", "Most-frequent imputer + OneHotEncoder(drop='first') for categoricals",
     "Drops one dummy per feature to avoid multicollinearity (dummy-variable trap); "
     "handle_unknown='ignore' keeps prediction from failing on unseen categories."),
    ("7", "Stratified 80/20 split (random_state = 42)",
     "Keeps the churn proportion the same in the training and test sets, which matters for imbalanced data."),
    ("8", "All transforms inside a scikit-learn Pipeline",
     "Imputers, scaler and encoder are fitted on the training data only and then applied to the test data, "
     "which prevents data leakage."),
]

# Notebook Q5 - the eight models, with hyper-parameters copied from the MODELS dict
MODEL_INFO = [
    {"name": "Logistic Regression", "kind": "Traditional - Linear", "weighted": True,
     "definition": "A linear classifier that passes a weighted sum of the features through the logistic "
                   "(sigmoid) function to estimate the probability of the positive class.",
     "why": "Interpretable baseline; works well for linearly separable classes.",
     "params": "max_iter=2000, C=1.0, class_weight='balanced'"},
    {"name": "K-Nearest Neighbors (KNN)", "kind": "Traditional - Instance-based", "weighted": False,
     "definition": "Predicts the class of a customer from the labels of the k most similar training "
                   "customers, measured with a distance metric.",
     "why": "Captures local patterns without assuming a distribution.",
     "params": "n_neighbors=9, metric='euclidean', weights='distance'"},
    {"name": "Support Vector Machine (SVM)", "kind": "Traditional - Kernel", "weighted": True,
     "definition": "Finds the decision boundary with the widest margin between classes; the RBF kernel "
                   "allows a non-linear boundary.",
     "why": "Effective in high-dimensional spaces after one-hot expansion.",
     "params": "kernel='rbf', C=1.0, gamma='scale', probability=True, class_weight='balanced'"},
    {"name": "Decision Tree", "kind": "Traditional - Tree", "weighted": True,
     "definition": "Learns a hierarchy of if-then rules that split customers on feature values until "
                   "each leaf is mostly one class.",
     "why": "Interpretable; reveals decision rules for churn drivers.",
     "params": "max_depth=6, min_samples_leaf=20, class_weight='balanced'"},
    {"name": "Random Forest", "kind": "Ensemble - Bagging", "weighted": True,
     "definition": "Trains many decision trees on bootstrap samples with random feature subsets and "
                   "averages their votes, which reduces the variance of a single tree.",
     "why": "Reduces variance; handles feature interactions well.",
     "params": "n_estimators=200, max_depth=8, min_samples_leaf=10, class_weight='balanced'"},
    {"name": "Extra Trees Classifier", "kind": "Ensemble - Bagging (extreme randomization)", "weighted": True,
     "definition": "Like Random Forest, but split thresholds are drawn at random instead of optimised; "
                   "predictions of the trees are averaged. (scikit-learn's default trains each tree on the "
                   "full training set rather than a bootstrap sample.)",
     "why": "Faster than Random Forest; random thresholds further reduce overfitting.",
     "params": "n_estimators=200, max_depth=8, min_samples_leaf=10, class_weight='balanced'"},
    {"name": "Gradient Boosting Classifier", "kind": "Ensemble - Boosting", "weighted": False,
     "definition": "Builds trees one after another, each fitted to reduce the remaining error "
                   "(loss gradient) of the ensemble built so far.",
     "why": "Sequential error correction; strong on tabular data.",
     "params": "n_estimators=150, learning_rate=0.08, max_depth=4, subsample=0.85"},
    {"name": "AdaBoost Classifier", "kind": "Ensemble - Boosting (adaptive)", "weighted": False,
     "definition": "Combines weak learners sequentially, increasing the weight of misclassified "
                   "customers so later learners concentrate on the hard cases.",
     "why": "Re-weights misclassified samples so later weak learners focus on hard cases.",
     "params": "n_estimators=200, learning_rate=0.5"},
]
ENSEMBLE_NAMES = {"Random Forest", "Extra Trees Classifier",
                  "Gradient Boosting Classifier", "AdaBoost Classifier"}
NO_CLASS_WEIGHT_KEYS = ("neighbors", "knn", "gradientboosting", "adaboost")

METRIC_DEFINITIONS = [
    ("Accuracy", "(TP + TN) / all predictions",
     "Share of all customers classified correctly. Can look good on imbalanced data even when "
     "many churners are missed, so it is secondary in this project."),
    ("Precision", "TP / (TP + FP)",
     "Of the customers predicted to churn, the share that really churned."),
    ("Recall", "TP / (TP + FN)",
     "Of the customers who really churned, the share the model found. The notebook prioritises recall "
     "because a missed churner (false negative) is the costlier error."),
    ("F1-Score", "2 x P x R / (P + R)",
     "Harmonic mean of precision and recall for the churn class."),
    ("ROC-AUC", "Area under the ROC curve",
     "Threshold-independent measure of how well predicted probabilities rank churners above "
     "non-churners (0.5 = random ranking, 1.0 = perfect). This is the project's primary ranking metric."),
]

# Notebook Q3 summary table - the "business impact" column (observed values are computed live)
INSIGHT_ACTIONS = {
    "class": "Accuracy alone can be misleading; also report ROC-AUC, precision, recall and F1-score.",
    "contract": "Test annual/biennial contract offers and communicate their benefits.",
    "tenure": "Improve onboarding and the first 90-day customer experience.",
    "charges": "Investigate price sensitivity and test targeted retention offers.",
    "payment": "Test whether payment-method guidance or auto-pay incentives improve retention.",
    "services": "Treat these as associations, not proof that the services cause lower churn.",
}

# Optional alias lists for reading model_results.csv without assuming column names
CANONICAL_COLUMNS = {
    "Model": ["model", "modelname", "classifier", "algorithm", "name"],
    "Type": ["type", "modeltype", "category"],
    "Accuracy": ["accuracy", "testaccuracy", "accuracyscore"],
    "Precision": ["precision", "testprecision"],
    "Recall": ["recall", "testrecall"],
    "F1-Score": ["f1score", "f1", "testf1"],
    "ROC-AUC": ["rocauc", "auc", "aucroc", "testrocauc", "testauc"],
}
METRICS = ["Accuracy", "Precision", "Recall", "F1-Score", "ROC-AUC"]

# =============================================================================
# 2. Styling
# =============================================================================

st.markdown(
    """
<style>
:root { --navy:#14243A; --blue:#2F6DB5; --text:#243247; --muted:#5E6E84;
        --line:#D7E1EE; --bg:#F4F7FB; }
[data-testid="stAppViewContainer"], [data-testid="stMain"] { background: var(--bg); }
html, [data-testid="stMain"] { scroll-behavior: smooth; }
[data-testid="stMain"] .block-container { max-width: 1280px; padding-top: 1.4rem; padding-bottom: 3rem; }
[data-testid="stMain"] p, [data-testid="stMain"] li,
[data-testid="stMain"] label, [data-testid="stMain"] td { color: var(--text); }
[data-testid="stMain"] h3, [data-testid="stMain"] h4, [data-testid="stMain"] h5 { color: var(--navy); }
[data-testid="stCaptionContainer"], [data-testid="stMain"] small { color: var(--muted) !important; }
[data-testid="stMetric"] { background:#fff; border:1px solid var(--line); border-radius:12px; padding:.8rem 1rem; }
[data-testid="stMetricLabel"] p { color: var(--muted) !important; }
[data-testid="stMetricValue"] { color: var(--navy) !important; }
[data-testid="stExpander"] { background:#fff; border:1px solid var(--line); border-radius:12px; }
.stTabs [data-baseweb="tab"] { font-weight: 600; }

[data-testid="stMain"] .hero { background: linear-gradient(120deg,#14243A 0%,#21507F 100%);
        border-radius:16px; padding:2rem 2.2rem; margin-bottom:1.4rem; }
[data-testid="stMain"] .hero h1 { color:#fff; font-size:2.1rem; margin:0 0 .5rem 0; padding:0; line-height:1.2; }
[data-testid="stMain"] .hero p { color:#DCE8F5; font-size:1rem; margin:0 0 1rem 0; max-width:850px; }
.hero .chip { display:inline-block; background:rgba(255,255,255,.14); color:#fff; border-radius:999px;
        padding:.22rem .8rem; margin:0 .4rem .4rem 0; font-size:.82rem; }

.anchor { position:relative; top:-70px; visibility:hidden; height:0; }
.section-head { display:flex; align-items:center; gap:.9rem; margin:2.2rem 0 .8rem 0;
        padding-bottom:.7rem; border-bottom:2px solid var(--line); }
.section-no { background:var(--navy); color:#fff; border-radius:10px; min-width:2.3rem; height:2.3rem;
        display:flex; align-items:center; justify-content:center; font-weight:700; }
.section-title { color:var(--navy); font-size:1.5rem; font-weight:700; line-height:1.2; }
.section-sub { color:var(--muted); font-size:.93rem; margin-top:.15rem; }
.back-link { margin-left:auto; font-size:.82rem; color:var(--blue) !important; text-decoration:none; white-space:nowrap; }

.card { background:#fff; border:1px solid var(--line); border-radius:14px; padding:1rem 1.15rem;
        margin-bottom:.8rem; height:calc(100% - .8rem); }
.card h4 { color:var(--navy); margin:0 0 .4rem 0; font-size:1.02rem; padding:0; }
.card p, .card li { color:var(--text); font-size:.92rem; line-height:1.55; margin:0 0 .35rem 0; }
.card .tag { display:inline-block; background:#E8F0FA; color:var(--blue); border-radius:6px;
        padding:.1rem .5rem; font-size:.74rem; font-weight:600; margin-bottom:.45rem; }
.card code { font-size:.78rem; }
.callout { background:#EAF1FA; border-left:4px solid var(--blue); border-radius:8px;
        padding:.8rem 1rem; color:var(--text); font-size:.93rem; margin:.4rem 0 .8rem 0; }

.flow { display:flex; flex-wrap:wrap; gap:.7rem; margin:.4rem 0 .6rem 0; }
.flow-step { flex:1 1 150px; position:relative; display:block; background:#fff; border:1px solid var(--line);
        border-top:4px solid var(--blue); border-radius:12px; padding:.85rem 1rem; text-decoration:none !important; }
.flow-step:hover { border-color: var(--blue); box-shadow:0 4px 14px rgba(20,36,58,.12); }
.flow-step .n { color:var(--blue); font-weight:700; font-size:.78rem; letter-spacing:.04em; }
.flow-step .t { color:var(--navy); font-weight:700; font-size:1rem; margin:.15rem 0; }
.flow-step .s { color:var(--muted); font-size:.8rem; line-height:1.35; }
.flow-step:not(:last-child)::after { content:"\\203A"; position:absolute; right:-.62rem; top:50%;
        transform:translateY(-50%); color:var(--blue); font-weight:700; font-size:1.2rem; }
@media (max-width: 760px) {
  .flow-step { flex:1 1 100%; }
  .flow-step::after { display:none; }
  [data-testid="stMain"] .hero { padding:1.4rem 1.2rem; }
  [data-testid="stMain"] .hero h1 { font-size:1.6rem; }
  .section-title { font-size:1.25rem; }
}
footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)

# =============================================================================
# 3. UI helpers
# =============================================================================


def _version() -> tuple:
    try:
        return tuple(int(p) for p in re.findall(r"\d+", st.__version__)[:2])
    except Exception:
        return (0, 0)


NEW_WIDTH_API = _version() >= (1, 51)  # width="stretch" replaced use_container_width


def stretch(fn, *args, **kwargs):
    """Call a Streamlit element so it fills the container on old and new versions."""
    if NEW_WIDTH_API:
        try:
            return fn(*args, width="stretch", **kwargs)
        except TypeError:
            pass
    return fn(*args, use_container_width=True, **kwargs)


_chart_ids = itertools.count()


def html_block(markup: str) -> None:
    """Render raw HTML; leading spaces are stripped so Markdown never treats it as code."""
    cleaned = "\n".join(line.strip() for line in markup.strip().splitlines() if line.strip())
    st.markdown(cleaned, unsafe_allow_html=True)


def section(anchor_id: str, letter: str, title: str, subtitle: str) -> None:
    html_block(
        f"""
        <div id="{anchor_id}" class="anchor"></div>
        <div class="section-head">
          <div class="section-no">{letter}</div>
          <div><div class="section-title">{title}</div><div class="section-sub">{subtitle}</div></div>
          <a class="back-link" href="#sec-workflow" target="_self">&uarr; Back to workflow</a>
        </div>
        """
    )


def card(title: str, body: str, tag: str | None = None) -> None:
    tag_html = f'<span class="tag">{tag}</span><br>' if tag else ""
    html_block(f'<div class="card">{tag_html}<h4>{title}</h4>{body}</div>')


def callout(text: str) -> None:
    html_block(f'<div class="callout">{text}</div>')


def show_table(data, **kwargs) -> None:
    stretch(st.dataframe, data, hide_index=kwargs.pop("hide_index", True), **kwargs)


def kpis(items: list[tuple[str, str]], per_row: int = 3) -> None:
    for start in range(0, len(items), per_row):
        chunk = items[start:start + per_row]
        for col, (label, value) in zip(st.columns(len(chunk)), chunk):
            col.metric(label, value)


def fmt_pct(value: float) -> str:
    return f"{value:.1f}%"


# =============================================================================
# 4. Chart helpers (Plotly when available, Matplotlib otherwise)
# =============================================================================


def _plotly_layout(fig, title: str, height: int = CHART_H, legend_title: str | None = None):
    fig.update_layout(
        template="plotly_white",
        height=height,
        title=dict(text=title, x=0.0, font=dict(size=15, color=NAVY)),
        margin=dict(l=10, r=10, t=55, b=10),
        font=dict(family="Inter, Segoe UI, Arial, sans-serif", size=12, color="#243247"),
        legend_title_text=legend_title,
    )
    return fig


def _show(fig) -> None:
    """Display a Plotly or Matplotlib figure."""
    if PLOTLY and not isinstance(fig, Figure if MPL else ()):
        stretch(st.plotly_chart, fig, key=f"pd_chart_{next(_chart_ids)}")
    else:
        st.pyplot(fig)


def _no_chart_library() -> bool:
    if not PLOTLY and not MPL:
        st.warning("Install plotly (preferred) or matplotlib to display charts: pip install plotly")
        return True
    return False


def bar_chart(data, x, y, title, x_title, y_title, color=None, color_map=None,
              fmt=".1f", y_range=None, height=CHART_H, legend_title=None):
    """Vertical bar chart; grouped when `color` is given."""
    if data is None or data.empty:
        st.info("No data available for this chart.")
        return
    if _no_chart_library():
        return
    if PLOTLY:
        fig = px.bar(
            data, x=x, y=y, color=color, barmode="group",
            color_discrete_map=color_map,
            color_discrete_sequence=CATEGORY_PALETTE if color_map is None else None,
            text_auto=fmt, labels={x: x_title, y: y_title},
        )
        if color is None:
            fig.update_traces(marker_color=BLUE)
        fig.update_traces(textposition="outside", cliponaxis=False)
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title=y_title, range=y_range)
        _plotly_layout(fig, title, height, legend_title)
        _show(fig)
        return
    fig = Figure(figsize=(7.5, height / 100))
    ax = fig.subplots()
    cats = data[x].astype(str).drop_duplicates().tolist()
    if color:
        pivot = data.assign(**{x: data[x].astype(str)}).pivot_table(
            index=x, columns=color, values=y, aggfunc="first").reindex(cats)
        groups = list(pivot.columns)
    else:
        pivot = pd.DataFrame({y: data.set_index(data[x].astype(str))[y]}).reindex(cats)
        groups = [y]
    width = 0.8 / len(groups)
    for i, group in enumerate(groups):
        positions = np.arange(len(cats)) + i * width - 0.4 + width / 2
        shade = (color_map or {}).get(group, CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]) if color else BLUE
        container = ax.bar(positions, pivot[group].values, width, label=str(group), color=shade)
        ax.bar_label(container, labels=["" if pd.isna(v) else format(v, fmt) for v in pivot[group].values],
                     fontsize=8, padding=2)
    ax.set_xticks(np.arange(len(cats)))
    ax.set_xticklabels(cats, rotation=20, ha="right")
    ax.set_title(title, loc="left", fontsize=12, color=NAVY)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    if y_range:
        ax.set_ylim(*y_range)
    if color:
        ax.legend(title=legend_title, fontsize=8)
    fig.tight_layout()
    _show(fig)


def hbar_chart(data, x, y, title, x_title, y_title, height=420):
    if data is None or data.empty:
        st.info("No data available for this chart.")
        return
    if _no_chart_library():
        return
    ordered = data.iloc[::-1]
    if PLOTLY:
        fig = px.bar(ordered, x=x, y=y, orientation="h", text_auto=".3f")
        fig.update_traces(marker_color=BLUE, textposition="outside", cliponaxis=False)
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title=y_title)
        _plotly_layout(fig, title, height)
        _show(fig)
        return
    fig = Figure(figsize=(7.5, height / 100))
    ax = fig.subplots()
    ax.barh(ordered[y].astype(str), ordered[x], color=BLUE)
    ax.set_title(title, loc="left", fontsize=12, color=NAVY)
    ax.set_xlabel(x_title)
    ax.set_ylabel(y_title)
    fig.tight_layout()
    _show(fig)


def box_chart(frame, column, title, y_title):
    plot = frame[["ChurnLabel", column]].dropna()
    if plot.empty:
        st.info(f"No numeric values available for {column}.")
        return
    if _no_chart_library():
        return
    if PLOTLY:
        fig = px.box(plot, x="ChurnLabel", y=column, color="ChurnLabel", points=False,
                     color_discrete_map=COLOR_MAP, category_orders={"ChurnLabel": LABEL_ORDER})
        fig.update_xaxes(title="Churn status")
        fig.update_yaxes(title=y_title)
        _plotly_layout(fig, title, 340)
        fig.update_layout(showlegend=False)
        _show(fig)
        return
    fig = Figure(figsize=(4.5, 3.6))
    ax = fig.subplots()
    groups = [plot.loc[plot["ChurnLabel"] == label, column] for label in LABEL_ORDER]
    ax.boxplot(groups)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(LABEL_ORDER)
    ax.set_title(title, loc="left", fontsize=11, color=NAVY)
    ax.set_xlabel("Churn status")
    ax.set_ylabel(y_title)
    fig.tight_layout()
    _show(fig)


def hist_chart(frame, column, title, x_title):
    plot = frame[["ChurnLabel", column]].dropna()
    if plot.empty:
        st.info(f"No numeric values available for {column}.")
        return
    if _no_chart_library():
        return
    if PLOTLY:
        fig = px.histogram(plot, x=column, color="ChurnLabel", nbins=30, barmode="overlay", opacity=0.65,
                           color_discrete_map=COLOR_MAP, category_orders={"ChurnLabel": LABEL_ORDER})
        fig.update_xaxes(title=x_title)
        fig.update_yaxes(title="Number of customers")
        _plotly_layout(fig, title, 340, "Churn status")
        _show(fig)
        return
    fig = Figure(figsize=(5.5, 3.6))
    ax = fig.subplots()
    for label in LABEL_ORDER:
        ax.hist(plot.loc[plot["ChurnLabel"] == label, column], bins=30, alpha=0.65,
                color=COLOR_MAP[label], label=label, edgecolor="white")
    ax.set_title(title, loc="left", fontsize=11, color=NAVY)
    ax.set_xlabel(x_title)
    ax.set_ylabel("Number of customers")
    ax.legend(title="Churn status", fontsize=8)
    fig.tight_layout()
    _show(fig)


def heatmap_chart(matrix: pd.DataFrame, title, scale, zmin, zmax, fmt, height=CHART_H,
                  x_title=None, y_title=None, show_scale=True):
    if _no_chart_library():
        return
    if PLOTLY:
        fig = px.imshow(matrix, text_auto=fmt, color_continuous_scale=scale, zmin=zmin, zmax=zmax,
                        aspect="auto", labels=dict(x=x_title or "", y=y_title or ""))
        if not show_scale:
            fig.update_coloraxes(showscale=False)
        _plotly_layout(fig, title, height)
        _show(fig)
        return
    fig = Figure(figsize=(5.5, height / 100))
    ax = fig.subplots()
    image = ax.imshow(matrix.values, cmap="RdBu_r" if scale == "RdBu_r" else "Blues", vmin=zmin, vmax=zmax)
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_xticklabels(matrix.columns, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(matrix.shape[0]))
    ax.set_yticklabels(matrix.index, fontsize=8)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, format(matrix.values[i, j], "d" if fmt == "d" else ".2f"),
                    ha="center", va="center", fontsize=8)
    ax.set_title(title, loc="left", fontsize=11, color=NAVY)
    if x_title:
        ax.set_xlabel(x_title)
    if y_title:
        ax.set_ylabel(y_title)
    if show_scale:
        fig.colorbar(image, ax=ax)
    fig.tight_layout()
    _show(fig)


def roc_chart(curves: dict, title: str):
    if _no_chart_library():
        return
    if PLOTLY:
        fig = go.Figure()
        for name, (fpr, tpr, auc) in curves.items():
            fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{name} (AUC = {auc:.4f})"))
        fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode="lines", name="Random classifier",
                                 line=dict(dash="dash", color="#7A8CA5")))
        fig.update_xaxes(title="False positive rate")
        fig.update_yaxes(title="True positive rate (recall)")
        _plotly_layout(fig, title, 460)
        fig.update_layout(legend=dict(orientation="h", y=-0.2))
        _show(fig)
        return
    fig = Figure(figsize=(7, 5))
    ax = fig.subplots()
    for name, (fpr, tpr, auc) in curves.items():
        ax.plot(fpr, tpr, label=f"{name} (AUC = {auc:.4f})")
    ax.plot([0, 1], [0, 1], "--", color="#7A8CA5", label="Random classifier")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (recall)")
    ax.set_title(title, loc="left", fontsize=12, color=NAVY)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    _show(fig)


# =============================================================================
# 5. Data loading
# =============================================================================


def find_dataset() -> Path | None:
    names = ["WA_Fn-UseC_-Telco-Customer-Churn.csv", "WA_Fn-UseC_-Telco-Customer-Churn(8).csv"]
    for folder in (DATA_DIR, ROOT):
        for name in names:
            if (folder / name).exists():
                return folder / name
    for folder in (DATA_DIR, ROOT):
        if folder.exists():
            hits = sorted(folder.glob("*[Tt]elco*[Cc]hurn*.csv"))
            if hits:
                return hits[0]
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str, modified: float) -> pd.DataFrame:
    """Read a CSV; the modified time busts the cache when the file changes."""
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    return frame


dataset_path = find_dataset()
df, data_error = None, None
if dataset_path is None:
    data_error = ("Dataset not found. Place the Telco churn CSV "
                  "(WA_Fn-UseC_-Telco-Customer-Churn.csv) in the project's data/ folder.")
else:
    try:
        df = load_csv(str(dataset_path), dataset_path.stat().st_mtime)
        if "Churn" not in df.columns:
            data_error, df = "The dataset has no 'Churn' target column.", None
        elif not set(df["Churn"].dropna().unique()) <= set(LABELS):
            data_error, df = "The 'Churn' column does not contain only Yes/No values.", None
    except Exception as exc:  # unreadable or malformed file
        data_error, df = f"Could not read the dataset: {exc}", None


def has(*columns: str) -> bool:
    return df is not None and all(c in df.columns for c in columns)


# Analysis frame: TotalCharges as numeric (blank strings -> NaN) and readable churn labels.
dfa = None
if df is not None:
    dfa = df.copy()
    if "TotalCharges" in dfa.columns:
        dfa["TotalCharges"] = pd.to_numeric(dfa["TotalCharges"].astype(str).str.strip(), errors="coerce")
    dfa["ChurnBin"] = (dfa["Churn"] == "Yes").astype(int)
    dfa["ChurnLabel"] = dfa["Churn"].map(LABELS)


def normalise(name: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(name).lower())


@st.cache_data(show_spinner=False)
def load_results(path: str, modified: float) -> pd.DataFrame:
    frame = pd.read_csv(path)
    frame.columns = frame.columns.astype(str).str.strip()
    return frame.loc[:, ~frame.columns.str.startswith("Unnamed")]


results_raw, results_error = None, None
if RESULTS_PATH.exists():
    try:
        results_raw = load_results(str(RESULTS_PATH), RESULTS_PATH.stat().st_mtime)
        if results_raw.empty:
            results_raw, results_error = None, "data/model_results.csv is empty."
    except Exception as exc:
        results_error = f"Could not read data/model_results.csv: {exc}"


def standardise_results(raw: pd.DataFrame | None):
    """Map whatever column names the CSV has onto canonical names; returns (frame, mapping)."""
    if raw is None:
        return None, {}
    lookup = {normalise(c): c for c in raw.columns}
    mapping = {}
    for canonical, aliases in CANONICAL_COLUMNS.items():
        for alias in aliases:
            if alias in lookup and lookup[alias] not in mapping.values():
                mapping[canonical] = lookup[alias]
                break
    if "Model" not in mapping:
        return None, mapping
    std = pd.DataFrame({"Model": raw[mapping["Model"]].astype(str)})
    if "Type" in mapping:
        std["Type"] = raw[mapping["Type"]].astype(str)
    for metric in METRICS:
        if metric in mapping:
            std[metric] = pd.to_numeric(raw[mapping[metric]], errors="coerce")
    return std, mapping


saved_std, saved_map = standardise_results(results_raw)

# =============================================================================
# 6. Cached computations
# =============================================================================


@st.cache_data(show_spinner=False)
def split_summary(churn_flags: tuple) -> dict | None:
    """Reproduce the notebook's stratified 80/20 split (random_state=42) to report its sizes."""
    try:
        from sklearn.model_selection import train_test_split
    except ImportError:
        return None
    y = pd.Series(churn_flags)
    idx_train, idx_test = train_test_split(np.arange(len(y)), test_size=0.20,
                                           random_state=RANDOM_STATE, stratify=y)
    return {"train": len(idx_train), "test": len(idx_test),
            "train_rate": float(y.iloc[idx_train].mean() * 100),
            "test_rate": float(y.iloc[idx_test].mean() * 100)}


@st.cache_resource(show_spinner=False)
def run_notebook_evaluation(path: str, modified: float) -> dict:
    """
    Re-run the notebook's Q4-Q6 pipeline once (cached) to obtain ROC curves, confusion
    matrices, the classification report and feature importances that are not stored in
    model_results.csv. Settings are copied from the notebook.
    """
    try:
        import sklearn
        from sklearn.compose import ColumnTransformer
        from sklearn.ensemble import (AdaBoostClassifier, ExtraTreesClassifier,
                                      GradientBoostingClassifier, RandomForestClassifier)
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                                     f1_score, precision_score, recall_score, roc_auc_score, roc_curve)
        from sklearn.model_selection import train_test_split
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import OneHotEncoder, StandardScaler
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier
    except ImportError as exc:
        return {"error": f"scikit-learn is required to regenerate the evaluation plots ({exc})."}
    try:
        data = pd.read_csv(path)
        data.columns = data.columns.astype(str).str.strip()
        data["TotalCharges"] = pd.to_numeric(data["TotalCharges"].astype(str).str.strip(), errors="coerce")
        data = data.drop(columns=["customerID"])
        data["Churn"] = data["Churn"].map({"Yes": 1, "No": 0})
        X, y = data.drop(columns=["Churn"]), data["Churn"]
        categorical = [c for c in X.columns if c not in NUMERICAL_COLS]
        preprocessor = ColumnTransformer([
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median")),
                              ("scaler", StandardScaler())]), NUMERICAL_COLS),
            ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                              ("onehot", OneHotEncoder(drop="first", sparse_output=False,
                                                       handle_unknown="ignore"))]), categorical),
        ])
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=RANDOM_STATE, stratify=y)
        rs = RANDOM_STATE
        models = {
            "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0, random_state=rs),
            "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=9, metric="euclidean", weights="distance"),
            "Support Vector Machine (SVM)": SVC(kernel="rbf", C=1.0, gamma="scale", probability=True,
                                                class_weight="balanced", random_state=rs),
            "Decision Tree": DecisionTreeClassifier(max_depth=6, min_samples_leaf=20,
                                                    class_weight="balanced", random_state=rs),
            "Random Forest": RandomForestClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                                    class_weight="balanced", n_jobs=-1, random_state=rs),
            "Extra Trees Classifier": ExtraTreesClassifier(n_estimators=200, max_depth=8, min_samples_leaf=10,
                                                           class_weight="balanced", n_jobs=-1, random_state=rs),
            "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=150, learning_rate=0.08,
                                                                       max_depth=4, subsample=0.85,
                                                                       random_state=rs),
            "AdaBoost Classifier": AdaBoostClassifier(n_estimators=200, learning_rate=0.5, random_state=rs),
        }
        rows, matrices, curves, fitted = [], {}, {}, {}
        for name, model in models.items():
            pipe = Pipeline([("preprocessor", preprocessor), ("classifier", model)]).fit(X_train, y_train)
            pred = pipe.predict(X_test)
            proba = pipe.predict_proba(X_test)[:, 1]
            auc = roc_auc_score(y_test, proba)
            fpr, tpr, _ = roc_curve(y_test, proba)
            rows.append({
                "Model": name, "Type": "Ensemble" if name in ENSEMBLE_NAMES else "Traditional",
                "Accuracy": round(accuracy_score(y_test, pred), 4),
                "Precision": round(precision_score(y_test, pred, zero_division=0), 4),
                "Recall": round(recall_score(y_test, pred, zero_division=0), 4),
                "F1-Score": round(f1_score(y_test, pred, zero_division=0), 4),
                "ROC-AUC": round(auc, 4)})
            matrices[name] = confusion_matrix(y_test, pred)
            curves[name] = (fpr, tpr, auc)
            fitted[name] = (pipe, pred)
        results = pd.DataFrame(rows).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
        best = results.iloc[0]["Model"]
        report = classification_report(y_test, fitted[best][1], target_names=["Retained (0)", "Churned (1)"],
                                       output_dict=True, zero_division=0)
        report_df = pd.DataFrame({k: v for k, v in report.items() if isinstance(v, dict)}).T
        importance = None
        ensembles = results[results["Type"] == "Ensemble"]
        best_ensemble = ensembles.iloc[0]["Model"]
        pipe = fitted[best_ensemble][0]
        classifier = pipe.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            ohe = pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["onehot"]
            names = NUMERICAL_COLS + list(ohe.get_feature_names_out(categorical))
            importance = (pd.DataFrame({"Feature": names, "Importance": classifier.feature_importances_})
                          .sort_values("Importance", ascending=False).head(15).reset_index(drop=True))
        return {"results": results, "matrices": matrices, "curves": curves, "best": best,
                "report": report_df, "report_accuracy": report["accuracy"], "importance": importance,
                "best_ensemble": best_ensemble, "n_train": len(X_train), "n_test": len(X_test),
                "n_encoded": len(NUMERICAL_COLS) + sum(
                    len(c) - 1 for c in preprocessor.named_transformers_["cat"].named_steps["onehot"].categories_),
                "sklearn": sklearn.__version__}
    except Exception as exc:
        return {"error": f"The evaluation could not be regenerated: {exc}"}


def churn_rate_by(frame: pd.DataFrame, column: str) -> pd.DataFrame:
    table = frame.groupby(column, dropna=False)["ChurnBin"].agg(["mean", "size"]).reset_index()
    table.columns = [column, "Churn rate (%)", "Customers"]
    table["Churn rate (%)"] = table["Churn rate (%)"] * 100
    table[column] = table[column].astype(str)
    return table


def rates_text(frame: pd.DataFrame, column: str) -> str:
    table = churn_rate_by(frame, column).sort_values("Churn rate (%)", ascending=False)
    return ", ".join(f"{row[column]}: {row['Churn rate (%)']:.1f}%" for _, row in table.iterrows())


# =============================================================================
# 7. Hero and sidebar
# =============================================================================

n_rows = len(df) if df is not None else None
chips = ["Supervised binary classification", "8 classifiers compared",
         "Primary metric: ROC-AUC", "Stratified 80/20 hold-out"]
if n_rows:
    chips.insert(0, f"{n_rows:,} customers")
chip_html = "".join(f'<span class="chip">{c}</span>' for c in chips)

html_block(
    f"""
    <div class="hero">
      <h1>Telco Customer Churn Prediction</h1>
      <p>Project report documenting the dataset, exploratory analysis, preprocessing pipeline,
      model training and evaluation, and the conclusion drawn in <b>{NOTEBOOK_NAME}</b>.</p>
      {chip_html}
    </div>
    """
)

with st.sidebar:
    st.markdown("**On this page**")
    html_block(
        """
        <div style="line-height:2">
        <a href="#sec-overview" target="_self">A &middot; Project Overview</a><br>
        <a href="#sec-dataset" target="_self">B &middot; Dataset &amp; Metadata</a><br>
        <a href="#sec-eda" target="_self">C &middot; Exploratory Analysis</a><br>
        <a href="#sec-preprocessing" target="_self">D &middot; Preprocessing</a><br>
        <a href="#sec-training" target="_self">E &middot; Model Training</a><br>
        <a href="#sec-evaluation" target="_self">F &middot; Evaluation &amp; Results</a><br>
        <a href="#sec-conclusion" target="_self">G &middot; Conclusion</a>
        </div>
        """
    )

# =============================================================================
# A. Project overview
# =============================================================================

section("sec-overview", "A", "Project Overview",
        "Problem statement, objectives, the definition of churn and the machine-learning workflow.")

left, right = st.columns([1.25, 1], gap="large")
with left:
    st.markdown("#### Problem statement")
    st.markdown(
        "A telecommunications company is experiencing revenue loss due to customer attrition (churn). "
        "Acquiring a new customer is stated in the notebook to cost 5-7x more than retaining an existing one "
        "(a business assumption, not derived from this dataset). The company needs to proactively identify "
        "at-risk customers so retention teams can act before the customer leaves."
    )
    callout(
        "<b>Task:</b> build a supervised binary classification model that predicts, given a customer's "
        "demographic profile, account history and subscribed services, whether that customer will churn "
        "(leave) &mdash; with high Recall and ROC-AUC to minimise missed churners."
    )
    st.markdown("#### Objectives")
    st.markdown(
        "1. **Primary:** train and evaluate 8 classification models and select the best using ROC-AUC and "
        "F1-Score (not raw Accuracy, because of class imbalance).\n"
        "2. **Secondary:** identify the features most associated with churn (e.g. contract type, tenure, "
        "monthly charges) using EDA and feature importance.\n"
        "3. **Business constraint:** prioritise Recall over Precision. A missed churner (false negative) causes "
        "permanent revenue loss, whereas a wrongly flagged customer (false positive) only wastes a low-cost "
        "retention offer."
    )
with right:
    card("What is customer churn?",
         "<p>Customer churn means a customer leaves the company's service. In this dataset the "
         "<code>Churn</code> column records <i>whether the customer left in the last month</i> "
         "(Yes = 1, No = 0).</p>", tag="Definition")
    card("Evaluation priority (from the notebook)",
         "<ol><li><b>ROC-AUC</b> - overall discrimination power</li>"
         "<li><b>F1-Score</b> - churn class</li><li><b>Recall</b> - catching actual churners</li>"
         "<li><b>Accuracy</b> - secondary, misleading alone under imbalance</li></ol>", tag="Selection rule")

html_block('<div id="sec-workflow" class="anchor"></div>')
st.markdown("#### Machine learning workflow")
st.caption("Click a step to jump to its section. Notebook references show where each step is implemented.")

workflow = [
    ("sec-dataset", "Dataset &amp; Metadata", "Notebook Q1 &middot; inspection, feature dictionary, data types"),
    ("sec-eda", "EDA", "Notebook Q3 &middot; target, numerical, categorical, service and correlation plots"),
    ("sec-preprocessing", "Preprocessing", "Notebook Q4 &middot; cleaning, imputation, scaling, encoding, split"),
    ("sec-training", "Model Training", "Notebook Q5 &middot; 8 classifiers (4 traditional, 4 ensemble)"),
    ("sec-evaluation", "Evaluation", "Notebook Q6 &middot; metrics, ROC, confusion matrices, ranking"),
    ("sec-conclusion", "Conclusion", "Notebook Answer 6 &middot; best model and limitations"),
]
html_block(
    '<div class="flow">'
    + "".join(
        f'<a class="flow-step" href="#{target}" target="_self"><div class="n">STEP {i}</div>'
        f'<div class="t">{label}</div><div class="s">{note}</div></a>'
        for i, (target, label, note) in enumerate(workflow, start=1)
    )
    + "</div>"
)

with st.expander("Which workflow stages exist in the notebook?"):
    show_table(pd.DataFrame([
        ("Dataset", "Q1", "Yes - loaded with pd.read_csv, metadata table and feature dictionary"),
        ("Data inspection", "Q1, Q3 (EDA cell)", "Yes - df.info(), describe(), head(), value counts"),
        ("EDA", "Q3 (3.1-3.6)", "Yes - six groups of plots and an insight summary"),
        ("Preprocessing", "Q4 steps 1-5", "Yes - cleaning, target encoding, ColumnTransformer"),
        ("Train/test split", "Q4 step 6", "Yes - stratified 80/20, random_state = 42"),
        ("Model training", "Q5", "Yes - 8 classifiers, each inside a Pipeline"),
        ("Evaluation", "Q6 (6.1-6.6)", "Yes - leaderboard, bar chart, ROC, confusion matrices, report, importances"),
        ("Best model", "Q6 / Answer 6", "Yes - ranked by ROC-AUC"),
        ("Deployment", "-", "Not in the notebook. The Streamlit app (app.py, pages/churn_prediction.py) is separate."),
    ], columns=["Stage", "Notebook section", "Implemented in the notebook?"]))

# =============================================================================
# B. Dataset and metadata
# =============================================================================

section("sec-dataset", "B", "Dataset & Metadata",
        "Dimensions, target, feature dictionary and data quality (notebook Q1).")

if df is None:
    st.warning(data_error or "Dataset information is unavailable.")
else:
    n_cols = df.shape[1]
    churn_pct = float(dfa["ChurnBin"].mean() * 100)
    blank_total = int((dfa["TotalCharges"].isna()).sum()) if has("TotalCharges") else 0
    dup_rows = int(df.duplicated().sum())
    model_features = n_cols - (1 if has("customerID") else 0) - 1

    kpis([
        ("Customer records", f"{n_rows:,}"),
        ("Columns", f"{n_cols}"),
        ("Features used for modelling", f"{model_features}"),
        ("Churn rate", fmt_pct(churn_pct)),
        ("Blank TotalCharges", f"{blank_total}"),
        ("Duplicate rows", f"{dup_rows}"),
    ], per_row=3)
    st.caption(
        f"Columns = customerID + {model_features} features + the Churn target. "
        f"Source file: {dataset_path.name}")

    st.markdown("#### Metadata summary")
    cat_features = [c for c in df.columns if c not in NUMERICAL_COLS + ["customerID", "Churn"]]
    counts = df["Churn"].value_counts()
    show_table(pd.DataFrame([
        ("Dataset name", "Telco Customer Churn (IBM Watson Analytics sample)"),
        ("Source (per notebook)", "Kaggle / IBM sample data"),
        ("Records", f"{n_rows:,} customers"),
        ("Target variable", "Churn - binary: Yes (1) / No (0)"),
        ("Numerical features used", ", ".join(c for c in NUMERICAL_COLS if c in df.columns)),
        ("Categorical features used", f"{len(cat_features)} columns (listed in the feature dictionary)"),
        ("Known data issue", "TotalCharges is stored as text and contains blank entries"),
        ("Class distribution", f"No Churn {counts.get('No', 0):,} ({counts.get('No', 0) / n_rows * 100:.1f}%) | "
                               f"Churn {counts.get('Yes', 0):,} ({counts.get('Yes', 0) / n_rows * 100:.1f}%) "
                               "- imbalanced"),
    ], columns=["Property", "Detail"]))

    st.markdown("#### Dataset preview")
    show_table(df.head(10))

    st.markdown("#### Data quality")
    nan_cells = int(df.isna().sum().sum())
    observations = [f"**Missing values:** {nan_cells} cells are recognised as missing by pandas."]
    if has("TotalCharges"):
        raw_blank = df["TotalCharges"].astype(str).str.strip().eq("")
        tenure_of_blank = sorted(df.loc[raw_blank, "tenure"].unique().tolist()) if has("tenure") else []
        observations.append(
            f"**Hidden missing values:** TotalCharges has {int(raw_blank.sum())} blank strings. "
            + (f"All of them belong to customers with tenure = {tenure_of_blank[0]}."
               if len(tenure_of_blank) == 1 and int(raw_blank.sum()) else
               "Their tenure values: " + ", ".join(map(str, tenure_of_blank)) + "." if tenure_of_blank else "")
            + " Pandas loads the column as text because of these blanks.")
    observations.append(f"**Duplicates:** {dup_rows} fully duplicated rows"
                        + (f"; customerID is unique for {df['customerID'].nunique():,} of {n_rows:,} rows."
                           if has("customerID") else "."))
    observations.append(f"**Imbalance:** No Churn : Churn is about "
                        f"{counts.get('No', 0) / max(counts.get('Yes', 1), 1):.2f} : 1, which is why accuracy "
                        "alone is not a sufficient metric.")
    if has("SeniorCitizen"):
        observations.append("**Encoding of SeniorCitizen:** stored as 0/1 integers, unlike the other binary "
                            "columns that use Yes/No text (the notebook treats it as numerical).")
    service_cols = [c for c in df.columns if df[c].astype(str).isin(["No internet service"]).any()]
    if service_cols:
        observations.append(f"**Service categories:** {len(service_cols)} columns contain the value "
                            "'No internet service'; it is a legitimate category, not a missing value.")
    for line in observations:
        st.markdown(f"- {line}")

    with st.expander("Feature dictionary (from the notebook, with values read from the loaded file)"):
        rows = []
        for feature, nb_type, category, description in FEATURE_DICTIONARY:
            present = feature in df.columns
            rows.append({
                "Feature": feature, "Category": category, "Description": description,
                "Notebook dtype": nb_type,
                "Loaded dtype": str(df[feature].dtype) if present else "not in file",
                "Unique values": int(df[feature].nunique()) if present else None,
            })
        show_table(pd.DataFrame(rows))
        extra = [c for c in df.columns if c not in [f[0] for f in FEATURE_DICTIONARY]]
        if extra:
            st.info("Columns in the file but not in the notebook's dictionary: " + ", ".join(extra))

    with st.expander("Descriptive statistics (numerical features)"):
        numeric_cols = [c for c in NUMERICAL_COLS if c in dfa.columns]
        show_table(dfa[numeric_cols].describe().T.reset_index().rename(columns={"index": "Feature"}))
        st.caption("TotalCharges is converted to numbers first; its blank entries are excluded from the count.")

    with st.expander("Category counts for a categorical feature"):
        cat_choice = st.selectbox("Feature", cat_features, key="b_cat")
        counts_df = df[cat_choice].astype(str).value_counts().rename_axis(cat_choice).reset_index(name="Customers")
        bar_chart(counts_df, cat_choice, "Customers", f"Customers by {cat_choice}", cat_choice,
                  "Number of customers", fmt=",.0f", height=320)

# =============================================================================
# C. Exploratory data analysis
# =============================================================================

section("sec-eda", "C", "Exploratory Data Analysis",
        "Visual analysis reproduced from notebook Q3. Findings are associations, not causes.")

if df is None:
    st.info("Load the dataset to display the exploratory analysis.")
else:
    tabs = st.tabs(["Target", "Numerical features", "Categorical features",
                    "Service add-ons", "Correlation", "Key insights"])

    # --- 3.1 target distribution -------------------------------------------
    with tabs[0]:
        target_df = (dfa["ChurnLabel"].value_counts().reindex(LABEL_ORDER).rename_axis("Churn status")
                     .reset_index(name="Customers"))
        target_df["Share (%)"] = target_df["Customers"] / target_df["Customers"].sum() * 100
        target_df["Label"] = target_df.apply(
            lambda r: f"{int(r['Customers']):,} ({r['Share (%)']:.1f}%)", axis=1)
        bar_chart(target_df, "Churn status", "Customers", "Target distribution: Churn vs No Churn",
                  "Churn status", "Number of customers", color="Churn status", color_map=COLOR_MAP,
                  fmt=",.0f", legend_title="Churn status")
        st.caption(
            f"Measures how many customers fall in each class. {target_df.loc[1, 'Share (%)']:.1f}% of customers "
            f"churned ({int(target_df.loc[1, 'Customers']):,} of {n_rows:,}), so the classes are imbalanced. "
            "ROC-AUC, recall and F1-score are therefore used alongside accuracy.")

    # --- 3.2 / 3.3 numerical features ---------------------------------------
    with tabs[1]:
        st.markdown("**Box plots by churn status** (notebook 3.2)")
        box_cols = [c for c in ["tenure", "MonthlyCharges", "TotalCharges"] if c in dfa.columns]
        titles = {"tenure": ("Tenure vs Churn", "Tenure (months)"),
                  "MonthlyCharges": ("Monthly Charges vs Churn", "Monthly charges (USD)"),
                  "TotalCharges": ("Total Charges vs Churn", "Total charges (USD)")}
        for col_ui, feature in zip(st.columns(len(box_cols)), box_cols):
            with col_ui:
                box_chart(dfa, feature, titles[feature][0], titles[feature][1])
        med = dfa.groupby("ChurnLabel")[box_cols].median()
        if set(LABEL_ORDER) <= set(med.index):
            parts = [f"median {f}: {med.loc['Churn', f]:,.1f} (churn) vs {med.loc['No Churn', f]:,.1f} (no churn)"
                     for f in box_cols]
            st.caption("Each box shows the median and spread of a feature within each class. In this dataset: "
                       + "; ".join(parts) + ". These are differences between groups, not evidence of causation.")

        st.markdown("**Distributions by churn status** (notebook 3.3)")
        hist_cols = [c for c in ["tenure", "MonthlyCharges"] if c in dfa.columns]
        hist_titles = {"tenure": ("Tenure distribution by churn status", "Tenure (months)"),
                       "MonthlyCharges": ("Monthly charges distribution by churn status", "Monthly charges (USD)")}
        for col_ui, feature in zip(st.columns(len(hist_cols)), hist_cols):
            with col_ui:
                hist_chart(dfa, feature, hist_titles[feature][0], hist_titles[feature][1])
        st.caption("Histograms count customers in 30 equal-width bins per class. The classes have different sizes, "
                   "so compare the shapes and where each class is concentrated, not the bar heights alone.")

    # --- 3.4 categorical features -------------------------------------------
    with tabs[2]:
        cat_plot = [c for c in ["Contract", "PaymentMethod", "InternetService", "gender"] if c in dfa.columns]
        st.markdown("**Churn rate by category** (notebook 3.4)")
        for pair in range(0, len(cat_plot), 2):
            for col_ui, feature in zip(st.columns(2), cat_plot[pair:pair + 2]):
                with col_ui:
                    table = churn_rate_by(dfa, feature)
                    bar_chart(table, feature, "Churn rate (%)", f"Churn rate by {feature}", feature,
                              "Customers who churned (%)", y_range=[0, max(60, table["Churn rate (%)"].max() + 8)],
                              height=320)
        notes = [f"{f}: {rates_text(dfa, f)}" for f in cat_plot]
        st.caption("Each bar is the share of customers in that category who churned (churn rate within the group), "
                   "which is comparable across groups of different sizes. Observed: " + " | ".join(notes) + ".")

    # --- 3.5 service add-ons -------------------------------------------------
    with tabs[3]:
        services = [c for c in ["PhoneService", "MultipleLines", "OnlineSecurity", "OnlineBackup",
                                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"]
                    if c in dfa.columns]
        long_rows = []
        for service in services:
            for _, r in churn_rate_by(dfa, service).iterrows():
                long_rows.append({"Service": service, "Category": r[service],
                                  "Churn rate (%)": r["Churn rate (%)"], "Customers": int(r["Customers"])})
        service_df = pd.DataFrame(long_rows)
        st.markdown("**Churn rate within each service category** (notebook 3.5)")
        bar_chart(service_df, "Service", "Churn rate (%)", "Churn rate within each service category",
                  "Service", "Customers who churned (%)", color="Category", fmt=".1f", height=420,
                  y_range=[0, 60], legend_title="Category")
        st.caption("Each bar is the churn rate among customers in that category. It is not the share of all "
                   "churners who use the service, and it does not show that a service causes lower or higher churn.")
        with st.expander("Values behind this chart"):
            show_table(service_df)

    # --- 3.6 correlation ------------------------------------------------------
    with tabs[4]:
        corr_cols = [c for c in NUMERICAL_COLS if c in dfa.columns]
        corr_frame = dfa[corr_cols].copy()
        corr_frame["Churn (1 = Yes)"] = dfa["ChurnBin"]
        corr = corr_frame.corr()
        heatmap_chart(corr, "Correlation heatmap (numerical features and churn)", "RdBu_r", -1, 1, ".2f",
                      height=420)
        pieces = []
        if "tenure" in corr.columns:
            pieces.append(f"tenure vs churn r = {corr.loc['tenure', 'Churn (1 = Yes)']:.2f}")
        if "MonthlyCharges" in corr.columns:
            pieces.append(f"MonthlyCharges vs churn r = {corr.loc['MonthlyCharges', 'Churn (1 = Yes)']:.2f}")
        if {"tenure", "TotalCharges"} <= set(corr.columns):
            pieces.append(f"tenure vs TotalCharges r = {corr.loc['tenure', 'TotalCharges']:.2f} "
                          "(these two features overlap strongly)")
        st.caption("Pearson correlation measures linear association between numerical columns; churn is coded 0/1. "
                   + "; ".join(pieces) + ". Correlation does not establish causation and can miss "
                   "non-linear relationships. SeniorCitizen is a 0/1 flag, so its correlations are indicative only.")

    # --- insight summary ------------------------------------------------------
    with tabs[5]:
        insight_rows = [("Class imbalance",
                         f"{churn_pct:.1f}% of customers churned ({int(dfa['ChurnBin'].sum()):,} of {n_rows:,}).",
                         INSIGHT_ACTIONS["class"])]
        if has("Contract"):
            insight_rows.append(("Contract type", "Churn rate by contract - " + rates_text(dfa, "Contract") + ".",
                                 INSIGHT_ACTIONS["contract"]))
        if has("tenure"):
            m = dfa.groupby("ChurnLabel")["tenure"].median()
            insight_rows.append(("Tenure",
                                 f"Median tenure is {m.get('Churn', float('nan')):.0f} months for churned vs "
                                 f"{m.get('No Churn', float('nan')):.0f} for retained customers.",
                                 INSIGHT_ACTIONS["tenure"]))
        if has("MonthlyCharges"):
            m = dfa.groupby("ChurnLabel")["MonthlyCharges"].median()
            r_mc = dfa["MonthlyCharges"].corr(dfa["ChurnBin"])
            insight_rows.append(("Monthly charges",
                                 f"Median monthly charge is {m.get('Churn', float('nan')):.2f} (churn) vs "
                                 f"{m.get('No Churn', float('nan')):.2f} (no churn); correlation with churn "
                                 f"r = {r_mc:.2f}.", INSIGHT_ACTIONS["charges"]))
        if has("PaymentMethod"):
            insight_rows.append(("Payment method", "Churn rate by method - " + rates_text(dfa, "PaymentMethod") + ".",
                                 INSIGHT_ACTIONS["payment"]))
        service_bits = [f"{c}: {rates_text(dfa, c)}" for c in ["OnlineSecurity", "TechSupport"] if c in dfa.columns]
        if service_bits:
            insight_rows.append(("Online security / tech support", " | ".join(service_bits) + ".",
                                 INSIGHT_ACTIONS["services"]))
        st.markdown("**Insights computed from the loaded data** (the final column is the notebook's suggested use)")
        show_table(pd.DataFrame(insight_rows, columns=["Topic", "Observed in this dataset", "Notebook's suggested action"]))
        st.caption("EDA findings describe associations in this dataset and do not by themselves establish causation.")

# =============================================================================
# D. Preprocessing
# =============================================================================

section("sec-preprocessing", "D", "Data Preprocessing",
        "The pipeline implemented in notebook Q4 and why each step is needed.")

if df is not None:
    cat_n = len([c for c in df.columns if c not in NUMERICAL_COLS + ["customerID", "Churn"]])
    encoded = len(NUMERICAL_COLS) + sum(
        df[c].nunique(dropna=True) - 1 for c in df.columns if c not in NUMERICAL_COLS + ["customerID", "Churn"])
    split = split_summary(tuple(dfa["ChurnBin"].tolist()))
    items = [("Numerical features", f"{len(NUMERICAL_COLS)}"), ("Categorical features", f"{cat_n}"),
             ("Features after one-hot encoding", f"{encoded}")]
    if split:
        items.append(("Train / test rows", f"{split['train']:,} / {split['test']:,}"))
    kpis(items, per_row=len(items))
    if split:
        st.caption(f"Churn share is {split['train_rate']:.2f}% in the training set and {split['test_rate']:.2f}% "
                   "in the test set, confirming that the split is stratified.")

st.markdown("#### Pipeline decisions")
show_table(pd.DataFrame(PREPROCESSING_STEPS, columns=["Step", "Action", "Why it is needed"]))

st.markdown("#### Preprocessing flow")
rows_label = f"{n_rows:,} rows" if n_rows else "raw rows"
train_label = f"Train {split['train']:,}" if df is not None and split else "Train 80%"
test_label = f"Test {split['test']:,}" if df is not None and split else "Test 20%"
flow_a, flow_b = st.columns(2, gap="small")
with flow_a:
    st.caption("1. Cleaning and splitting (applied once, before modelling)")

    stretch(st.graphviz_chart, f"""
    digraph G {{
      rankdir=LR;
      bgcolor="transparent";
      nodesep=0.15;
      ranksep=0.2;
      margin=0.02;
      size="5,2.5!";

      node [
        shape=box,
        style="rounded,filled",
        fillcolor="#E8F0FA",
        color="#2F6DB5",
        fontname="Helvetica",
        fontsize=8,
        fontcolor="#14243A",
        margin="0.06,0.04"
      ];

      edge [color="#2F6DB5", arrowsize=0.55];

      raw [label="Raw CSV\\n{rows_label}"];
      fix [label="Convert TotalCharges\\nblanks to NaN"];
      drop [label="Drop\\ncustomerID"];
      target [label="Encode target\\nYes = 1, No = 0"];
      split [
        label="Stratified split\\n80/20",
        fillcolor="#2F6DB5",
        fontcolor="white"
      ];

      tr [label="{train_label}"];
      te [label="{test_label}"];

      raw -> fix -> drop -> target -> split;
      split -> tr;
      split -> te;
    }}""")
with flow_b:
    st.caption("2. Inside each model Pipeline (fitted on training data only)")
    stretch(st.graphviz_chart, f"""
    digraph G {{
      rankdir=TB; bgcolor="transparent";
      node [shape=box, style="rounded,filled", fillcolor="#E8F0FA", color="#2F6DB5",
            fontname="Helvetica", fontsize=11, fontcolor="#14243A"];
      edge [color="#2F6DB5", fontname="Helvetica", fontsize=9];
      xtr [label="X_train"]; xte [label="X_test"];
      fit [label="ColumnTransformer.fit\\nnumerical: median imputer + StandardScaler\\ncategorical: most-frequent imputer + OneHot(drop first)", fillcolor="#2F6DB5", fontcolor="white"];
      clf [label="Classifier.fit(train features, y_train)"];
      tf [label="ColumnTransformer.transform\\n(learned medians, means, scales, categories)"];
      pred [label="predict / predict_proba"]; met [label="Metrics against y_test"];
      xtr -> fit -> clf; xte -> tf; fit -> tf [style=dashed, label="parameters from train"];
      clf -> pred; tf -> pred -> met;
    }}""")

callout(
    "<b>Fitted on training data vs applied to test data:</b> converting TotalCharges, dropping customerID and "
    "encoding the target are fixed rules applied to the whole table. Imputation medians, scaler statistics and "
    "one-hot categories are <i>learned</i> only when each Pipeline is fitted on X_train and are then applied "
    "unchanged to X_test."
)

# =============================================================================
# E. Model development
# =============================================================================

section("sec-training", "E", "Model Training",
        "The eight classifiers implemented in notebook Q5 and the metrics used to judge them.")

st.markdown("#### Classifiers")
for start in range(0, len(MODEL_INFO), 2):
    for col_ui, info in zip(st.columns(2, gap="large"), MODEL_INFO[start:start + 2]):
        with col_ui:
            card(info["name"],
                 f"<p>{info['definition']}</p><p><b>Why chosen:</b> {info['why']}</p>"
                 f"<p><code>{info['params']}</code></p>", tag=info["kind"])

st.markdown("#### Role of the ensemble models")
callout(
    "Four of the eight models are ensembles, which combine many weak or randomised trees into one predictor. "
    "<b>Random Forest</b> and <b>Extra Trees</b> average many de-correlated trees to reduce variance. "
    "<b>Gradient Boosting</b> and <b>AdaBoost</b> add learners one at a time so that each corrects the mistakes "
    "of the previous ones. The four traditional models (Logistic Regression, KNN, SVM, Decision Tree) act as "
    "baselines for comparison."
)
st.caption("Class imbalance handling: class_weight='balanced' is set for Logistic Regression, SVM, Decision Tree, "
           "Random Forest and Extra Trees. It is not set for KNN, Gradient Boosting and AdaBoost.")

st.markdown("#### Training and evaluation workflow")
stretch(st.graphviz_chart, """
digraph G {
  rankdir=LR; bgcolor="transparent";
  node [shape=box, style="rounded,filled", fillcolor="#E8F0FA", color="#2F6DB5",
        fontname="Helvetica", fontsize=11, fontcolor="#14243A"];
  edge [color="#2F6DB5"];
  a [label="For each of the 8 models"]; b [label="Pipeline:\\npreprocessor -> classifier"];
  c [label="fit(X_train, y_train)"]; d [label="predict and predict_proba\\non X_test"];
  e [label="Accuracy, Precision, Recall,\\nF1, ROC-AUC"]; f [label="Leaderboard sorted\\nby ROC-AUC", fillcolor="#2F6DB5", fontcolor="white"];
  a -> b -> c -> d -> e -> f;
}""")
st.caption("Metrics are computed on the held-out test set (churn = positive class) using each classifier's default "
           "decision threshold. No hyper-parameter search is performed in the notebook; the settings above were fixed manually.")

st.markdown("#### Evaluation metrics")
show_table(pd.DataFrame(METRIC_DEFINITIONS, columns=["Metric", "Formula", "What it measures"]))
st.caption("TP, FP, TN, FN are counts from the confusion matrix with Churn as the positive class.")

# =============================================================================
# F. Model comparison and results
# =============================================================================

section("sec-evaluation", "F", "Model Comparison & Results",
        "Test-set results from data/model_results.csv, plus plots regenerated with the notebook's pipeline.")

repro = None
active, active_source = None, None

if results_raw is None:
    st.warning((results_error or "data/model_results.csv was not found.")
               + " The saved comparison table is unavailable. You can still regenerate the notebook's evaluation "
                 "below, but it is labelled as recomputed rather than saved.")
else:
    st.markdown("#### Saved model comparison (data/model_results.csv)")
    show_df = results_raw
    try:
        highlight_cols = [saved_map[m] for m in METRICS if m in saved_map]
        show_df = results_raw.style.format(precision=4, subset=highlight_cols).highlight_max(
            subset=highlight_cols, color="#DCE9F8")
    except Exception:
        show_df = results_raw
    show_table(show_df)
    st.caption("Table as stored in the file. Cells highlighted blue are the highest value in each metric column. "
               "Confirm in the notebook that the values come from the test set (the notebook computes them on X_test).")
    if saved_std is None:
        st.info("No model-name column was recognised in the file, so automatic charts and statements are "
                "disabled. Recognised names: " + ", ".join(CANONICAL_COLUMNS["Model"]) + ".")
    else:
        missing_metrics = [m for m in METRICS if m not in saved_std.columns]
        if missing_metrics:
            st.info("Metrics not found in the file: " + ", ".join(missing_metrics) + ".")
        active, active_source = saved_std, "saved results (data/model_results.csv)"

# --- regenerate plots on demand -------------------------------------------------
st.markdown("#### Evaluation plots")
st.caption("ROC curves, confusion matrices, the classification report and feature importances are not stored in "
           "the results file. Generate them by re-running the notebook's Q4-Q6 pipeline once on this page "
           "(8 models, usually a few seconds, then cached).")
if df is None:
    st.info("The dataset is needed to regenerate evaluation plots.")
else:
    if "show_eval" not in st.session_state:
        st.session_state["show_eval"] = False

    if st.button("Generate evaluation plots", key="run_eval"):
        st.session_state["show_eval"] = True

    if st.session_state["show_eval"]:
        with st.spinner("Training the notebook's eight pipelines..."):
            repro = run_notebook_evaluation(
                str(dataset_path),
                dataset_path.stat().st_mtime
            )

        if "error" in repro:
            st.error(repro["error"])
            repro = None

if repro is not None and active is None:
    active, active_source = repro["results"], "results recomputed on this page"

if repro is not None:
    st.success(f"Evaluation regenerated on the stratified 80/20 split ({repro['n_train']:,} train / "
               f"{repro['n_test']:,} test rows, scikit-learn {repro['sklearn']}).")
    if saved_std is not None and "ROC-AUC" in saved_std.columns:
        merged = saved_std.assign(key=saved_std["Model"].map(normalise)).merge(
            repro["results"].assign(key=repro["results"]["Model"].map(normalise)),
            on="key", suffixes=(" saved", " recomputed"))
        diffs = []
        for metric in METRICS:
            a, b = f"{metric} saved", f"{metric} recomputed"
            if a in merged.columns and b in merged.columns:
                diffs.append((merged[a] - merged[b]).abs())
        if len(merged) and diffs:
            worst = float(pd.concat(diffs, axis=1).max().max())
            if worst <= 0.0005:
                st.caption(f"Consistency check: the saved results match the recomputed metrics for "
                           f"{len(merged)} models (largest difference {worst:.4f}).")
            else:
                st.warning(f"Consistency check: saved and recomputed metrics differ by up to {worst:.4f}. "
                           "The saved file may come from an earlier notebook run or another library version. "
                           "Re-run the notebook from the top and re-export model_results.csv so the table, "
                           "plots and conclusion agree.")
        else:
            st.caption("Consistency check skipped: model names in the saved file could not be matched.")

if active is not None:
    metric_tabs = st.tabs(["Metric comparison", "ROC curves", "Confusion matrices",
                           "Best-model report", "Feature importance"])
    available = [m for m in METRICS if m in active.columns and active[m].notna().any()]

    with metric_tabs[0]:
        if not available:
            st.info("No numeric metric columns were recognised.")
        else:
            view = st.selectbox("Metric view", ["All metrics"] + available, key="metric_view")
            order_col = "ROC-AUC" if "ROC-AUC" in available else available[0]
            ordered = active.sort_values(order_col, ascending=False)
            if view == "All metrics":
                long = ordered.melt(id_vars="Model", value_vars=available, var_name="Metric", value_name="Score")
                top = float(long["Score"].max())
                bar_chart(long, "Model", "Score", f"Model comparison: all metrics (sorted by {order_col})",
                          "Model", "Score", color="Metric", fmt=".3f", height=440,
                          y_range=[0, 1.12 if top <= 1.0 else None], legend_title="Metric")
            else:
                single = active.sort_values(view, ascending=False)[["Model", view]].dropna()
                top = float(single[view].max())
                bar_chart(single, "Model", view, f"Model comparison: {view}", "Model", view, fmt=".3f",
                          y_range=[0, 1.12 if top <= 1.0 else None])
            st.caption(f"Source: {active_source}. Bars start at zero so differences are not exaggerated; "
                       "the table above gives the exact values.")

    with metric_tabs[1]:
        if repro is None:
            st.info("Click 'Generate evaluation plots' above to display ROC curves.")
        else:
            top4 = repro["results"]["Model"].head(4).tolist()
            roc_chart({n: repro["curves"][n] for n in top4}, "ROC curves: top 4 models by ROC-AUC")
            st.caption("The ROC curve plots recall against the false positive rate across all decision thresholds. "
                       "Curves closer to the top-left corner and larger areas indicate better ranking of churners "
                       "above non-churners; the dashed diagonal is random guessing. Test set only.")

    with metric_tabs[2]:
        if repro is None:
            st.info("Click 'Generate evaluation plots' above to display confusion matrices.")
        else:
            names = repro["results"]["Model"].tolist()
            for start in range(0, len(names), 4):
                for col_ui, name in zip(st.columns(4), names[start:start + 4]):
                    with col_ui:
                        cm = pd.DataFrame(repro["matrices"][name], index=["No Churn", "Churn"],
                                          columns=["No Churn", "Churn"])
                        heatmap_chart(cm, name, "Blues", None, None, "d", height=300,
                                      x_title="Predicted", y_title="Actual", show_scale=False)
            st.caption("Rows are actual classes and columns are predicted classes. The bottom-left cell counts "
                       "missed churners (false negatives) and the top-right cell counts customers wrongly "
                       "flagged as churners (false positives).")

    with metric_tabs[3]:
        if repro is None:
            st.info("Click 'Generate evaluation plots' above to display the classification report.")
        else:
            st.markdown(f"**Best model by ROC-AUC (recomputed): {repro['best']}**")
            report = repro["report"].round(4)
            report["support"] = report["support"].astype(int)
            show_table(report.reset_index().rename(columns={"index": "Class"}))
            st.caption(f"Overall accuracy: {repro['report_accuracy']:.4f}. Precision, recall and F1 are reported "
                       "per class; support is the number of test customers in that class.")

    with metric_tabs[4]:
        if repro is None:
            st.info("Click 'Generate evaluation plots' above to display feature importances.")
        elif repro["importance"] is None:
            st.info("The best ensemble model does not expose feature importances.")
        else:
            hbar_chart(repro["importance"], "Importance", "Feature",
                       f"Top 15 feature importances: {repro['best_ensemble']}", "Importance score", "Feature")
            st.caption("Importance shows how much each encoded feature contributes to the model's splits. It "
                       "describes the model, not causes of churn, and can be shared across correlated features.")

# =============================================================================
# G. Conclusion and limitations
# =============================================================================

section("sec-conclusion", "G", "Conclusion & Limitations",
        "Summary grounded in the results shown above.")


def result_summary(res: pd.DataFrame | None):
    """Facts for the conclusion, derived only from the active results table."""
    if res is None or "ROC-AUC" not in res.columns or res["ROC-AUC"].notna().sum() == 0:
        return None
    ranked = res.dropna(subset=["ROC-AUC"]).sort_values("ROC-AUC", ascending=False).reset_index(drop=True)
    summary = {"ranked": ranked, "best": ranked.iloc[0], "n": len(ranked)}
    if len(ranked) >= 2:
        summary["gap_second"] = float(ranked.iloc[0]["ROC-AUC"] - ranked.iloc[1]["ROC-AUC"])
        k = min(4, len(ranked))
        summary["spread_topk"] = (k, float(ranked.iloc[0]["ROC-AUC"] - ranked.iloc[k - 1]["ROC-AUC"]))
    for metric in ("F1-Score", "Recall"):
        if metric in ranked.columns and ranked[metric].notna().any():
            summary[metric] = ranked.loc[ranked[metric].idxmax()]
    if "Recall" in ranked.columns:
        keys = ranked["Model"].map(normalise)
        unweighted = keys.map(lambda k: any(t in k for t in NO_CLASS_WEIGHT_KEYS))
        if unweighted.any() and (~unweighted).any():
            summary["recall_groups"] = (float(ranked.loc[unweighted, "Recall"].mean()),
                                        float(ranked.loc[~unweighted, "Recall"].mean()))
    return summary


facts = result_summary(active)

st.markdown("#### Problem addressed")
st.markdown("Customer churn lowers revenue, and retention teams need to know which customers are at risk. The "
            "project treats this as a supervised binary classification task on the Telco customer dataset, "
            "with Churn = 1 as the positive class.")

if df is not None:
    st.markdown("#### What the EDA showed")
    bullets = [f"{churn_pct:.1f}% of customers churned, so the classes are imbalanced."]
    if has("Contract"):
        bullets.append("Churn rate by contract: " + rates_text(dfa, "Contract") + ".")
    if has("tenure"):
        m = dfa.groupby("ChurnLabel")["tenure"].median()
        bullets.append(f"Median tenure: {m.get('Churn', float('nan')):.0f} months (churn) vs "
                       f"{m.get('No Churn', float('nan')):.0f} months (no churn).")
    if has("PaymentMethod"):
        bullets.append("Churn rate by payment method: " + rates_text(dfa, "PaymentMethod") + ".")
    for line in bullets:
        st.markdown(f"- {line}")
    st.caption("These are associations observed in this dataset, not proof of cause.")

st.markdown("#### Models evaluated and the selected model")
if facts is None:
    st.info("A results table with a ROC-AUC column is required to state the selected model. Add "
            "data/model_results.csv or generate the evaluation plots in Section F.")
else:
    best = facts["best"]

    def val(row, metric):
        return f"{row[metric]:.4f}" if metric in row.index and pd.notna(row[metric]) else "n/a"

    st.markdown(
        f"{facts['n']} classifiers were evaluated on the held-out test set (source: {active_source}). Ranked by "
        f"ROC-AUC, the project's primary criterion, the highest-ranked model is **{best['Model']}** "
        f"(ROC-AUC {val(best, 'ROC-AUC')}, F1-Score {val(best, 'F1-Score')}, Recall {val(best, 'Recall')}, "
        f"Precision {val(best, 'Precision')}, Accuracy {val(best, 'Accuracy')}).")
    kpis([("Selected model (highest ROC-AUC)", str(best["Model"])), ("ROC-AUC", val(best, "ROC-AUC")),
          ("F1-Score", val(best, "F1-Score")), ("Recall", val(best, "Recall"))], per_row=4)

    evidence = []
    if "F1-Score" in facts:
        f1_row = facts["F1-Score"]
        evidence.append("It also has the highest F1-Score of all models." if f1_row["Model"] == best["Model"] else
                        f"The highest F1-Score belongs to {f1_row['Model']} ({val(f1_row, 'F1-Score')}), "
                        "so the two selection metrics do not agree on one model.")
    if "Recall" in facts:
        rec_row = facts["Recall"]
        evidence.append("It also has the highest recall." if rec_row["Model"] == best["Model"] else
                        f"The highest recall is {val(rec_row, 'Recall')} ({rec_row['Model']}). Because the business "
                        "constraint favours recall, that model is a candidate if missed churners are costly.")
    if "spread_topk" in facts:
        k, spread = facts["spread_topk"]
        evidence.append(f"The top {k} models differ by only {spread:.4f} in ROC-AUC (the gap between first and second "
                        f"is {facts['gap_second']:.4f}), so the ranking rests on small differences from a single "
                        "hold-out split.")
    if "recall_groups" in facts:
        low, high = facts["recall_groups"]
        evidence.append(f"Models trained without class_weight (KNN, Gradient Boosting, AdaBoost) have a mean recall "
                        f"of {low:.3f} against {high:.3f} for the others, consistent with the trade-off between "
                        "recall and precision.")
    for line in evidence:
        st.markdown(f"- {line}")

st.markdown("#### Purpose of the deployed application")
st.markdown("The Streamlit application provides an interface (pages/churn_prediction.py) for estimating whether a "
            "customer is likely to churn. The notebook itself does not export a model, so confirm that the "
            "application uses the same preprocessing and the model selected above.")

st.markdown("#### Limitations and future work")
st.markdown(
    "- **Single hold-out split.** All results come from one stratified 80/20 split. The notebook states that "
    "cross-validation would give a more stable estimate before final deployment.\n"
    "- **No hyper-parameter tuning.** Model settings were fixed manually, so the comparison reflects those "
    "settings rather than each model's best possible performance.\n"
    "- **Default decision threshold.** Precision, recall and F1 depend on the threshold. If missing a churner "
    "costs far more than contacting a loyal customer, compare thresholds and business costs before deployment.\n"
    "- **One dataset.** It is a single sample from one company, so performance on other customers or markets is "
    "not established.\n"
    "- **Associations only.** EDA and feature importance describe patterns in the data and do not show that a "
    "feature causes churn."
)

st.divider()
st.caption(f"Telco Customer Churn Prediction | MCA Machine Learning Lab | Source of truth: {NOTEBOOK_NAME}")
