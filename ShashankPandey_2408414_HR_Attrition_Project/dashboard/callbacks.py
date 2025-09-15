import os
import re
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio  # NEW: for global theming

from dash import Input, Output, callback
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_curve, auc, precision_recall_curve
)
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

# SciPy is optional (for chi-square p-value)
try:
    from scipy.stats import chi2_contingency
    HAVE_SCIPY = True
except Exception:
    HAVE_SCIPY = False


# ========= POLISH: Global Visual Theme (Dark, clean, consistent) =========
# Color set chosen for good contrast on dark backgrounds
COLORWAY = [
    "#60a5fa", "#34d399", "#f59e0b", "#f472b6", "#22d3ee",
    "#a78bfa", "#f87171", "#10b981", "#c084fc", "#fb7185"
]

# Register a reusable template
pio.templates["dash_dark_hr"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="#0b1220",
        plot_bgcolor="#0b1220",
        font=dict(family="Inter, Segoe UI, system-ui, -apple-system, Roboto",
                  size=13, color="#e8eaed"),
        title=dict(font=dict(size=18, color="#f3f4f6")),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=-0.25,
            xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)"
        ),
        margin=dict(l=40, r=24, t=56, b=56),
        xaxis=dict(
            showgrid=True, gridcolor="#1f2a44",
            zeroline=False, linecolor="#334155", tickcolor="#334155"
        ),
        yaxis=dict(
            showgrid=True, gridcolor="#1f2a44",
            zeroline=False, linecolor="#334155", tickcolor="#334155"
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor="#475569",
            font=dict(color="#e5e7eb")
        ),
        colorway=COLORWAY
    )
)

# Make it the default for all figures created below
pio.templates.default = "dash_dark_hr"

# Shared helper: finalize any fig with consistent height/hover
def _style(fig: go.Figure, title: str | None = None, height: int = 380):
    if title:
        fig.update_layout(title=title)
    fig.update_layout(height=height)
    # Nicer default hover
    fig.update_traces(hovertemplate="%{x}<br>%{y}<extra></extra>")
    return fig

# Bar-like helper for count histograms
def _style_hist(fig: go.Figure, title: str):
    fig.update_traces(marker_line_width=0)  # cleaner bars
    fig.update_layout(barmode="group")
    return _style(fig, title)

# Line-curve helper
def _style_line(fig: go.Figure, title: str):
    fig.update_traces(line=dict(width=3))
    return _style(fig, title)

# Heatmap helper
def _style_heat(fig: go.Figure, title: str):
    fig.update_layout(coloraxis_showscale=True)
    # A color scale that pops on dark bg
    fig.update_coloraxes(colorscale=[
        [0.0, "#0ea5e9"], [0.25, "#22d3ee"], [0.5, "#34d399"],
        [0.75, "#f59e0b"], [1.0, "#ef4444"]
    ])
    return _style(fig, title)


# ----------- DATA LOADER (prefers RAW dataset) -----------
RAW_FILES = [
    os.path.join("data", "HR-Employee-Attrition.csv"),
    os.path.join("data", "HR_Employee_Attrition.csv"),
]
CLEAN_FILES = [
    os.path.join("data", "HR_Cleaned.csv"),
]


def _find_first_existing(paths):
    """Search for a file relative to CWD and project root (parent of dashboard/)."""
    for p in paths:
        # Try from CWD
        abs1 = os.path.abspath(os.path.join(os.getcwd(), p))
        if os.path.exists(abs1):
            return abs1
        # Try from project root (parent of this file's dir)
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
        abs2 = os.path.abspath(os.path.join(project_root, p))
        if os.path.exists(abs2):
            return abs2
    return None


def load_data() -> pd.DataFrame:
    # 1) Prefer RAW file; 2) else fallback to cleaned
    path = _find_first_existing(RAW_FILES) or _find_first_existing(CLEAN_FILES)
    if path is None:
        raise FileNotFoundError("Place the RAW file at data/HR-Employee-Attrition.csv")

    print(f"[dash] Loading data from: {path}")
    df = pd.read_csv(path, encoding="utf-8", on_bad_lines="skip")

    # Normalize obvious column spellings (just in case)
    df.columns = [str(c).strip() for c in df.columns]
    lower = {c: re.sub(r"\s+", "", c.lower()) for c in df.columns}
    ren = {}
    for orig, low in lower.items():
        if low == "monthlyincome": ren[orig] = "MonthlyIncome"
        elif low == "overtime":    ren[orig] = "OverTime"
        elif low == "jobrole":     ren[orig] = "JobRole"
        elif low == "department":  ren[orig] = "Department"
        elif low == "attrition":   ren[orig] = "Attrition"
        elif low == "age":         ren[orig] = "Age"
    if ren:
        df = df.rename(columns=ren)

    # Expect RAW labels/columns
    for col in ["Attrition", "Department", "JobRole", "OverTime", "Age"]:
        if col not in df.columns:
            raise ValueError("This file doesn't look like the RAW IBM HR dataset (missing expected columns).")

    # Clean types/values
    df["Attrition"] = df["Attrition"].astype(str).str.strip().str.title()  # Yes/No
    df["OverTime"]  = df["OverTime"].astype(str).str.strip().str.title()    # Yes/No
    df["Age"] = pd.to_numeric(df["Age"], errors="coerce")
    df = df[(df["Age"] >= 16) & (df["Age"] <= 80)]
    if "MonthlyIncome" in df.columns:
        df["MonthlyIncome"] = pd.to_numeric(df["MonthlyIncome"], errors="coerce")

    # Binary target
    df["Attrition_bin"] = (df["Attrition"] == "Yes").astype(int)

    # Drop rows without required fields
    df = df.dropna(subset=["Attrition", "Department", "JobRole", "OverTime", "Age"])

    if df.empty:
        raise ValueError("Dataset is empty after basic cleaning. Check the RAW file.")
    return df


def filter_df(df, dept, role, ot, age_range):
    if dept:
        df = df[df["Department"].isin(dept)]
    if role:
        df = df[df["JobRole"].isin(role)]
    if ot:
        df = df[df["OverTime"].isin(ot)]
    if age_range:
        lo, hi = age_range
        df = df[(df["Age"] >= lo) & (df["Age"] <= hi)]
    return df


def make_model(name: str):
    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1000, class_weight="balanced", solver="liblinear")
    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=5, class_weight="balanced", random_state=42)
    if name == "Random Forest":
        return RandomForestClassifier(n_estimators=200, max_depth=12,
                                      class_weight="balanced", random_state=42)
    # SVM
    return SVC(probability=True, class_weight="balanced", random_state=42)


def _make_ohe():
    """Return a OneHotEncoder that works across sklearn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=True)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=True)


def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(with_mean=False), num_cols),
            ("cat", _make_ohe(), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=1.0
    )


# ----------------- DATA BOOTSTRAP -----------------
@callback(Output("store-data", "data"),
          Input("tabs", "value"))
def boot_data(_):
    try:
        df = load_data()
        return {
            "data": df.to_dict("records"),
            "age_min": int(df["Age"].min()),
            "age_max": int(df["Age"].max()),
        }
    except Exception as e:
        print("[dash][ERROR] boot_data failed:", repr(e))
        return {"data": [], "age_min": 18, "age_max": 60, "error": str(e)}


# Build dropdown options *from rows* so they never show empty
@callback(
    Output("dept-dd", "options"),
    Output("role-dd", "options"),
    Output("age-slider", "min"),
    Output("age-slider", "max"),
    Output("age-slider", "value"),
    Input("store-data", "data"),
)
def populate_filters(store):
    df = pd.DataFrame(store.get("data", []))
    if df.empty:
        return [], [], 18, 60, [18, 60]

    depts = sorted(df["Department"].dropna().astype(str).unique().tolist())
    roles = sorted(df["JobRole"].dropna().astype(str).unique().tolist())
    dept_opts = [{"label": d, "value": d} for d in depts]
    role_opts = [{"label": r, "value": r} for r in roles]

    lo, hi = int(df["Age"].min()), int(df["Age"].max())
    return dept_opts, role_opts, lo, hi, [lo, hi]


@callback(Output("data-error", "children"), Input("store-data", "data"))
def show_error(store):
    if isinstance(store, dict) and store.get("error"):
        return f"⚠️ {store['error']}"
    return ""


# ----------------- KPIs -----------------
@callback(
    Output("kpi-emps", "children"),
    Output("kpi-attr", "children"),
    Output("kpi-rate", "children"),
    Output("kpi-income", "children"),
    Input("store-data", "data"),
    Input("dept-dd", "value"),
    Input("role-dd", "value"),
    Input("ot-dd", "value"),
    Input("age-slider", "value"),
)
def update_kpis(store, dept, role, ot, age_rng):
    df = pd.DataFrame(store.get("data", []))
    if df.empty:
        return "0", "0", "0.0%", "—"

    df = filter_df(df, dept, role, ot, age_rng)
    n = len(df)
    n_attr = int(df["Attrition_bin"].sum())
    rate = (100 * n_attr / n) if n > 0 else 0
    avg_inc = df["MonthlyIncome"].mean() if "MonthlyIncome" in df.columns and n > 0 else np.nan
    return f"{n:,}", f"{n_attr:,}", f"{rate:.1f}%", (f"{avg_inc:,.0f}" if pd.notnull(avg_inc) else "—")


# ----------------- EDA FIGS -----------------
@callback(
    Output("fig-attr-by-dept", "figure"),
    Output("fig-attr-by-role", "figure"),
    Output("fig-age-dist", "figure"),
    Output("fig-income-dist", "figure"),
    Input("store-data", "data"),
    Input("dept-dd", "value"),
    Input("role-dd", "value"),
    Input("ot-dd", "value"),
    Input("age-slider", "value"),
)
def eda_figs(store, dept, role, ot, age_rng):
    df = pd.DataFrame(store.get("data", []))
    if df.empty:
        empty = go.Figure()
        empty.update_layout(title="No data loaded", height=360)
        return empty, empty, empty, empty

    df = filter_df(df, dept, role, ot, age_rng)

    f1 = px.histogram(
        df, x="Department", color="Attrition", barmode="group",
        category_orders={"Attrition": ["No", "Yes"]},
        labels={"count": "Employees"}
    )
    f1 = _style_hist(f1, "Attrition by Department")

    f2 = px.histogram(
        df, x="JobRole", color="Attrition", barmode="group",
        category_orders={"Attrition": ["No", "Yes"]},
        labels={"count": "Employees"}
    )
    f2.update_layout(xaxis_tickangle=25)
    f2 = _style_hist(f2, "Attrition by Job Role")

    f3 = px.histogram(df, x="Age", nbins=24, marginal="box", labels={"Age": "Age"})
    f3.update_traces(marker_line_width=0)
    f3 = _style(f3, "Age Distribution")

    if "MonthlyIncome" in df.columns:
        f4 = px.histogram(df, x="MonthlyIncome", nbins=30, marginal="box",
                          labels={"MonthlyIncome": "Monthly Income"})
        f4.update_traces(marker_line_width=0)
        f4 = _style(f4, "Monthly Income Distribution")
    else:
        f4 = go.Figure()
        f4 = _style(f4, "Monthly Income Distribution")

    return f1, f2, f3, f4


# ----------------- Hypothesis: Chi-square -----------------
@callback(
    Output("fig-dept-attr-heat", "figure"),
    Output("chi2-summary", "children"),
    Output("chi2-interpret", "children"),
    Input("store-data", "data"),
    Input("dept-dd", "value"),
    Input("age-slider", "value"),
)
def chi_square(store, dept, age_rng):
    df = pd.DataFrame(store.get("data", []))
    if df.empty:
        return _style(go.Figure(), "Department × Attrition (Counts)"), "No data.", ""

    df = filter_df(df, dept, None, None, age_rng)
    ct = pd.crosstab(df["Department"], df["Attrition"])
    for col in ["No", "Yes"]:
        if col not in ct.columns:
            ct[col] = 0
    ct = ct[["No", "Yes"]]

    fig = px.imshow(
        ct, text_auto=True, aspect="auto",
        labels=dict(color="Count"),
    )
    fig = _style_heat(fig, "Department × Attrition (Counts)")

    if HAVE_SCIPY:
        chi2, p, dof, _ = chi2_contingency(ct)
        expl = f"χ²={chi2:.2f}, dof={dof}, p={p:.4f}"
        interp = ("**Reject H₀**: Evidence of association (p<0.05)."
                  if p < 0.05 else
                  "**Fail to Reject H₀**: No significant association at α=0.05.")
    else:
        expl = "Install SciPy to compute p-value."
        interp = ""
    return fig, expl, interp


# ----------------- Modeling -----------------
@callback(
    Output("fig-roc", "figure"),
    Output("fig-pr", "figure"),
    Output("model-metrics", "children"),
    Input("store-data", "data"),
    Input("model-dd", "value"),
    Input("dept-dd", "value"),
    Input("role-dd", "value"),
    Input("ot-dd", "value"),
    Input("age-slider", "value"),
)
def train_model(store, model_name, dept, role, ot, age_rng):
    df = pd.DataFrame(store.get("data", []))
    if df.empty:
        fig = _style(go.Figure(), "No data")
        return fig, fig, "<b>No data to train.</b>"

    df = filter_df(df, dept, role, ot, age_rng)

    y = df["Attrition_bin"].astype(int)
    X = df.drop(columns=["Attrition", "Attrition_bin"])
    if X.empty:
        fig = _style(go.Figure(), "No features")
        return fig, fig, "<b>No features available after filtering.</b>"

    # Guard against not enough samples per class
    if y.nunique() < 2 or (y.value_counts().min() < 2):
        fig = _style(go.Figure(), "Not enough class variety to train")
        return fig, fig, "<b>Need at least 2 samples of each class after filters.</b>"

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.30, random_state=42, stratify=y
    )

    pre = build_preprocessor(X_train)
    clf = make_model(model_name)
    pipe = Pipeline([("pre", pre), ("clf", clf)])
    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else None

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)

    roc_fig = go.Figure(); pr_fig = go.Figure()
    if y_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        roc_auc = auc(fpr, tpr)
        roc_fig.add_trace(go.Scatter(
            x=fpr, y=tpr, mode="lines", name=f"AUC={roc_auc:.3f}",
            line=dict(width=4)
        ))
        roc_fig.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines",
            name="Chance", line=dict(dash="dash", width=2, color="#64748b")
        ))
        roc_fig = _style_line(roc_fig, "ROC Curve")
        roc_fig.update_xaxes(title="False Positive Rate")
        roc_fig.update_yaxes(title="True Positive Rate")

        precision, recall, _ = precision_recall_curve(y_test, y_proba)
        pr_fig.add_trace(go.Scatter(
            x=recall, y=precision, mode="lines", name="PR", line=dict(width=4)
        ))
        pr_fig = _style_line(pr_fig, "Precision–Recall Curve")
        pr_fig.update_xaxes(title="Recall")
        pr_fig.update_yaxes(title="Precision")
    else:
        roc_fig = _style(roc_fig, "ROC Curve (proba not available)")
        pr_fig = _style(pr_fig, "Precision–Recall Curve (proba not available)")

    metrics_html = f"""
    <div class="metrics">
      <h6>Performance — {model_name}</h6>
      <ul>
        <li><b>Accuracy:</b> {acc:.3f}</li>
        <li><b>Precision:</b> {prec:.3f}</li>
        <li><b>Recall:</b> {rec:.3f}</li>
        <li><b>F1-score:</b> {f1:.3f}</li>
      </ul>
    </div>
    """
    return roc_fig, pr_fig, metrics_html
