# =============================================================================
#  Student Performance Predictor — Streamlit App
#  Run:  streamlit run streamlit_app.py
#  Requires: streamlit scikit-learn pandas numpy matplotlib seaborn
# =============================================================================

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import io, os, warnings
warnings.filterwarnings("ignore")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student Performance Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3, h4 { font-family: 'Syne', sans-serif !important; }

.stApp { background: #060d1a; color: #e2e8f0; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #0d1b2e !important;
    border-right: 1px solid #1e3a5f;
}
[data-testid="stSidebar"] * { color: #c8d8ef !important; }
[data-testid="stSidebar"] .stSelectbox label,
[data-testid="stSidebar"] .stSlider label { color: #7db3e8 !important; font-size: 0.78rem !important; font-weight: 500; letter-spacing: 0.05em; text-transform: uppercase; }

/* ── Cards ── */
.metric-card {
    background: linear-gradient(135deg, #0d1b2e 0%, #102040 100%);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 1.1rem 1.4rem;
    text-align: center;
    box-shadow: 0 4px 24px rgba(0,0,0,0.4);
}
.metric-card .label {
    font-size: 0.72rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #5b8fbe;
    font-family: 'Syne', sans-serif;
    margin-bottom: 0.3rem;
}
.metric-card .value {
    font-size: 2.1rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    color: #38bdf8;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 0.72rem;
    color: #4a7599;
    margin-top: 0.15rem;
}

/* ── Prediction banner ── */
.pred-banner {
    background: linear-gradient(135deg, #0c2340 0%, #0a3060 100%);
    border: 2px solid #38bdf8;
    border-radius: 16px;
    padding: 1.8rem 2rem;
    text-align: center;
    box-shadow: 0 0 40px rgba(56,189,248,0.15);
    margin: 1rem 0;
}
.pred-banner .pred-label {
    font-size: 0.8rem;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #7db3e8;
    font-family: 'Syne', sans-serif;
    margin-bottom: 0.5rem;
}
.pred-banner .pred-score {
    font-size: 4.5rem;
    font-weight: 800;
    font-family: 'Syne', sans-serif;
    color: #38bdf8;
    line-height: 1;
}
.pred-banner .pred-grade {
    font-size: 1.2rem;
    font-weight: 600;
    margin-top: 0.4rem;
    font-family: 'Syne', sans-serif;
}

/* ── Section headers ── */
.section-header {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #38bdf8;
    border-bottom: 1px solid #1e3a5f;
    padding-bottom: 0.4rem;
    margin: 1.5rem 0 1rem 0;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] { background: #0d1b2e; border-radius: 10px; gap: 4px; padding: 4px; }
.stTabs [data-baseweb="tab"] { background: transparent; color: #5b8fbe; border-radius: 8px; font-family: 'Syne', sans-serif; font-size: 0.82rem; letter-spacing: 0.04em; padding: 0.45rem 1.1rem; }
.stTabs [aria-selected="true"] { background: #1e3a5f !important; color: #38bdf8 !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #1565c0, #0288d1);
    color: white;
    border: none;
    border-radius: 10px;
    padding: 0.6rem 1.8rem;
    font-family: 'Syne', sans-serif;
    font-weight: 700;
    letter-spacing: 0.06em;
    font-size: 0.9rem;
    transition: all 0.2s;
    width: 100%;
}
.stButton > button:hover { background: linear-gradient(135deg, #1976d2, #039be5); transform: translateY(-1px); box-shadow: 0 6px 20px rgba(56,189,248,0.25); }

/* ── Inputs / Sliders ── */
.stSlider [data-baseweb="slider"] div[role="slider"] { background: #38bdf8 !important; }
div[data-baseweb="select"] { background: #0d1b2e !important; border-color: #1e3a5f !important; }

/* ── File uploader ── */
[data-testid="stFileUploader"] { border: 2px dashed #1e3a5f; border-radius: 12px; background: #0a1628; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; } ::-webkit-scrollbar-track { background: #0d1b2e; }
::-webkit-scrollbar-thumb { background: #1e3a5f; border-radius: 3px; }

/* Hide Streamlit default branding */
#MainMenu, footer, header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Plot theme helper ─────────────────────────────────────────────────────────
PLT_PARAMS = {
    "figure.facecolor": "#060d1a", "axes.facecolor": "#0d1b2e",
    "axes.edgecolor": "#1e3a5f",   "axes.labelcolor": "#c8d8ef",
    "xtick.color": "#5b8fbe",      "ytick.color": "#5b8fbe",
    "text.color": "#e2e8f0",       "grid.color": "#1e3a5f",
    "grid.linestyle": "--",        "grid.alpha": 0.6,
    "font.family": "DejaVu Sans",
}
ACCENT  = "#38bdf8"
ACCENT2 = "#f472b6"
GOOD    = "#4ade80"
WARN    = "#fb923c"


# ═══════════════════════════════════════════════════════════════════════════════
#  DATA & MODEL  (cached)
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_data(show_spinner=False)
def load_and_train(uploaded_bytes: bytes):
    df = pd.read_csv(io.BytesIO(uploaded_bytes))

    # ── Clean
    drop_cols = [c for c in ["student_id", "final_grade"] if c in df.columns]
    df.drop(columns=drop_cols, inplace=True)
    df.dropna(subset=["overall_score"], inplace=True)

    # ── Encode
    X = df.drop(columns=["overall_score"])
    y = df["overall_score"]
    cat_cols = X.select_dtypes(include=["object", "string"]).columns.tolist()
    X_enc = pd.get_dummies(X, columns=cat_cols, drop_first=True)

    # ── Split & train
    X_train, X_test, y_train, y_test = train_test_split(
        X_enc, y, test_size=0.20, random_state=42
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "r2":   r2_score(y_test, y_pred),
        "mae":  mean_absolute_error(y_test, y_pred),
        "rmse": np.sqrt(mean_squared_error(y_test, y_pred)),
    }
    coeff_df = pd.DataFrame({
        "Feature":     X_enc.columns,
        "Coefficient": model.coef_,
    }).sort_values("Coefficient", key=abs, ascending=False).reset_index(drop=True)

    return df, X_enc, y, model, X_train, X_test, y_train, y_test, y_pred, metrics, coeff_df, cat_cols


# ═══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR  ─  upload + controls
# ═══════════════════════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 1rem 0 1.5rem 0;'>
        <div style='font-size:2.5rem'>🎓</div>
        <div style='font-family:Syne,sans-serif; font-size:1.05rem; font-weight:800;
                    color:#38bdf8; letter-spacing:0.05em;'>STUDENT PERFORMANCE</div>
        <div style='font-family:Syne,sans-serif; font-size:0.68rem; letter-spacing:0.22em;
                    color:#5b8fbe; text-transform:uppercase;'>ML Predictor · v1.0</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='section-header'>Dataset</div>", unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")

    # Try bundled CSV path as fallback
    BUNDLED = "Student_Performance.csv"
    use_bundled = (uploaded is None) and os.path.exists(BUNDLED)

    if use_bundled:
        st.info(f"📂 Using bundled `{BUNDLED}`", icon="📂")

    st.markdown("<div class='section-header'>Visualisation</div>", unsafe_allow_html=True)
    top_n_coeff = st.slider("Top N coefficients", 5, 25, 15)
    show_residuals = st.checkbox("Show residuals histogram", value=True)


# ═══════════════════════════════════════════════════════════════════════════════
#  MAIN  ─  load data or show landing
# ═══════════════════════════════════════════════════════════════════════════════

# ── Hero header
st.markdown("""
<div style='padding: 2rem 0 0.5rem 0;'>
    <div style='font-family:Syne,sans-serif; font-size:2.4rem; font-weight:800;
                color:#f1f5f9; line-height:1.1;'>
        Student Performance
        <span style='color:#38bdf8;'>Predictor</span>
    </div>
    <div style='color:#4a7599; font-size:0.9rem; margin-top:0.3rem;'>
        Multiple Linear Regression · Scikit-Learn · Interactive Dashboard
    </div>
</div>
""", unsafe_allow_html=True)

# ── Gate: need a data source
if uploaded is None and not use_bundled:
    st.markdown("""
    <div style='margin-top:3rem; text-align:center; padding:3rem;
                background:#0d1b2e; border:2px dashed #1e3a5f; border-radius:16px;'>
        <div style='font-size:3rem;'>📂</div>
        <div style='font-family:Syne,sans-serif; font-size:1.3rem; color:#7db3e8;
                    margin:0.8rem 0 0.4rem;'>Upload your CSV to begin</div>
        <div style='color:#4a7599; font-size:0.85rem;'>
            Expected columns: student_id · age · gender · school_type ·
            parent_education · study_hours · attendance_percentage ·
            internet_access · travel_time · extra_activities · study_method ·
            math_score · science_score · english_score · <b style="color:#38bdf8">overall_score</b>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

# ── Load
with st.spinner("Training model…"):
    raw_bytes = uploaded.read() if uploaded else open(BUNDLED, "rb").read()
    (df, X_enc, y, model,
     X_train, X_test, y_train, y_test, y_pred,
     metrics, coeff_df, cat_cols) = load_and_train(raw_bytes)


# ═══════════════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════════════

tab_overview, tab_viz, tab_predict, tab_data = st.tabs([
    "📊  Model Overview",
    "📈  Visualisations",
    "🔮  Predict Score",
    "🗂️  Raw Data",
])


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 ─ Overview
# ─────────────────────────────────────────────────────────────────────────────
with tab_overview:

    # ── Metric cards row
    st.markdown("<div class='section-header'>Evaluation Metrics</div>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)

    def metric_card(col, label, value, sub=""):
        col.markdown(f"""
        <div class='metric-card'>
            <div class='label'>{label}</div>
            <div class='value'>{value}</div>
            <div class='sub'>{sub}</div>
        </div>""", unsafe_allow_html=True)

    metric_card(c1, "R² Score",  f"{metrics['r2']:.4f}",  f"{metrics['r2']*100:.1f}% variance explained")
    metric_card(c2, "MAE",       f"{metrics['mae']:.3f}", "Mean Absolute Error")
    metric_card(c3, "RMSE",      f"{metrics['rmse']:.3f}","Root Mean Squared Error")
    metric_card(c4, "Samples",   f"{len(df):,}",          f"Train {len(X_train):,} · Test {len(X_test):,}")

    # ── Dataset info
    st.markdown("<div class='section-header'>Dataset at a Glance</div>", unsafe_allow_html=True)
    col_a, col_b = st.columns(2)
    with col_a:
        num_df = df.select_dtypes(include=[np.number])
        st.dataframe(num_df.describe().round(2).T.style
                     .background_gradient(cmap="Blues", subset=["mean", "std"])
                     .set_table_styles([{"selector": "th", "props": [("background", "#0d1b2e"), ("color", "#7db3e8")]}]),
                     use_container_width=True)
    with col_b:
        st.markdown("**Features after One-Hot Encoding**")
        st.markdown(f"- **{len(X_enc.columns)}** total features fed to model")
        st.markdown(f"- **{len(cat_cols)}** categorical → one-hot encoded: `{'` · `'.join(cat_cols)}`")
        st.markdown(f"- **{len(df.select_dtypes(include=[np.number]).columns) - 1}** numeric features")
        st.markdown("**Train / Test Split**")
        st.progress(0.8, text="80 % train  ·  20 % test")

    # ── Top coefficients table
    st.markdown("<div class='section-header'>Top Feature Coefficients</div>", unsafe_allow_html=True)
    display_coeff = coeff_df.head(top_n_coeff).copy()
    display_coeff["Impact"] = display_coeff["Coefficient"].apply(
        lambda v: "🟢 Positive" if v > 0 else "🟠 Negative")
    display_coeff["Coefficient"] = display_coeff["Coefficient"].round(4)
    st.dataframe(
        display_coeff.style
        .bar(subset=["Coefficient"], color=["#fb923c", "#38bdf8"])
        .set_table_styles([{"selector": "th", "props": [("background", "#0d1b2e"), ("color", "#38bdf8")]}]),
        use_container_width=True, height=380
    )


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 ─ Visualisations
# ─────────────────────────────────────────────────────────────────────────────
with tab_viz:

    plt.rcParams.update(PLT_PARAMS)

    # ── Actual vs Predicted + Residuals
    st.markdown("<div class='section-header'>Actual vs. Predicted Scores</div>", unsafe_allow_html=True)
    n_cols = 2 if show_residuals else 1
    fig, axes = plt.subplots(1, n_cols, figsize=(14 if show_residuals else 8, 5))
    if n_cols == 1:
        axes = [axes]

    ax1 = axes[0]
    ax1.scatter(y_test, y_pred, alpha=0.3, s=16, color=ACCENT, edgecolors="none")
    lims = [max(y_test.min(), y_pred.min()) - 2, min(y_test.max(), y_pred.max()) + 2]
    ax1.plot(lims, lims, color=ACCENT2, lw=2, linestyle="--", label="Perfect Fit")
    ax1.set_xlabel("Actual Score"); ax1.set_ylabel("Predicted Score")
    ax1.set_title("Actual vs. Predicted", fontweight="bold", color="#e2e8f0")
    ax1.legend(fontsize=9)
    ax1.text(0.05, 0.93, f"R²={metrics['r2']:.4f}\nMAE={metrics['mae']:.3f}\nRMSE={metrics['rmse']:.3f}",
             transform=ax1.transAxes, fontsize=9, va="top", color="#f8fafc",
             bbox=dict(boxstyle="round,pad=0.4", facecolor="#060d1a", alpha=0.75))

    if show_residuals:
        ax2 = axes[1]
        residuals = y_test.values - y_pred
        ax2.hist(residuals, bins=60, color=ACCENT, edgecolor="#060d1a", alpha=0.85)
        ax2.axvline(0, color=ACCENT2, lw=2, linestyle="--")
        ax2.set_xlabel("Residual (Actual − Predicted)"); ax2.set_ylabel("Frequency")
        ax2.set_title("Residual Distribution", fontweight="bold", color="#e2e8f0")

    plt.tight_layout()
    st.pyplot(fig, use_container_width=True)
    plt.close()

    # ── Correlation Heatmap
    st.markdown("<div class='section-header'>Correlation Heatmap</div>", unsafe_allow_html=True)
    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr()
    fig2, ax = plt.subplots(figsize=(11, 8))
    cmap = sns.diverging_palette(220, 10, as_cmap=True)
    sns.heatmap(corr, annot=True, fmt=".2f", cmap=cmap,
                linewidths=0.5, linecolor="#060d1a",
                annot_kws={"size": 9, "color": "#f1f5f9"},
                ax=ax, vmin=-1, vmax=1, square=True,
                cbar_kws={"shrink": 0.8})
    ax.set_title("Numeric Feature Correlations", fontsize=12,
                 fontweight="bold", color="#e2e8f0", pad=12)
    ax.tick_params(colors="#7db3e8", labelsize=9)
    plt.xticks(rotation=45, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig2, use_container_width=True)
    plt.close()

    # ── Feature coefficients
    st.markdown("<div class='section-header'>Feature Coefficients (Top {top_n})</div>".replace("{top_n}", str(top_n_coeff)),
                unsafe_allow_html=True)
    top_df = coeff_df.head(top_n_coeff)
    colors = [GOOD if c > 0 else WARN for c in top_df["Coefficient"]]
    fig3, ax = plt.subplots(figsize=(12, max(5, top_n_coeff * 0.45)))
    bars = ax.barh(top_df["Feature"], top_df["Coefficient"],
                   color=colors, edgecolor="#060d1a", height=0.65)
    ax.axvline(0, color="#5b8fbe", lw=1.2, linestyle="--")
    ax.set_xlabel("Regression Coefficient")
    ax.set_title(f"Top {top_n_coeff} Most Influential Features",
                 fontweight="bold", color="#e2e8f0")
    ax.invert_yaxis()
    for bar, val in zip(bars, top_df["Coefficient"]):
        offset = 0.02 if val >= 0 else -0.02
        ax.text(val + offset, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center",
                ha="left" if val >= 0 else "right",
                fontsize=8, color="#f1f5f9")
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(facecolor=GOOD, label="Positive ↑"),
                        Patch(facecolor=WARN, label="Negative ↓")],
              fontsize=9, loc="lower right")
    plt.tight_layout()
    st.pyplot(fig3, use_container_width=True)
    plt.close()


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 ─ Predict
# ─────────────────────────────────────────────────────────────────────────────
with tab_predict:
    st.markdown("<div class='section-header'>Enter Student Profile</div>", unsafe_allow_html=True)

    # ── Infer unique values from dataset
    def uniq(col):
        return sorted(df[col].dropna().unique().tolist()) if col in df.columns else []

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📋 Demographics**")
        age    = st.slider("Age", int(df["age"].min()), int(df["age"].max()), 16)
        gender = st.selectbox("Gender",      options=uniq("gender"))
        school = st.selectbox("School Type", options=uniq("school_type"))
        parent = st.selectbox("Parent Education", options=uniq("parent_education"))

    with col2:
        st.markdown("**📚 Academic Habits**")
        study_hours   = st.slider("Study Hours / day", 0.0, float(df["study_hours"].max()), 3.0, 0.5)
        attendance    = st.slider("Attendance %",       0.0, 100.0, 75.0, 0.5)
        internet      = st.selectbox("Internet Access",   options=uniq("internet_access"))
        extra         = st.selectbox("Extra Activities",  options=uniq("extra_activities"))

    with col3:
        st.markdown("**📐 Scores & Other**")
        math_s   = st.slider("Math Score",    0.0, 100.0, 65.0, 0.5)
        sci_s    = st.slider("Science Score", 0.0, 100.0, 65.0, 0.5)
        eng_s    = st.slider("English Score", 0.0, 100.0, 65.0, 0.5)
        travel   = st.selectbox("Travel Time",   options=uniq("travel_time"))
        method   = st.selectbox("Study Method",  options=uniq("study_method"))

    st.markdown("")
    predict_btn = st.button("🔮  Predict Overall Score", use_container_width=True)

    if predict_btn:
        # Build a one-row dataframe matching training schema
        input_dict = {
            "age":                   age,
            "gender":                gender,
            "school_type":           school,
            "parent_education":      parent,
            "study_hours":           study_hours,
            "attendance_percentage": attendance,
            "internet_access":       internet,
            "travel_time":           travel,
            "extra_activities":      extra,
            "study_method":          method,
            "math_score":            math_s,
            "science_score":         sci_s,
            "english_score":         eng_s,
        }
        input_df  = pd.DataFrame([input_dict])
        input_enc = pd.get_dummies(input_df, columns=cat_cols, drop_first=True)

        # Align to training columns (fill missing dummies with 0)
        input_enc = input_enc.reindex(columns=X_enc.columns, fill_value=0)

        prediction = float(model.predict(input_enc)[0])
        prediction = max(0.0, min(100.0, prediction))

        # Grade mapping
        def grade(s):
            if s >= 90: return ("A+", "#4ade80")
            if s >= 80: return ("A",  "#86efac")
            if s >= 70: return ("B",  "#38bdf8")
            if s >= 60: return ("C",  "#fbbf24")
            if s >= 50: return ("D",  "#fb923c")
            return           ("F",  "#f87171")

        g_label, g_color = grade(prediction)

        st.markdown(f"""
        <div class='pred-banner'>
            <div class='pred-label'>Predicted Overall Score</div>
            <div class='pred-score'>{prediction:.1f}</div>
            <div class='pred-grade' style='color:{g_color};'>Grade: {g_label}</div>
        </div>
        """, unsafe_allow_html=True)

        # Score bar
        pct = prediction / 100
        bar_color = g_color
        st.markdown(f"""
        <div style='margin-top:0.5rem;'>
            <div style='display:flex; justify-content:space-between;
                        font-size:0.75rem; color:#5b8fbe; margin-bottom:4px;'>
                <span>0</span><span>50</span><span>100</span>
            </div>
            <div style='background:#0d1b2e; border-radius:8px; height:14px; overflow:hidden;
                        border:1px solid #1e3a5f;'>
                <div style='width:{pct*100:.1f}%; background:{bar_color};
                            height:100%; border-radius:8px;
                            box-shadow: 0 0 12px {bar_color}88;
                            transition: width 0.6s ease;'></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Mini insights
        top3 = coeff_df.head(3)
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**📌 Top 3 Drivers of This Prediction**")
        for _, row in top3.iterrows():
            icon = "🟢" if row["Coefficient"] > 0 else "🟠"
            st.markdown(f"{icon} `{row['Feature']}` — coefficient **{row['Coefficient']:.4f}**")


# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 ─ Raw Data
# ─────────────────────────────────────────────────────────────────────────────
with tab_data:
    st.markdown("<div class='section-header'>Dataset Explorer</div>", unsafe_allow_html=True)
    col_filter, col_sort = st.columns([3, 1])
    with col_filter:
        search = st.text_input("🔍 Filter by study method or gender",
                               placeholder="e.g.  notes  or  female")
    with col_sort:
        sort_col = st.selectbox("Sort by", options=df.columns.tolist(),
                                index=df.columns.tolist().index("overall_score"))

    view_df = df.copy()
    if search.strip():
        mask = view_df.apply(lambda r: r.astype(str).str.contains(
            search.strip(), case=False).any(), axis=1)
        view_df = view_df[mask]
    view_df = view_df.sort_values(sort_col, ascending=False)

    st.caption(f"Showing {len(view_df):,} of {len(df):,} rows")
    st.dataframe(view_df.reset_index(drop=True), use_container_width=True, height=460)

    # Download
    csv_bytes = view_df.to_csv(index=False).encode()
    st.download_button("⬇️  Download filtered CSV", data=csv_bytes,
                       file_name="student_performance_filtered.csv",
                       mime="text/csv")
