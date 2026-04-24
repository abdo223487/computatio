import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, classification_report,
    confusion_matrix, ConfusionMatrixDisplay
)
from sklearn.model_selection import cross_val_score
from scipy.sparse import issparse

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Loan Default Predictor — SVM",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'Syne', sans-serif;
}

.stApp {
    background: #0a0e1a;
    color: #e8eaf6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(160deg, #0d1226 0%, #111827 100%);
    border-right: 1px solid #1e2a45;
}

[data-testid="stSidebar"] * {
    color: #c9d1e9 !important;
}

/* Headers */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 800 !important;
    font-size: 2.6rem !important;
    background: linear-gradient(90deg, #60a5fa, #a78bfa, #34d399);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.2rem !important;
}

h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #93c5fd !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(135deg, #111827, #1a2540);
    border: 1px solid #1e3a5f;
    border-radius: 12px;
    padding: 16px !important;
}

[data-testid="stMetricValue"] {
    font-family: 'Space Mono', monospace !important;
    font-size: 1.8rem !important;
    color: #60a5fa !important;
}

[data-testid="stMetricLabel"] {
    color: #94a3b8 !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2563eb, #7c3aed);
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.4) !important;
}

.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(99,102,241,0.6) !important;
}

/* Inputs */
.stNumberInput input, .stSelectbox select, div[data-baseweb="select"] {
    background: #111827 !important;
    border: 1px solid #1e3a5f !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* Info/success/error boxes */
.stAlert {
    border-radius: 12px !important;
}

/* Prediction result card */
.result-card {
    padding: 24px;
    border-radius: 16px;
    text-align: center;
    margin: 20px 0;
    font-family: 'Space Mono', monospace;
}

.result-safe {
    background: linear-gradient(135deg, #052e16, #064e3b);
    border: 2px solid #34d399;
    color: #6ee7b7;
}

.result-risk {
    background: linear-gradient(135deg, #1c0a0a, #450a0a);
    border: 2px solid #f87171;
    color: #fca5a5;
}

.result-title {
    font-size: 2rem;
    font-weight: 700;
    margin-bottom: 8px;
}

.result-sub {
    font-size: 0.95rem;
    opacity: 0.8;
}

/* Tab styling */
.stTabs [data-baseweb="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #64748b !important;
}

.stTabs [aria-selected="true"] {
    color: #60a5fa !important;
    border-bottom-color: #60a5fa !important;
}

/* Code blocks */
code {
    font-family: 'Space Mono', monospace !important;
    background: #111827 !important;
    color: #34d399 !important;
}

/* Progress bar */
.stProgress > div > div {
    background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
}

/* Divider */
hr {
    border-color: #1e2a45 !important;
}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CACHING: LOAD & TRAIN
# ─────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_and_train(csv_path: str):
    """Full pipeline: load → clean → preprocess → PCA → SVM."""
    df = pd.read_csv(csv_path)

    # ── Split ──
    X = df.drop(columns=["TARGET", "SK_ID_CURR"])
    y = df["TARGET"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ── Clean ──
    X_train_c, X_test_c = X_train.copy(), X_test.copy()
    X_train_c["DAYS_EMPLOYED"] = X_train_c["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_test_c["DAYS_EMPLOYED"]  = X_test_c["DAYS_EMPLOYED"].replace(365243, np.nan)
    X_train_c["CODE_GENDER"] = X_train_c["CODE_GENDER"].replace("XNA", np.nan)
    X_test_c["CODE_GENDER"]  = X_test_c["CODE_GENDER"].replace("XNA", np.nan)

    num_cols = X_train_c.select_dtypes(include=["int64","float64"]).columns.tolist()
    cat_cols = X_train_c.select_dtypes(include=["object"]).columns.tolist()

    # Clip outliers
    for col in num_cols:
        Q1, Q3 = X_train_c[col].quantile(0.25), X_train_c[col].quantile(0.75)
        IQR = Q3 - Q1
        lo, hi = Q1 - 1.5*IQR, Q3 + 1.5*IQR
        X_train_c[col] = X_train_c[col].clip(lo, hi)
        X_test_c[col]  = X_test_c[col].clip(lo, hi)

    # ── Preprocess pipeline ──
    num_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("sc",  StandardScaler())
    ])
    cat_pipe = Pipeline([
        ("imp", SimpleImputer(strategy="most_frequent")),
        ("enc", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    preprocessor = ColumnTransformer([
        ("num", num_pipe, num_cols),
        ("cat", cat_pipe, cat_cols)
    ])

    X_tr_proc = preprocessor.fit_transform(X_train_c)
    X_te_proc = preprocessor.transform(X_test_c)

    # ── PCA ──
    pca = PCA(n_components=0.95, random_state=42)
    X_tr_pca = pca.fit_transform(X_tr_proc)
    X_te_pca = pca.transform(X_te_proc)

    # ── SVM (5 % subsample for speed) ──
    X_sm, _, y_sm, _ = train_test_split(
        X_tr_pca, y_train, train_size=0.05, stratify=y_train, random_state=42
    )
    svm = SVC(kernel="linear", C=1, class_weight="balanced",
              random_state=42, probability=True)
    svm.fit(X_sm, y_sm)

    return (df, X_train, X_test, y_train, y_test,
            X_train_c, X_test_c, num_cols, cat_cols,
            preprocessor, pca, svm,
            X_tr_pca, X_te_pca)


# ─────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Loan Default Predictor")
    st.markdown("---")
    st.markdown("### 📂 Load Dataset")
    csv_path = st.text_input(
        "Path to application_data.csv",
        value=r"C:\Users\Lenovo\Downloads\archive\application_data.csv",
        help="Absolute path to the CSV file on your machine"
    )
    load_btn = st.button("🚀 Load & Train Model", use_container_width=True)
    st.markdown("---")
    st.markdown("### 📌 Navigation")
    st.markdown("""
    - **Overview** — Dataset summary
    - **EDA** — Visualizations
    - **Model** — Training results
    - **Predict** — Live inference
    - **Pipeline** — Architecture
    """)
    st.markdown("---")
    st.markdown(
        "<div style='font-size:0.75rem;color:#475569;'>Alexandria University · CS Dept<br>Data Computation Spring'26</div>",
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────────────
# STATE
# ─────────────────────────────────────────────────────────────
if "trained" not in st.session_state:
    st.session_state.trained = False

if load_btn:
    if not os.path.exists(csv_path):
        st.error(f"File not found: `{csv_path}`")
    else:
        with st.spinner("⏳ Loading data & training model — this may take a minute…"):
            results = load_and_train(csv_path)
            (st.session_state.df,
             st.session_state.X_train, st.session_state.X_test,
             st.session_state.y_train, st.session_state.y_test,
             st.session_state.X_train_c, st.session_state.X_test_c,
             st.session_state.num_cols, st.session_state.cat_cols,
             st.session_state.preprocessor,
             st.session_state.pca, st.session_state.svm,
             st.session_state.X_tr_pca, st.session_state.X_te_pca) = results
            st.session_state.trained = True
        st.success("✅ Model trained and ready!")


# ─────────────────────────────────────────────────────────────
# HERO HEADER
# ─────────────────────────────────────────────────────────────
st.markdown("# 🏦 Loan Default Predictor")
st.markdown(
    "<p style='color:#64748b;font-size:1.05rem;margin-top:-8px;'>"
    "SVM · PCA · Home Credit Default Risk · Alexandria University — Spring 2026"
    "</p>", unsafe_allow_html=True
)
st.markdown("---")

if not st.session_state.trained:
    st.info("👈  Enter the CSV path in the sidebar and click **Load & Train Model** to begin.")
    st.stop()


# ─────────────────────────────────────────────────────────────
# SHORTCUT ALIASES
# ─────────────────────────────────────────────────────────────
df           = st.session_state.df
X_train      = st.session_state.X_train
X_test       = st.session_state.X_test
y_train      = st.session_state.y_train
y_test       = st.session_state.y_test
X_train_c    = st.session_state.X_train_c
X_test_c     = st.session_state.X_test_c
num_cols     = st.session_state.num_cols
cat_cols     = st.session_state.cat_cols
preprocessor = st.session_state.preprocessor
pca          = st.session_state.pca
svm          = st.session_state.svm
X_tr_pca     = st.session_state.X_tr_pca
X_te_pca     = st.session_state.X_te_pca

df_train_orig = df.loc[X_train.index].copy()
df_train_orig["TARGET"] = y_train.values


# ─────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🔍 EDA", "🤖 Model Results", "🎯 Predict", "⚙️ Pipeline"
])


# ══════════════════════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════════════════════
with tab1:
    st.markdown("## 📊 Dataset Overview")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Records",    f"{df.shape[0]:,}")
    c2.metric("Total Features",   f"{df.shape[1]:,}")
    c3.metric("Default Rate",     f"{df['TARGET'].mean()*100:.1f}%")
    c4.metric("Class Imbalance",  f"{int(df['TARGET'].value_counts()[0]/df['TARGET'].value_counts()[1])}:1")

    st.markdown("### 🗂️ Feature Breakdown")
    c1, c2, c3 = st.columns(3)
    c1.metric("Numerical Features",   len(num_cols))
    c2.metric("Categorical Features", len(cat_cols))
    c3.metric("PCA Components",       X_tr_pca.shape[1])

    st.markdown("### 📋 Sample Data")
    st.dataframe(df.head(10), use_container_width=True)

    st.markdown("### ❓ Missing Values (Top 20)")
    miss = df.isnull().sum()
    miss_pct = (miss / len(df) * 100).round(2)
    miss_df = pd.DataFrame({"Missing Count": miss, "Missing %": miss_pct})
    miss_df = miss_df[miss_df["Missing Count"] > 0].sort_values("Missing %", ascending=False)
    st.dataframe(miss_df.head(20), use_container_width=True)

    st.markdown("### 📐 Train / Test Split")
    c1, c2 = st.columns(2)
    c1.metric("Training Rows", f"{X_train.shape[0]:,}")
    c2.metric("Test Rows",     f"{X_test.shape[0]:,}")


# ══════════════════════════════════════════════════════════════
# TAB 2 — EDA
# ══════════════════════════════════════════════════════════════
with tab2:
    st.markdown("## 🔍 Exploratory Data Analysis")
    st.caption("All plots are computed on the training set only to avoid data leakage.")

    sns.set_theme(style="darkgrid", palette="muted")
    DARK_BG = "#0a0e1a"

    def dark_fig(w=12, h=5):
        fig, ax = plt.subplots(figsize=(w, h))
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        ax.tick_params(colors="#94a3b8")
        ax.xaxis.label.set_color("#94a3b8")
        ax.yaxis.label.set_color("#94a3b8")
        ax.title.set_color("#e2e8f0")
        for spine in ax.spines.values():
            spine.set_edgecolor("#1e2a45")
        return fig, ax

    # Plot 1 — Target Distribution
    st.markdown("### 1. Target Class Distribution")
    fig, ax = dark_fig(7, 4)
    counts = df_train_orig["TARGET"].value_counts()
    bars = ax.bar(["No Default (0)", "Default (1)"], counts.values,
                  color=["#3b82f6", "#ef4444"], edgecolor="none", width=0.5)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_x()+b.get_width()/2, b.get_height()+500,
                f"{v:,}\n({v/len(df_train_orig)*100:.1f}%)",
                ha="center", fontsize=10, color="#e2e8f0", fontweight="bold")
    ax.set_title("Target Variable Distribution (Training Set)", fontsize=13, fontweight="bold")
    ax.set_ylim(0, counts.max()*1.2)
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Plot 2 — Age Distribution
    st.markdown("### 2. Age Distribution by Default Status")
    fig, ax = dark_fig(13, 5)
    df_train_orig["AGE_YEARS"] = (-df_train_orig["DAYS_BIRTH"]/365).round(1)
    for tgt, col, lbl in [(0,"#3b82f6","No Default"),(1,"#ef4444","Default")]:
        sub = df_train_orig[df_train_orig["TARGET"]==tgt]["AGE_YEARS"]
        ax.hist(sub, bins=40, alpha=0.6, color=col, label=lbl, edgecolor="none")
    ax.set_title("Age Distribution by Default Status", fontsize=13, fontweight="bold")
    ax.set_xlabel("Age (years)")
    ax.set_ylabel("Count")
    ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Plot 3 — Income Boxplot
    st.markdown("### 3. Income Distribution by Default Status")
    fig, ax = dark_fig(11, 5)
    df_inc = df_train_orig[df_train_orig["AMT_INCOME_TOTAL"] <
                            df_train_orig["AMT_INCOME_TOTAL"].quantile(0.99)].copy()
    df_inc["Target Label"] = df_inc["TARGET"].map({0:"No Default",1:"Default"})
    sns.boxplot(data=df_inc, x="Target Label", y="AMT_INCOME_TOTAL",
                palette={"No Default":"#3b82f6","Default":"#ef4444"}, ax=ax)
    ax.set_title("Income Distribution by Default Status", fontsize=13, fontweight="bold")
    ax.set_ylabel("Annual Income")
    ax.set_xlabel("")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    # Plot 4 — Scatter Credit vs Goods
    st.markdown("### 4. Credit Amount vs Goods Price")
    fig, ax = dark_fig(13, 5)
    sample = df_train_orig.dropna(subset=["AMT_CREDIT","AMT_GOODS_PRICE"]).sample(
        min(4000, len(df_train_orig)), random_state=42)
    sc = ax.scatter(sample["AMT_GOODS_PRICE"], sample["AMT_CREDIT"],
                    c=sample["TARGET"], cmap="coolwarm", alpha=0.35, s=8)
    plt.colorbar(sc, ax=ax, label="Target")
    ax.set_title("Credit Amount vs Goods Price", fontsize=13, fontweight="bold")
    ax.set_xlabel("Goods Price")
    ax.set_ylabel("Credit Amount")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    col_a, col_b = st.columns(2)

    # Plot 5 — Default Rate by Education
    with col_a:
        st.markdown("### 5. Default Rate by Education Type")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        edu = (df_train_orig.groupby("NAME_EDUCATION_TYPE")["TARGET"]
               .mean().sort_values(ascending=False))
        bars = ax.barh(edu.index, edu.values*100,
                       color=sns.color_palette("rocket_r", len(edu)))
        for b, v in zip(bars, edu.values*100):
            ax.text(b.get_width()+0.1, b.get_y()+b.get_height()/2,
                    f"{v:.1f}%", va="center", fontsize=9, color="#e2e8f0")
        ax.set_xlabel("Default Rate (%)")
        ax.set_title("Default Rate by Education", fontsize=11, fontweight="bold")
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    # Plot 6 — Default Rate by Income Type
    with col_b:
        st.markdown("### 6. Default Rate by Income Type")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        inc = (df_train_orig.groupby("NAME_INCOME_TYPE")["TARGET"]
               .mean().sort_values(ascending=False))
        ax.bar(inc.index, inc.values*100,
               color=sns.color_palette("mako_r", len(inc)))
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Income Type", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=30)
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Plot 7 — Correlation Heatmap
    st.markdown("### 7. Correlation Heatmap — Key Features")
    key_cols = ["TARGET","AMT_INCOME_TOTAL","AMT_CREDIT","AMT_ANNUITY",
                "AMT_GOODS_PRICE","DAYS_BIRTH","DAYS_EMPLOYED",
                "EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3",
                "REGION_RATING_CLIENT","CNT_CHILDREN","CNT_FAM_MEMBERS"]
    avail = [c for c in key_cols if c in df_train_orig.columns]
    fig, ax = plt.subplots(figsize=(12, 9))
    fig.patch.set_facecolor(DARK_BG)
    ax.set_facecolor("#111827")
    corr = df_train_orig[avail].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, vmin=-1, vmax=1, linewidths=0.4, ax=ax, annot_kws={"size":8})
    ax.set_title("Correlation Heatmap", fontsize=13, fontweight="bold", color="#e2e8f0")
    ax.tick_params(colors="#94a3b8")
    plt.tight_layout()
    st.pyplot(fig)

    st.markdown("---")

    col_c, col_d = st.columns(2)

    # Plot 8 — Employment Length
    with col_c:
        st.markdown("### 8. Employment Length Distribution")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        df_emp = df_train_orig.copy()
        df_emp["EMP_YEARS"] = (-df_emp["DAYS_EMPLOYED"]/365).clip(upper=40)
        for tgt, col, lbl in [(0,"#3b82f6","No Default"),(1,"#ef4444","Default")]:
            sub = df_emp[df_emp["TARGET"]==tgt]["EMP_YEARS"]
            ax.hist(sub, bins=35, alpha=0.6, color=col, label=lbl, edgecolor="none")
        ax.set_xlabel("Years Employed")
        ax.set_ylabel("Count")
        ax.set_title("Employment Length by Default", fontsize=11, fontweight="bold")
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    # Plot 9 — Gender vs Default
    with col_d:
        st.markdown("### 9. Loan Count & Default by Gender")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        gdf = df_train_orig.groupby(["CODE_GENDER","TARGET"]).size().unstack(fill_value=0)
        gdf.columns = ["No Default","Default"]
        gdf[["No Default","Default"]].plot(kind="bar", ax=ax,
            color=["#3b82f6","#ef4444"], edgecolor="none")
        gdf["DR"] = (gdf["Default"]/gdf.sum(axis=1)*100).round(2)
        for i,(idx,row) in enumerate(gdf.iterrows()):
            ax.text(i, row["No Default"]+row["Default"]+200,
                    f"DR: {row['DR']:.1f}%", ha="center", fontsize=9, color="#e2e8f0")
        ax.set_title("Gender vs Default", fontsize=11, fontweight="bold")
        ax.set_xlabel("Gender")
        ax.tick_params(axis="x", rotation=0)
        ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    col_e, col_f = st.columns(2)

    # Plot 10 — DAYS_EMPLOYED Anomaly
    with col_e:
        st.markdown("### 10. DAYS_EMPLOYED Anomaly Detection")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        an = (df_train_orig["DAYS_EMPLOYED"]==365243).sum()
        nm = (df_train_orig["DAYS_EMPLOYED"]!=365243).sum()
        bars = ax.bar(["Normal\nEmployment","Anomalous\n(365243)"],[nm,an],
                      color=["#22c55e","#ef4444"], edgecolor="none", width=0.45)
        for b, v in zip(bars,[nm,an]):
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+200,
                    f"{v:,}\n({v/len(df_train_orig)*100:.1f}%)",
                    ha="center", fontweight="bold", color="#e2e8f0")
        ax.set_title("DAYS_EMPLOYED Anomaly", fontsize=11, fontweight="bold")
        ax.set_ylabel("Count")
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    # Plot 11 — Weekday Default Rate
    with col_f:
        st.markdown("### 11. Default Rate by Weekday")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        wk_order = ["MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"]
        wk = (df_train_orig.groupby("WEEKDAY_APPR_PROCESS_START")["TARGET"]
              .mean()*100).reindex(wk_order)
        ax.bar(wk.index, wk.values, color=sns.color_palette("rocket", len(wk)), edgecolor="none")
        for i, v in enumerate(wk.values):
            ax.text(i, v+0.05, f"{v:.1f}%", ha="center", fontsize=9, color="#e2e8f0")
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Default Rate by Weekday", fontsize=11, fontweight="bold")
        ax.tick_params(axis="x", rotation=25)
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    col_g, col_h = st.columns(2)

    # Plot 12 — Children vs Default
    with col_g:
        st.markdown("### 12. Default Rate by Number of Children")
        fig, ax = dark_fig(8, 5)
        fig.patch.set_facecolor(DARK_BG)
        ax.set_facecolor("#111827")
        ch = (df_train_orig[df_train_orig["CNT_CHILDREN"]<=5]
              .groupby("CNT_CHILDREN")["TARGET"].mean()*100)
        ax.bar(ch.index, ch.values, color=sns.color_palette("YlOrRd", len(ch)), edgecolor="none")
        for i, v in enumerate(ch.values):
            ax.text(ch.index[i], v+0.2, f"{v:.1f}%", ha="center", fontsize=9, color="#e2e8f0")
        ax.set_xlabel("Number of Children")
        ax.set_ylabel("Default Rate (%)")
        ax.set_title("Children vs Default Rate", fontsize=11, fontweight="bold")
        ax.title.set_color("#e2e8f0")
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)

    # Plot 13 — EXT_SOURCE Violin
    with col_h:
        st.markdown("### 13. EXT_SOURCE Scores by Default Status")
        fig, axes = plt.subplots(1, 3, figsize=(12, 5))
        fig.patch.set_facecolor(DARK_BG)
        ext_cols = ["EXT_SOURCE_1","EXT_SOURCE_2","EXT_SOURCE_3"]
        colors   = ["#22c55e","#f59e0b","#a78bfa"]
        for ax_, col, color in zip(axes, ext_cols, colors):
            ax_.set_facecolor("#111827")
            sub = df_train_orig[[col,"TARGET"]].dropna()
            sub["Label"] = sub["TARGET"].map({0:"No Default",1:"Default"})
            sns.violinplot(data=sub, x="Label", y=col,
                           palette={"No Default":"#475569","Default":color}, ax=ax_)
            ax_.set_title(col, fontsize=10, fontweight="bold", color="#e2e8f0")
            ax_.set_xlabel("")
            ax_.tick_params(colors="#94a3b8")
            for sp in ax_.spines.values():
                sp.set_edgecolor("#1e2a45")
        plt.suptitle("EXT_SOURCE Scores vs Default", fontsize=12, fontweight="bold", color="#e2e8f0")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")

    # Plot 14 — PCA
    st.markdown("### 14. PCA: Cumulative Explained Variance")
    fig, ax = dark_fig(10, 5)
    ax.plot(np.cumsum(pca.explained_variance_ratio_), color="#60a5fa", linewidth=2.5)
    ax.axhline(y=0.95, color="#ef4444", linestyle="--", linewidth=1.5, label="95% Threshold")
    ax.fill_between(range(len(pca.explained_variance_ratio_)),
                    np.cumsum(pca.explained_variance_ratio_),
                    alpha=0.15, color="#60a5fa")
    ax.set_xlabel("Number of Components")
    ax.set_ylabel("Cumulative Explained Variance")
    ax.set_title("PCA Analysis: Variance vs Components", fontsize=13, fontweight="bold")
    ax.legend(facecolor="#111827", labelcolor="#e2e8f0")
    plt.tight_layout()
    st.pyplot(fig)


# ══════════════════════════════════════════════════════════════
# TAB 3 — MODEL RESULTS
# ══════════════════════════════════════════════════════════════
with tab3:
    st.markdown("## 🤖 SVM Model Results")

    y_pred   = svm.predict(X_te_pca)
    y_scores = svm.decision_function(X_te_pca)

    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec  = recall_score(y_test, y_pred)
    f1   = f1_score(y_test, y_pred)
    auc  = roc_auc_score(y_test, y_scores)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Accuracy",  f"{acc:.4f}")
    c2.metric("Precision", f"{prec:.4f}")
    c3.metric("Recall",    f"{rec:.4f}")
    c4.metric("F1-Score",  f"{f1:.4f}")
    c5.metric("ROC-AUC",   f"{auc:.4f}")

    st.markdown("---")

    col_r, col_cm = st.columns([1, 1])

    with col_r:
        st.markdown("### Classification Report")
        report = classification_report(y_test, y_pred,
                    target_names=["No Default","Default"], output_dict=True)
        report_df = pd.DataFrame(report).T.round(4)
        st.dataframe(report_df, use_container_width=True)

    with col_cm:
        st.markdown("### Confusion Matrix")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#111827")
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=["No Default","Default"],
                    yticklabels=["No Default","Default"],
                    linewidths=0.5, ax=ax)
        ax.set_title("Confusion Matrix — SVM", fontsize=13, fontweight="bold", color="#e2e8f0")
        ax.set_xlabel("Predicted", color="#94a3b8")
        ax.set_ylabel("Actual",    color="#94a3b8")
        ax.tick_params(colors="#94a3b8")
        plt.tight_layout()
        st.pyplot(fig)

    st.markdown("---")
    st.markdown("### 🧾 Model Configuration")
    cfg = {
        "Kernel":       "linear",
        "C":            1.0,
        "class_weight": "balanced",
        "Training size": "5% of training set (stratified)",
        "PCA components": X_tr_pca.shape[1],
        "Explained variance": f"{pca.explained_variance_ratio_.sum():.4f}",
    }
    st.table(pd.DataFrame.from_dict(cfg, orient="index", columns=["Value"]))

    # Save model button
    if st.button("💾 Save Model (svm_final_model.pkl)"):
        joblib.dump(svm, "svm_final_model.pkl")
        st.success("Model saved as `svm_final_model.pkl`")


# ══════════════════════════════════════════════════════════════
# TAB 4 — LIVE PREDICTION
# ══════════════════════════════════════════════════════════════
with tab4:
    st.markdown("## 🎯 Live Loan Default Prediction")
    st.markdown("Fill in the applicant's details below and click **Predict** to get an instant risk assessment.")
    st.markdown("---")

    with st.form("prediction_form"):
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("**💰 Financial Info**")
            amt_income = st.number_input("Annual Income", min_value=0, value=150000, step=5000)
            amt_credit = st.number_input("Credit Amount", min_value=0, value=500000, step=10000)
            amt_annuity = st.number_input("Loan Annuity", min_value=0, value=25000, step=1000)
            amt_goods  = st.number_input("Goods Price",  min_value=0, value=450000, step=10000)

        with col2:
            st.markdown("**👤 Personal Info**")
            days_birth    = st.number_input("Age (years)", min_value=18, max_value=90, value=35)
            days_employed = st.number_input("Years Employed", min_value=0, max_value=50, value=5)
            gender = st.selectbox("Gender", ["M", "F"])
            education = st.selectbox("Education Type", [
                "Higher education", "Secondary / secondary special",
                "Incomplete higher", "Lower secondary", "Academic degree"
            ])
            income_type = st.selectbox("Income Type", [
                "Working", "Commercial associate", "Pensioner",
                "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"
            ])
            cnt_children = st.number_input("Number of Children", min_value=0, max_value=20, value=0)
            family_status = st.selectbox("Family Status", [
                "Married", "Single / not married", "Civil marriage", "Separated", "Widow"
            ])

        with col3:
            st.markdown("**📊 External Scores**")
            ext1 = st.slider("EXT_SOURCE_1", 0.0, 1.0, 0.5, 0.01)
            ext2 = st.slider("EXT_SOURCE_2", 0.0, 1.0, 0.5, 0.01)
            ext3 = st.slider("EXT_SOURCE_3", 0.0, 1.0, 0.5, 0.01)
            st.markdown("**🏠 Housing & Region**")
            region_rating = st.selectbox("Region Rating", [1, 2, 3])
            housing_type  = st.selectbox("Housing Type", [
                "House / apartment", "Rented apartment", "Municipal apartment",
                "With parents", "Co-op apartment", "Office apartment"
            ])
            contract_type = st.selectbox("Contract Type", ["Cash loans", "Revolving loans"])
            weekday = st.selectbox("Application Weekday", [
                "MONDAY","TUESDAY","WEDNESDAY","THURSDAY","FRIDAY","SATURDAY","SUNDAY"
            ])

        submitted = st.form_submit_button("⚡ Predict Risk", use_container_width=True)

    if submitted:
        # ── Build a row matching training columns ──
        # We build a DataFrame with one row, filling all training columns with defaults,
        # then override with user inputs.
        sample_row = X_train.iloc[[0]].copy()
        sample_row[:] = np.nan

        # Numerical overrides
        col_map = {
            "AMT_INCOME_TOTAL":    amt_income,
            "AMT_CREDIT":          amt_credit,
            "AMT_ANNUITY":         amt_annuity,
            "AMT_GOODS_PRICE":     amt_goods,
            "DAYS_BIRTH":          -(days_birth * 365),
            "DAYS_EMPLOYED":       -(days_employed * 365),
            "EXT_SOURCE_1":        ext1,
            "EXT_SOURCE_2":        ext2,
            "EXT_SOURCE_3":        ext3,
            "REGION_RATING_CLIENT": region_rating,
            "CNT_CHILDREN":        cnt_children,
        }
        for col, val in col_map.items():
            if col in sample_row.columns:
                sample_row[col] = val

        # Categorical overrides
        cat_map = {
            "CODE_GENDER":               gender,
            "NAME_EDUCATION_TYPE":       education,
            "NAME_INCOME_TYPE":          income_type,
            "NAME_FAMILY_STATUS":        family_status,
            "NAME_HOUSING_TYPE":         housing_type,
            "NAME_CONTRACT_TYPE":        contract_type,
            "WEEKDAY_APPR_PROCESS_START": weekday,
        }
        for col, val in cat_map.items():
            if col in sample_row.columns:
                sample_row[col] = val

        # Fix anomalous value
        if "DAYS_EMPLOYED" in sample_row.columns:
            sample_row["DAYS_EMPLOYED"] = sample_row["DAYS_EMPLOYED"].replace(365243, np.nan)

        # Preprocess → PCA → Predict
        proc = preprocessor.transform(sample_row)
        pca_feat = pca.transform(proc)
        prediction  = svm.predict(pca_feat)[0]
        confidence  = abs(svm.decision_function(pca_feat)[0])
        proba = svm.predict_proba(pca_feat)[0]

        # ── Display result ──
        if prediction == 0:
            st.markdown("""
            <div class="result-card result-safe">
                <div class="result-title">✅ LOW RISK — NO DEFAULT</div>
                <div class="result-sub">This applicant is predicted to repay the loan successfully.</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="result-card result-risk">
                <div class="result-title">⚠️ HIGH RISK — DEFAULT LIKELY</div>
                <div class="result-sub">This applicant is predicted to default. Further review recommended.</div>
            </div>
            """, unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        c1.metric("Prediction",       "Default" if prediction==1 else "No Default")
        c2.metric("Confidence Score", f"{confidence:.3f}")
        c3.metric("Default Probability", f"{proba[1]*100:.1f}%")

        # Mini bar chart of probabilities
        fig, ax = plt.subplots(figsize=(5, 2.5))
        fig.patch.set_facecolor("#0a0e1a")
        ax.set_facecolor("#111827")
        ax.barh(["No Default","Default"], [proba[0], proba[1]],
                color=["#3b82f6","#ef4444"], edgecolor="none")
        ax.set_xlim(0, 1)
        ax.set_title("Predicted Probabilities", color="#e2e8f0", fontsize=10)
        ax.tick_params(colors="#94a3b8")
        for sp in ax.spines.values():
            sp.set_edgecolor("#1e2a45")
        plt.tight_layout()
        st.pyplot(fig)


# ══════════════════════════════════════════════════════════════
# TAB 5 — PIPELINE
# ══════════════════════════════════════════════════════════════
with tab5:
    st.markdown("## ⚙️ Full Pipeline Architecture")
    st.caption("Click any step to expand its details.")

    n_pca = X_tr_pca.shape[1]
    reduction_pct = round((1 - n_pca / 243) * 100, 1)

    pipeline_html = f"""
    <style>
      @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@600;800&display=swap');
      .pipe-wrap {{ font-family: 'Syne', sans-serif; padding: 8px 0 24px; background: transparent; }}
      @keyframes pulse-dot {{ 0%,100% {{ transform:scale(1);opacity:.9; }} 50% {{ transform:scale(1.6);opacity:.4; }} }}
      @keyframes slide-down {{ from {{ opacity:0;transform:translateY(-8px); }} to {{ opacity:1;transform:translateY(0); }} }}
      @keyframes shimmer {{ 0% {{ background-position:-400px 0; }} 100% {{ background-position:400px 0; }} }}
      .pipe-step {{ position:relative;border-radius:16px;border:1.5px solid transparent;margin-bottom:6px;cursor:pointer;transition:border-color .25s,box-shadow .25s,transform .18s;background:#111827;overflow:hidden; }}
      .pipe-step:hover {{ transform:translateX(4px);box-shadow:0 4px 24px rgba(0,0,0,.5); }}
      .pipe-step.active {{ border-color:var(--accent);box-shadow:0 0 0 3px color-mix(in srgb,var(--accent) 20%,transparent); }}
      .pipe-step::before {{ content:'';position:absolute;inset:0;background:linear-gradient(90deg,transparent 0%,rgba(255,255,255,.05) 50%,transparent 100%);background-size:400px 100%;opacity:0;transition:opacity .3s;pointer-events:none; }}
      .pipe-step:hover::before {{ opacity:1;animation:shimmer .8s linear infinite; }}
      .step-header {{ display:flex;align-items:center;gap:14px;padding:16px 20px; }}
      .step-badge {{ width:36px;height:36px;border-radius:10px;display:flex;align-items:center;justify-content:center;font-family:'Space Mono',monospace;font-size:13px;font-weight:700;flex-shrink:0;background:color-mix(in srgb,var(--accent) 18%,transparent);color:var(--accent);border:1px solid color-mix(in srgb,var(--accent) 30%,transparent); }}
      .step-title {{ font-size:15px;font-weight:600;color:#e2e8f0;flex:1; }}
      .step-meta {{ font-family:'Space Mono',monospace;font-size:11px;color:#475569;margin-left:auto; }}
      .step-arrow {{ font-size:12px;color:#475569;transition:transform .25s; }}
      .pipe-step.active .step-arrow {{ transform:rotate(90deg); }}
      .step-body {{ display:none;padding:0 20px 18px 70px;animation:slide-down .2s ease; }}
      .pipe-step.active .step-body {{ display:block; }}
      .detail-row {{ display:flex;align-items:flex-start;gap:10px;padding:7px 0;border-bottom:1px solid rgba(255,255,255,.05);font-size:13px;color:#94a3b8; }}
      .detail-row:last-child {{ border-bottom:none; }}
      .detail-icon {{ color:var(--accent);font-size:15px;flex-shrink:0;margin-top:1px; }}
      .detail-label {{ color:#cbd5e1;font-weight:600;min-width:180px; }}
      .tag {{ display:inline-block;padding:2px 8px;border-radius:6px;font-family:'Space Mono',monospace;font-size:11px;background:color-mix(in srgb,var(--accent) 15%,transparent);color:var(--accent);margin:2px 3px; }}
      .pipe-connector {{ display:flex;align-items:center;justify-content:flex-start;gap:10px;margin:4px 0;padding-left:38px; }}
      .conn-line {{ width:2px;height:28px;background:linear-gradient(to bottom,var(--from-color),var(--to-color));border-radius:2px;position:relative;overflow:visible; }}
      .conn-dot {{ width:7px;height:7px;border-radius:50%;background:var(--dot-color);position:absolute;left:50%;transform:translateX(-50%);animation:pulse-dot 1.6s ease-in-out infinite; }}
      .conn-label {{ font-family:'Space Mono',monospace;font-size:10px;color:#475569; }}
      .decisions-grid {{ display:grid;grid-template-columns:1fr 1fr;gap:10px;margin-top:16px; }}
      .decision-card {{ background:#111827;border:1px solid rgba(255,255,255,.07);border-radius:12px;padding:14px 16px;transition:border-color .2s,transform .18s; }}
      .decision-card:hover {{ border-color:rgba(255,255,255,.18);transform:translateY(-2px); }}
      .decision-title {{ font-size:13px;font-weight:600;color:#93c5fd;margin-bottom:5px; }}
      .decision-text {{ font-size:12px;color:#64748b;line-height:1.5; }}
      .uni-badge {{ margin-top:24px;background:linear-gradient(135deg,#0f172a,#1e293b);border:1px solid rgba(255,255,255,.08);border-radius:14px;padding:18px 22px;display:flex;align-items:center;gap:16px; }}
      .uni-icon {{ font-size:28px; }}
      .uni-text {{ font-size:13px;color:#94a3b8;line-height:1.6; }}
      .uni-text strong {{ color:#e2e8f0; }}
    </style>

    <div class="pipe-wrap">
      <div class="pipe-step" style="--accent:#60a5fa" onclick="this.classList.toggle('active')">
        <div class="step-header">
          <div class="step-badge">00</div>
          <div class="step-title">Train / Test Split</div>
          <div class="step-meta">80% / 20%</div>
          <div class="step-arrow">&#9658;</div>
        </div>
        <div class="step-body">
          <div class="detail-row"><span class="detail-icon">&#9881;</span><span class="detail-label">Strategy</span><span>Stratified split preserving class ratio</span></div>
          <div class="detail-row"><span class="detail-icon">&#9881;</span><span class="detail-label">Random state</span><span><span class="tag">42</span></span></div>
          <div class="detail-row"><span class="detail-icon">&#9881;</span><span class="detail-label">Why first?</span><span>Prevents any data leakage into cleaning or scaling steps</span></div>
        </div>
      </div>

      <div class="pipe-connector">
        <div class="conn-line" style="--from-color:#60a5fa;--to-color:#34d399">
          <div class="conn-dot" style="top:4px;--dot-color:#60a5fa"></div>
          <div class="conn-dot" style="top:16px;animation-delay:.5s;--dot-color:#4ade80"></div>
        </div>
        <div class="conn-label">307,511 rows</div>
      </div>

      <div class="pipe-step" style="--accent:#34d399" onclick="this.classList.toggle('active')">
        <div class="step-header">
          <div class="step-badge">01</div>
          <div class="step-title">Data Cleaning</div>
          <div class="step-meta">3 operations</div>
          <div class="step-arrow">&#9658;</div>
        </div>
        <div class="step-body">
          <div class="detail-row"><span class="detail-icon">&#128295;</span><span class="detail-label">Anomaly fix</span><span><span class="tag">DAYS_EMPLOYED = 365243</span> &rarr; <span class="tag">NaN</span></span></div>
          <div class="detail-row"><span class="detail-icon">&#128295;</span><span class="detail-label">Gender fix</span><span><span class="tag">CODE_GENDER = XNA</span> &rarr; <span class="tag">NaN</span></span></div>
          <div class="detail-row"><span class="detail-icon">&#128295;</span><span class="detail-label">Outlier clipping</span><span>IQR-based (Q1-1.5xIQR, Q3+1.5xIQR) fitted on train only</span></div>
        </div>
      </div>

      <div class="pipe-connector">
        <div class="conn-line" style="--from-color:#34d399;--to-color:#f59e0b">
          <div class="conn-dot" style="top:4px;--dot-color:#34d399"></div>
          <div class="conn-dot" style="top:16px;animation-delay:.7s;--dot-color:#f59e0b"></div>
        </div>
        <div class="conn-label">cleaned features</div>
      </div>

      <div class="pipe-step" style="--accent:#f59e0b" onclick="this.classList.toggle('active')">
        <div class="step-header">
          <div class="step-badge">02</div>
          <div class="step-title">ColumnTransformer &mdash; Preprocessing</div>
          <div class="step-meta">&rarr; 243 features</div>
          <div class="step-arrow">&#9658;</div>
        </div>
        <div class="step-body">
          <div class="detail-row"><span class="detail-icon">&#128290;</span><span class="detail-label">Numerical pipeline</span><span>Median imputer &rarr; StandardScaler</span></div>
          <div class="detail-row"><span class="detail-icon">&#128292;</span><span class="detail-label">Categorical pipeline</span><span>Most-frequent imputer &rarr; OneHotEncoder (handle_unknown=ignore)</span></div>
          <div class="detail-row"><span class="detail-icon">&#128202;</span><span class="detail-label">Output shape</span><span>Train: <span class="tag">246,008 x 243</span> | Test: <span class="tag">61,503 x 243</span></span></div>
          <div class="detail-row"><span class="detail-icon">&#10003;</span><span class="detail-label">NaN check</span><span>0 missing values after pipeline</span></div>
        </div>
      </div>

      <div class="pipe-connector">
        <div class="conn-line" style="--from-color:#f59e0b;--to-color:#a78bfa">
          <div class="conn-dot" style="top:4px;--dot-color:#f59e0b"></div>
          <div class="conn-dot" style="top:16px;animation-delay:.4s;--dot-color:#a78bfa"></div>
        </div>
        <div class="conn-label">243 encoded features</div>
      </div>

      <div class="pipe-step" style="--accent:#a78bfa" onclick="this.classList.toggle('active')">
        <div class="step-header">
          <div class="step-badge">03</div>
          <div class="step-title">PCA &mdash; Dimensionality Reduction</div>
          <div class="step-meta">243 &rarr; {n_pca} components</div>
          <div class="step-arrow">&#9658;</div>
        </div>
        <div class="step-body">
          <div class="detail-row"><span class="detail-icon">&#128201;</span><span class="detail-label">n_components</span><span><span class="tag">0.95</span> (retain 95% variance)</span></div>
          <div class="detail-row"><span class="detail-icon">&#128201;</span><span class="detail-label">Reduced to</span><span><span class="tag">{n_pca} components</span> from 243</span></div>
          <div class="detail-row"><span class="detail-icon">&#128201;</span><span class="detail-label">Reduction ratio</span><span>{reduction_pct}% feature reduction</span></div>
          <div class="detail-row"><span class="detail-icon">&#128201;</span><span class="detail-label">Benefit</span><span>Speeds up SVM, removes noise and multicollinearity</span></div>
        </div>
      </div>

      <div class="pipe-connector">
        <div class="conn-line" style="--from-color:#a78bfa;--to-color:#f87171">
          <div class="conn-dot" style="top:4px;--dot-color:#a78bfa"></div>
          <div class="conn-dot" style="top:16px;animation-delay:.9s;--dot-color:#f87171"></div>
        </div>
        <div class="conn-label">{n_pca} PCA features</div>
      </div>

      <div class="pipe-step" style="--accent:#f87171" onclick="this.classList.toggle('active')">
        <div class="step-header">
          <div class="step-badge">04</div>
          <div class="step-title">SVM Classifier</div>
          <div class="step-meta">linear kernel &middot; C=1</div>
          <div class="step-arrow">&#9658;</div>
        </div>
        <div class="step-body">
          <div class="detail-row"><span class="detail-icon">&#129302;</span><span class="detail-label">Kernel</span><span><span class="tag">linear</span></span></div>
          <div class="detail-row"><span class="detail-icon">&#129302;</span><span class="detail-label">C (regularization)</span><span><span class="tag">1.0</span></span></div>
          <div class="detail-row"><span class="detail-icon">&#129302;</span><span class="detail-label">class_weight</span><span><span class="tag">balanced</span> handles 11:1 imbalance</span></div>
          <div class="detail-row"><span class="detail-icon">&#129302;</span><span class="detail-label">Training subset</span><span>5% stratified sample (SVM is O(n^2))</span></div>
          <div class="detail-row"><span class="detail-icon">&#129302;</span><span class="detail-label">Probability</span><span><span class="tag">True</span> enables predict_proba for UI</span></div>
        </div>
      </div>

      <div style="margin-top:28px;">
        <div style="font-size:16px;font-weight:600;color:#93c5fd;margin-bottom:12px;">Key Design Decisions</div>
        <div class="decisions-grid">
          <div class="decision-card"><div class="decision-title">Train-first split</div><div class="decision-text">Data split BEFORE any cleaning or scaling to prevent data leakage and ensure true model generalization.</div></div>
          <div class="decision-card"><div class="decision-title">Outlier clipping</div><div class="decision-text">IQR bounds computed on training data only, then applied to test to avoid test-set contamination.</div></div>
          <div class="decision-card"><div class="decision-title">DAYS_EMPLOYED fix</div><div class="decision-text">Value 365243 is a sentinel for unemployed/pensioners encoded incorrectly, replaced with NaN for honest imputation.</div></div>
          <div class="decision-card"><div class="decision-title">Balanced class weight</div><div class="decision-text">Addresses severe 11:1 class imbalance without synthetic oversampling, keeping the training set clean.</div></div>
          <div class="decision-card"><div class="decision-title">5% SVM subsample</div><div class="decision-text">SVM complexity is O(n^2). A 5% stratified sample gives strong signal while keeping runtime practical.</div></div>
          <div class="decision-card"><div class="decision-title">PCA at 95%</div><div class="decision-text">Reduces 243 encoded features to {n_pca}, eliminating multicollinearity and noise while retaining most information.</div></div>
        </div>
      </div>

      <div class="uni-badge">
        <div class="uni-icon">&#127891;</div>
        <div class="uni-text"><strong>Alexandria University &mdash; Faculty of Computers &amp; Data Science</strong><br>Data Computation &middot; Spring 2026 Final Project &middot; SVM Classification</div>
      </div>
    </div>
    """

    st.components.v1.html(pipeline_html, height=1150, scrolling=True)
