
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from io import StringIO, BytesIO
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

st.set_page_config(page_title="Insurance Policy — HR Insights & ML", layout="wide")

# ----------------------------
# Utilities
# ----------------------------
@st.cache_data
def load_default():
    # Try common locations
    import os
    for p in ["/mnt/data/Insurance.csv", "/mnt/data/insurance.csv", "Insurance.csv", "insurance.csv"]:
        if os.path.exists(p):
            df = pd.read_csv(p)
            return df
    return None

def clean_columns(df):
    df = df.copy()
    df.columns = df.columns.str.strip()
    return df

def impute_df(df):
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        if df[c].isna().any():
            mode = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode.iloc[0] if len(mode)>0 else "Missing")
    return df

def encode_fit(df):
    df = df.copy()
    encoders = {}
    for col in df.select_dtypes(include=["object", "category"]).columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = {cls: int(i) for i, cls in enumerate(le.classes_)}
    return df, encoders

def encode_transform(df, encoders):
    df = df.copy()
    for col, mapping in encoders.items():
        if col in df.columns:
            # map unseen categories to a safe default (most frequent seen during train, index 0)
            df[col] = df[col].astype(str).map(lambda x: mapping.get(x, 0))
    return df

def stratified_split(X, y):
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def train_models(X_train, y_train):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42)
    }
    for name, m in models.items():
        m.fit(X_train, y_train)
    return models

def metrics_table(models, X_train, y_train, X_test, y_test):
    rows = []
    proba_ok = len(np.unique(y_train)) == 2
    for name, m in models.items():
        ytr = m.predict(X_train)
        yte = m.predict(X_test)
        row = {
            "Model": name,
            "Train Accuracy": accuracy_score(y_train, ytr),
            "Test Accuracy": accuracy_score(y_test, yte),
            "Precision": precision_score(y_test, yte, average="weighted", zero_division=0),
            "Recall": recall_score(y_test, yte, average="weighted", zero_division=0),
            "F1": f1_score(y_test, yte, average="weighted", zero_division=0),
        }
        if proba_ok and hasattr(m, "predict_proba"):
            try:
                proba = m.predict_proba(X_test)[:,1]
                row["ROC-AUC"] = roc_auc_score(y_test, proba)
            except Exception:
                pass
        rows.append(row)
    return pd.DataFrame(rows).sort_values("Test Accuracy", ascending=False)

def plot_confmat(cm, class_labels, title):
    fig, ax = plt.subplots(figsize=(5,4))
    im = ax.imshow(cm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_xticks(range(len(class_labels)))
    ax.set_xticklabels(class_labels, rotation=45, ha="right")
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center')
    fig.tight_layout()
    return fig

def feature_importance_figure(model, feature_names, title, top_n=15):
    if not hasattr(model, "feature_importances_"):
        return None
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:top_n]
    df_imp = pd.DataFrame({"Feature": np.array(feature_names)[order], "Importance": imp[order]})
    fig = px.bar(df_imp.sort_values("Importance"), x="Importance", y="Feature", orientation="h", title=title)
    return fig

# ----------------------------
# Data Loading
# ----------------------------
st.sidebar.header("Data")
uploaded = st.sidebar.file_uploader("Upload your insurance CSV (optional)", type=["csv"])
df_default = load_default()
if uploaded:
    df_raw = pd.read_csv(uploaded)
elif df_default is not None:
    df_raw = df_default
else:
    st.error("Please upload a dataset to proceed.")
    st.stop()

df_raw = clean_columns(df_raw)

if "POLICY_STATUS" not in df_raw.columns:
    st.error("Target column 'POLICY_STATUS' not found. Please ensure the CSV has this column (whitespace will be stripped).")
    st.stop()

# Keep a pristine copy
df0 = df_raw.copy()
df = impute_df(df_raw)

# Choose numeric column to act as 'satisfaction-related' slider (flexible)
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
default_numeric = "SUM_ASSURED" if "SUM_ASSURED" in numeric_cols else (numeric_cols[0] if numeric_cols else None)

# ----------------------------
# Filters (applies to charts)
# ----------------------------
st.sidebar.header("Filters")
occ_col = "PI_OCCUPATION" if "PI_OCCUPATION" in df.columns else None
if occ_col:
    occ_options = sorted(df[occ_col].dropna().astype(str).unique().tolist())
    occ_sel = st.sidebar.multiselect("Job Role (PI_OCCUPATION)", occ_options, default=occ_options)
else:
    occ_sel = None

sat_col = st.sidebar.selectbox("Satisfaction-related numeric filter (choose any numeric column):",
                               options=numeric_cols, index=numeric_cols.index(default_numeric) if default_numeric in numeric_cols else 0)

sat_min, sat_max = float(df[sat_col].min()), float(df[sat_col].max())
sat_range = st.sidebar.slider(f"Range for {sat_col}", min_value=sat_min, max_value=sat_max, value=(sat_min, sat_max))

def apply_filters(df_):
    dff = df_.copy()
    if occ_sel is not None:
        dff = dff[dff.get("PI_OCCUPATION", "").astype(str).isin(occ_sel)]
    dff = dff[(dff[sat_col] >= sat_range[0]) & (dff[sat_col] <= sat_range[1])]
    return dff

df_f = apply_filters(df)

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3 = st.tabs(["📊 HR Insights (5 Charts)", "🤖 Train & Evaluate Models", "🔮 Predict on New Data"])

# ----------------------------
# Tab 1: HR Insights
# ----------------------------
with tab1:
    st.subheader("Interactive Insights for Policy Decisions")
    st.caption("All charts respect the sidebar filters (Job Role & Satisfaction-related numeric range).")

    # 1) Claim rate by Occupation
    if "PI_OCCUPATION" in df_f.columns:
        t1 = (df_f.groupby("PI_OCCUPATION")["POLICY_STATUS"]
              .apply(lambda s: (s==1).mean() if s.dtype!=object else (pd.factorize(s)[0]==1).mean())
              .reset_index(name="ClaimRate"))
        fig1 = px.bar(t1.sort_values("ClaimRate", ascending=False), x="PI_OCCUPATION", y="ClaimRate",
                      title="Claim Rate by Job Role (Policy Status=1 as Claim/Active)")
        st.plotly_chart(fig1, use_container_width=True)

    # 2) Claim rate by Zone
    if "ZONE" in df_f.columns:
        t2 = df_f.groupby("ZONE")["POLICY_STATUS"].apply(lambda s: (s==1).mean()).reset_index(name="ClaimRate")
        fig2 = px.bar(t2.sort_values("ClaimRate", ascending=False), x="ZONE", y="ClaimRate",
                      title="Claim Rate by Zone")
        st.plotly_chart(fig2, use_container_width=True)

    # 3) Age vs Claim rate (binned)
    if "PI_AGE" in df_f.columns:
        bins = pd.cut(df_f["PI_AGE"], bins=[0,25,35,45,55,65,100], right=False)
        t3 = df_f.groupby(bins)["POLICY_STATUS"].apply(lambda s: (s==1).mean()).reset_index(name="ClaimRate")
        fig3 = px.line(t3, x="PI_AGE", y="ClaimRate", title="Claim Rate across Age Bands")
        st.plotly_chart(fig3, use_container_width=True)

    # 4) Sum Assured buckets vs Claim rate
    if "SUM_ASSURED" in df_f.columns:
        bins_sa = pd.qcut(df_f["SUM_ASSURED"], q=5, duplicates="drop")
        t4 = df_f.groupby(bins_sa)["POLICY_STATUS"].apply(lambda s: (s==1).mean()).reset_index(name="ClaimRate")
        t4["Bucket"] = t4["SUM_ASSURED"].astype(str)
        fig4 = px.bar(t4, x="Bucket", y="ClaimRate", title="Claim Rate by Sum Assured Quintiles")
        st.plotly_chart(fig4, use_container_width=True)

    # 5) Payment Mode distribution by Status (stacked)
    if "PAYMENT_MODE" in df_f.columns:
        t5 = (df_f.groupby(["PAYMENT_MODE", "POLICY_STATUS"]).size()
              .reset_index(name="Count"))
        fig5 = px.bar(t5, x="PAYMENT_MODE", y="Count", color="POLICY_STATUS",
                      barmode="stack", title="Payment Mode × Policy Status")
        st.plotly_chart(fig5, use_container_width=True)

    st.info("Tip: Use the filters in the sidebar to explore how claim behavior shifts across job roles and numeric thresholds (e.g., Sum Assured).")

# ----------------------------
# Tab 2: Train & Evaluate
# ----------------------------
with tab2:
    st.subheader("Train & Evaluate: Decision Tree, Random Forest, Gradient Boosting")
    st.caption("80:20 split with stratification. Encodes categoricals, imputes nulls, and reports metrics, confusion matrices, and feature importances.")
    target = "POLICY_STATUS"

    run = st.button("Train Models")
    if run:
        # Prepare encoded data on the whole (unfiltered) dataset
        df_cln = impute_df(df0)
        df_enc, enc_map = encode_fit(df_cln)

        if target not in df_enc.columns:
            st.error("Target column missing after encoding.")
            st.stop()

        X = df_enc.drop(columns=[target])
        y = df_enc[target]
        X_train, X_test, y_train, y_test = stratified_split(X, y)

        models = train_models(X_train, y_train)
        mt = metrics_table(models, X_train, y_train, X_test, y_test)
        st.dataframe(mt, use_container_width=True)

        # Confusion matrices and feature importances
        class_labels = [str(c) for c in sorted(y.unique())]
        for name, m in models.items():
            y_pred = m.predict(X_test)
            cm = confusion_matrix(y_test, y_pred, labels=sorted(y.unique()))
            st.pyplot(plot_confmat(cm, class_labels, f"{name} — Confusion Matrix"))

            fig_imp = feature_importance_figure(m, X.columns, f"{name} — Feature Importances (Top 15)")
            if fig_imp is not None:
                st.plotly_chart(fig_imp, use_container_width=True)

# ----------------------------
# Tab 3: Predict New Data
# ----------------------------
with tab3:
    st.subheader("Predict on New Dataset & Download Results")
    st.caption("Upload a fresh CSV. We'll clean nulls, encode using mappings learned on the current dataset, predict POLICY_STATUS, and let you download results.")

    # Fit encoders and a default model on the base dataset for inference
    df_base = impute_df(df0)
    df_base_enc, base_enc_map = encode_fit(df_base)
    Xb = df_base_enc.drop(columns=["POLICY_STATUS"])
    yb = df_base_enc["POLICY_STATUS"]
    Xtr, Xte, ytr, yte = stratified_split(Xb, yb)
    inf_model = GradientBoostingClassifier(random_state=42).fit(Xtr, ytr)

    new_file = st.file_uploader("Upload new CSV for prediction", type=["csv"], key="pred_csv")
    if new_file is not None:
        new_df_raw = pd.read_csv(new_file)
        new_df_raw = clean_columns(new_df_raw)
        # If target exists, we won't use it for prediction
        if "POLICY_STATUS" in new_df_raw.columns:
            new_df_raw = new_df_raw.drop(columns=["POLICY_STATUS"])

        # Ensure columns exist; missing ones are created with safe defaults
        for col in df_base.columns:
            if col == "POLICY_STATUS":
                continue
            if col not in new_df_raw.columns:
                # create with base mode/mean
                if col in df_base.select_dtypes(include=[np.number]).columns:
                    new_df_raw[col] = df_base[col].mean()
                else:
                    new_df_raw[col] = df_base[col].mode(dropna=True).iloc[0] if not df_base[col].mode().empty else "Missing"

        new_df_imp = impute_df(new_df_raw)
        new_df_enc = encode_transform(new_df_imp, base_enc_map)

        preds = inf_model.predict(new_df_enc[df_base_enc.drop(columns=["POLICY_STATUS"]).columns])
        out = new_df_imp.copy()
        out["PREDICTED_POLICY_STATUS"] = preds

        st.dataframe(out.head(25), use_container_width=True)

        csv_bytes = out.to_csv(index=False).encode("utf-8")
        st.download_button("Download Predictions CSV", data=csv_bytes, file_name="predictions_with_labels.csv", mime="text/csv")

st.caption("© Streamlit Dashboard — Insurance HR Insights & ML")
