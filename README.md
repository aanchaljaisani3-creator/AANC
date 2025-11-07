
# Streamlit Insurance Dashboard (No Folders)

This package contains a **single-file** Streamlit app (plus helpers) to:
- Show **5 interactive charts** with filters (job role + numeric range).
- Train/evaluate **Decision Tree, Random Forest, Gradient Boosting** with 80/20 stratified split.
- Display **accuracy, precision, recall, F1, ROC-AUC**, confusion matrices, and feature importance.
- **Upload** a new dataset and **download predictions** with the label.

## Files
- `app.py` — main Streamlit app.
- `app.yy` — identical copy (if your deployment needs a different main filename).
- `requirements.txt` — minimal packages without version pinning.

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Streamlit Cloud
- Push these files to a GitHub repo **root** (no folders).
- In Streamlit Cloud, set **Main file path** to `app.py`.
- (You can set it to `app.yy`, but Streamlit strongly prefers `.py` files.)

## Data
The app will automatically try to load `Insurance.csv` if present. Otherwise, use the sidebar to upload your CSV.
Columns are whitespace-trimmed; nulls are imputed (mean for numeric, mode for categorical).
The target column must be named **POLICY_STATUS**.
