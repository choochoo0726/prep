# ML Template Notebook Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a self-contained, interview-ready ML pipeline notebook covering regression, binary, and multiclass tasks with ElasticNet and LightGBM, switchable via a single config cell.

**Architecture:** A `generate_data.py` script produces a synthetic parquet file. A single `ml_template.ipynb` reads it and runs: EDA → preprocessing → feature selection (variance filter → correlation filter → ElasticNet path → SHAP ranking) → model fitting (ElasticNet + LightGBM with Optuna) → evaluation with a side-by-side comparison table. All task-specific branching is controlled by `TASK` and `CV_STRATEGY` in the config cell.

**Tech Stack:** Python 3.11+, uv, polars, scikit-learn, lightgbm, shap, optuna, plotly.express, nbformat, pyarrow, ipykernel

---

## File Map

| File | Purpose |
|---|---|
| `pyproject.toml` | uv project config and dependencies |
| `generate_data.py` | Synthetic data generator → writes `data/synthetic.parquet` |
| `data/synthetic.parquet` | 2000 rows × 19 cols (10 numeric, 5 categorical, 3 targets, 1 date) |
| `ml_template.ipynb` | Main notebook — all modeling sections |

---

### Task 1: Initialize project environment

**Files:**
- Create: `pyproject.toml`
- Create: `uv.lock`

- [ ] **Step 1: Initialize uv project**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template
uv init --no-workspace
```
Expected: `pyproject.toml` and `hello.py` created (delete `hello.py`).

- [ ] **Step 2: Add all dependencies**

```bash
uv add polars pyarrow scikit-learn lightgbm shap optuna plotly nbformat ipykernel jupyter
```
Expected: `uv.lock` created, packages installed under `.venv/`.

- [ ] **Step 3: Verify imports work**

```bash
uv run python -c "import polars, lightgbm, shap, optuna, plotly, nbformat; print('All imports OK')"
```
Expected: `All imports OK`

---

### Task 2: Write the synthetic data generator

**Files:**
- Create: `generate_data.py`

- [ ] **Step 1: Write `generate_data.py`**

Create `/Users/wangxiao/Desktop/Claude/ml_template/generate_data.py`:

```python
"""Generate synthetic ML data and save to data/synthetic.parquet."""
import datetime
from pathlib import Path

import numpy as np
import polars as pl


def generate_synthetic_data(n_samples: int = 2000, random_state: int = 42) -> pl.DataFrame:
    rng = np.random.default_rng(random_state)

    # --- Date column (for time-series CV) ---
    start = datetime.date(2020, 1, 1)
    dates = [start + datetime.timedelta(days=i) for i in range(n_samples)]

    # --- Numeric features ---
    # Correlated pair (num_1 ↔ num_2, corr ≈ 0.85)
    num_1 = rng.normal(0, 1, n_samples)
    num_2 = num_1 * 0.85 + rng.normal(0, 0.3, n_samples)
    # Correlated pair (num_3 ↔ num_4, corr ≈ 0.75)
    num_3 = rng.normal(2, 1.5, n_samples)
    num_4 = num_3 * 0.75 + rng.normal(0, 0.5, n_samples)
    # Independent informative features
    num_5 = rng.exponential(1, n_samples)
    num_6 = rng.uniform(-3, 3, n_samples)
    num_9 = rng.normal(1, 2, n_samples)
    num_10 = rng.normal(-1, 1.5, n_samples)
    # Near-zero-variance noise features (should be dropped by variance filter)
    num_7 = rng.normal(0, 0.008, n_samples)
    num_8 = rng.normal(0, 0.005, n_samples)

    nums = np.column_stack(
        [num_1, num_2, num_3, num_4, num_5, num_6, num_7, num_8, num_9, num_10]
    ).astype(float)

    # Inject ~2% missing values into numeric features
    missing_mask = rng.random(nums.shape) < 0.02
    nums[missing_mask] = np.nan

    # --- Categorical features ---
    cat_1 = rng.choice(["A", "B", "C"], n_samples)               # low cardinality
    cat_2 = rng.choice(["X", "Y"], n_samples)                     # binary
    cat_3 = rng.choice([f"val_{i}" for i in range(20)], n_samples)  # high cardinality
    cat_4 = rng.choice(["low", "mid", "high"], n_samples, p=[0.3, 0.5, 0.2])
    cat_5 = rng.choice(["P", "Q", "R", "S"], n_samples)

    # --- Targets (signal from known features so selection is testable) ---
    cat_1_effect = np.where(cat_1 == "A", 1.0, np.where(cat_1 == "B", -1.0, 0.0))
    signal = (
        2.0 * num_1
        - 1.5 * num_3
        + 0.8 * num_5
        + 0.5 * num_9
        + cat_1_effect
        + rng.normal(0, 0.5, n_samples)
    )

    target_reg = signal
    target_bin = (signal > float(np.median(signal))).astype(int)
    target_multi = np.digitize(signal, np.percentile(signal, [33.3, 66.6])).astype(int)

    return pl.DataFrame(
        {
            "date": dates,
            **{f"num_{i + 1}": nums[:, i] for i in range(10)},
            "cat_1": cat_1,
            "cat_2": cat_2,
            "cat_3": cat_3,
            "cat_4": cat_4,
            "cat_5": cat_5,
            "target_reg": target_reg,
            "target_bin": target_bin,
            "target_multi": target_multi,
        }
    )


if __name__ == "__main__":
    Path("data").mkdir(exist_ok=True)
    df = generate_synthetic_data()
    df.write_parquet("data/synthetic.parquet")
    print(f"Saved {len(df)} rows × {len(df.columns)} columns to data/synthetic.parquet")
    print(df.head(3))
```

- [ ] **Step 2: Run the generator**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run python generate_data.py
```
Expected:
```
Saved 2000 rows × 19 columns to data/synthetic.parquet
```

- [ ] **Step 3: Verify parquet is readable**

```bash
uv run python -c "import polars as pl; df = pl.read_parquet('data/synthetic.parquet'); print(df.shape, df.columns)"
```
Expected: `(2000, 19)` and a list of column names.

---

### Task 3: Create notebook skeleton — title, imports, config, data loading

**Files:**
- Create: `ml_template.ipynb`

- [ ] **Step 1: Create empty notebook**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run python -c "
import nbformat
nb = nbformat.v4.new_notebook()
nbformat.write(nb, 'ml_template.ipynb')
print('created ml_template.ipynb')
"
```

- [ ] **Step 2: Add title markdown cell**

Add markdown cell to `ml_template.ipynb` at index 0:
```markdown
# ML Pipeline Template — Regression / Binary / Multiclass

A reusable, interview-ready end-to-end ML pipeline.

**Usage:** Set `TASK` and `CV_STRATEGY` in the Config cell below, then **Run All Cells**.

| `TASK` | Target column | Models |
|---|---|---|
| `"regression"` | `target_reg` | ElasticNetCV, LGBMRegressor |
| `"binary"` | `target_bin` | LogisticRegressionCV (elasticnet), LGBMClassifier |
| `"multiclass"` | `target_multi` | LogisticRegressionCV (elasticnet, multinomial), LGBMClassifier |
```

- [ ] **Step 3: Add imports cell**

```python
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import shap
import optuna
optuna.logging.set_verbosity(optuna.logging.WARNING)

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import ElasticNetCV, LogisticRegressionCV
from sklearn.model_selection import KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error, r2_score,
    roc_auc_score, f1_score, log_loss, accuracy_score,
    confusion_matrix, roc_curve, precision_recall_curve,
)
import lightgbm as lgb
```

- [ ] **Step 4: Add config cell**

```python
# ── CONFIG — change these to switch task / CV strategy ─────────────────────
# TASK options: "regression" | "binary" | "multiclass"
TASK = "regression"

# CV_STRATEGY options: "kfold" | "timeseries"
# kfold     → KFold / StratifiedKFold (i.i.d. data)
# timeseries → TimeSeriesSplit expanding window (time-ordered data)
CV_STRATEGY = "kfold"

# Feature selection
TOP_K_SHAP       = 10     # features to retain after SHAP ranking
N_SPLITS         = 5      # CV folds
VARIANCE_THRESH  = 0.01   # drop features with variance below this
CORR_THRESH      = 0.90   # drop one feature from each highly-correlated pair
TEST_FRAC        = 0.20   # fraction of data held out as final test set

# LightGBM tuning
N_OPTUNA_TRIALS  = 50

# Auto-set target column (or override manually)
_TARGET_MAP = {"regression": "target_reg", "binary": "target_bin", "multiclass": "target_multi"}
TARGET_COL = _TARGET_MAP[TASK]

print(f"TASK={TASK!r}  TARGET_COL={TARGET_COL!r}  CV_STRATEGY={CV_STRATEGY!r}")
```

- [ ] **Step 5: Add data loading cell (Section 1)**

```python
# ── 1. DATA LOADING ─────────────────────────────────────────────────────────
df = pl.read_parquet("data/synthetic.parquet")

print(f"Shape : {df.shape}")
print(f"\nDtypes:\n{df.schema}")
print(f"\nMissing values per column:")
print(df.null_count())
```

- [ ] **Step 6: Verify notebook runs clean**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to notebook --execute ml_template.ipynb --output ml_template_test.ipynb 2>&1 | tail -5
rm -f ml_template_test.ipynb
```
Expected: no errors, execution completes.

---

### Task 4: EDA section

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add EDA section header cell**

Add markdown cell:
```markdown
## 2. Exploratory Data Analysis
```

- [ ] **Step 2: Add numeric distributions cell**

```python
# Numeric feature box plots
_num_cols = [c for c in df.columns if c.startswith("num_")]
_df_box = df.select(_num_cols).to_pandas().melt(var_name="feature", value_name="value")
fig = px.box(_df_box, x="feature", y="value", title="Numeric Feature Distributions")
fig.update_layout(xaxis_tickangle=-45)
fig.show()
```

- [ ] **Step 3: Add correlation heatmap cell**

```python
# Correlation heatmap (numeric features)
_corr = df.select(_num_cols).to_pandas().corr().round(2)
fig = px.imshow(
    _corr, text_auto=True,
    title="Numeric Feature Correlation Heatmap",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
)
fig.show()
```

- [ ] **Step 4: Add target distribution cell**

```python
# Target distribution / class balance
if TASK == "regression":
    fig = px.histogram(df.to_pandas(), x=TARGET_COL, nbins=50,
                       title=f"Target Distribution: {TARGET_COL}")
else:
    _vc = df[TARGET_COL].value_counts().sort("count", descending=True).to_pandas()
    fig = px.bar(_vc, x=TARGET_COL, y="count",
                 title=f"Target Class Balance: {TARGET_COL}")
fig.show()
```

---

### Task 5: Preprocessing section

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add preprocessing header cell**

Add markdown cell:
```markdown
## 3. Preprocessing

The `preprocessor` pipeline is defined here but **fit only on training data** to prevent leakage.

- Numeric: median imputation → `StandardScaler`
- Categorical: mode imputation → `OrdinalEncoder` (handles unseen categories as -1)
```

- [ ] **Step 2: Add feature identification cell**

```python
# ── 3. PREPROCESSING ────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns
                if c not in ("date", "target_reg", "target_bin", "target_multi")]
NUM_COLS = [c for c in FEATURE_COLS if c.startswith("num_")]
CAT_COLS = [c for c in FEATURE_COLS if c.startswith("cat_")]

# Convert to pandas for sklearn
X_all = df.select(FEATURE_COLS).to_pandas()
y_all = df[TARGET_COL].to_numpy()
dates_all = df["date"].to_numpy()

print(f"Features: {len(FEATURE_COLS)} total ({len(NUM_COLS)} numeric, {len(CAT_COLS)} categorical)")
```

- [ ] **Step 3: Add preprocessor definition cell**

```python
# Preprocessing pipeline — not yet fit (fit inside CV folds / train-only)
_num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])
_cat_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
])
preprocessor = ColumnTransformer(
    [("num", _num_pipe, NUM_COLS), ("cat", _cat_pipe, CAT_COLS)],
    remainder="drop",
)
# Post-transform feature names in the same order as ColumnTransformer output
PROC_FEATURE_NAMES = NUM_COLS + CAT_COLS

print("preprocessor defined (not yet fit)")
```

---

### Task 6: Sample split and CV scaffold

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add split header cell**

Add markdown cell:
```markdown
## 4. Sample Split & Cross-Validation

| Strategy | Splitter | When to use |
|---|---|---|
| `kfold` | `KFold` / `StratifiedKFold` | i.i.d. data |
| `timeseries` | `TimeSeriesSplit` expanding window | time-ordered panel data |

Data is sorted by `date` before splitting. The **last 20%** is held out as the final test set and never touched during feature selection or hyperparameter tuning.
```

- [ ] **Step 2: Add train/test split cell**

```python
# ── 4. SAMPLE SPLIT ─────────────────────────────────────────────────────────
# Sort by date (required for timeseries CV; harmless for kfold)
_sort_idx = np.argsort(dates_all)
X_all     = X_all.iloc[_sort_idx].reset_index(drop=True)
y_all     = y_all[_sort_idx]
dates_all = dates_all[_sort_idx]

_n = len(X_all)
_cut = int(_n * (1 - TEST_FRAC))
X_train_raw, X_test_raw = X_all.iloc[:_cut], X_all.iloc[_cut:]
y_train,     y_test     = y_all[:_cut],       y_all[_cut:]

print(f"Train : {len(X_train_raw)} rows")
print(f"Test  : {len(X_test_raw)} rows (held out — not used until final evaluation)")
```

- [ ] **Step 3: Add CV splitter cell**

```python
# CV splitter
if CV_STRATEGY == "timeseries":
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"CV strategy : TimeSeriesSplit (expanding window, {N_SPLITS} splits)")
else:
    if TASK == "regression":
        cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        print(f"CV strategy : KFold ({N_SPLITS} folds)")
    else:
        cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)
        print(f"CV strategy : StratifiedKFold ({N_SPLITS} folds)")

print("\nFold sizes:")
for i, (tr_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train)):
    print(f"  Fold {i+1}: train={len(tr_idx):4d}  val={len(val_idx):4d}")
```

---

### Task 7: Feature selection — filter stage

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add feature selection header cell**

Add markdown cell:
```markdown
## 5. Feature Selection Pipeline

Three progressive stages, all fit on **training data only**:

| Stage | Method | Removes |
|---|---|---|
| **5a Filter** | Variance threshold + correlation filter | Zero-variance noise; redundant correlated features |
| **5b Embedded** | ElasticNet / Lasso coefficient paths | Features zeroed by L1 regularization |
| **5c SHAP** | LightGBM SHAP mean\|value\| ranking | Bottom features by contribution; keeps top-K |
```

- [ ] **Step 2: Add filter stage cell**

```python
# ── 5a. FILTER STAGE ────────────────────────────────────────────────────────
# Fit preprocessor on training set (for feature selection only — not the final model preprocessor)
_preprocessor_fs = ColumnTransformer(
    [("num", _num_pipe, NUM_COLS), ("cat", _cat_pipe, CAT_COLS)],
    remainder="drop",
)
X_train_proc = _preprocessor_fs.fit_transform(X_train_raw)  # shape (n_train, n_features)

# -- Variance filter --
_var_sel = VarianceThreshold(threshold=VARIANCE_THRESH)
_var_sel.fit(X_train_proc)
_var_mask = _var_sel.get_support()
features_after_var = [f for f, m in zip(PROC_FEATURE_NAMES, _var_mask) if m]
print(f"Variance filter : dropped {sum(~_var_mask)} features → {len(features_after_var)} remain")

# -- Correlation filter --
X_var_filtered = X_train_proc[:, _var_mask]
_corr_mat = np.abs(np.corrcoef(X_var_filtered.T))
np.fill_diagonal(_corr_mat, 0)

_corr_drop_idx = set()
for _i in range(_corr_mat.shape[0]):
    for _j in range(_i + 1, _corr_mat.shape[1]):
        if _corr_mat[_i, _j] > CORR_THRESH:
            # Drop the feature with lower variance
            _vi = X_var_filtered[:, _i].var()
            _vj = X_var_filtered[:, _j].var()
            _corr_drop_idx.add(_j if _vi >= _vj else _i)

_corr_mask = np.array([i not in _corr_drop_idx for i in range(len(features_after_var))])
features_after_corr = [f for f, m in zip(features_after_var, _corr_mask) if m]
X_corr_filtered = X_var_filtered[:, _corr_mask]

print(f"Correlation filter: dropped {sum(~_corr_mask)} features → {len(features_after_corr)} remain")
print(f"Remaining: {features_after_corr}")
```

---

### Task 8: Feature selection — ElasticNet path

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add ElasticNet path cell**

```python
# ── 5b. EMBEDDED STAGE: ELASTICNET / LASSO PATH ─────────────────────────────
_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 1.0]

if TASK == "regression":
    _en_sel = ElasticNetCV(l1_ratio=_L1_RATIOS, cv=cv, max_iter=5000, random_state=42)
    _en_sel.fit(X_corr_filtered, y_train)
    _en_coef = np.abs(_en_sel.coef_)
    print(f"ElasticNetCV  alpha={_en_sel.alpha_:.5f}  l1_ratio={_en_sel.l1_ratio_:.2f}")
else:
    _en_sel = LogisticRegressionCV(
        penalty="elasticnet", solver="saga",
        l1_ratios=_L1_RATIOS,
        cv=cv, max_iter=5000, random_state=42,
        multi_class="multinomial" if TASK == "multiclass" else "auto",
    )
    _en_sel.fit(X_corr_filtered, y_train)
    # For multiclass coef_ is (n_classes, n_features); take mean |coef| across classes
    _en_coef = np.abs(_en_sel.coef_).mean(axis=0) if _en_sel.coef_.ndim > 1 else np.abs(_en_sel.coef_[0])
    print(f"LogisticRegressionCV (elasticnet) fitted")

_en_nonzero = _en_coef != 0
features_after_en = [f for f, m in zip(features_after_corr, _en_nonzero) if m]
X_en_filtered = X_corr_filtered[:, _en_nonzero]

print(f"ElasticNet path: zeroed {sum(~_en_nonzero)} features → {len(features_after_en)} remain")
print(f"Remaining: {features_after_en}")

# Coefficient bar chart
_coef_df = pd.DataFrame({"feature": features_after_corr, "coef": _en_coef})
_coef_df = _coef_df.sort_values("coef", ascending=False)
fig = px.bar(
    _coef_df, x="feature", y="coef",
    title="ElasticNet Coefficients (|coef|, sorted)",
    color="coef", color_continuous_scale="Blues",
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()
```

---

### Task 9: Feature selection — SHAP stage

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add SHAP selection cell**

```python
# ── 5c. SHAP STAGE ───────────────────────────────────────────────────────────
# Fit a lightweight (untuned) LightGBM purely for SHAP-based feature ranking.
# The final tuned model is trained separately in Section 6.
_lgb_quick_params = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                         random_state=42, verbose=-1)
if TASK == "regression":
    _lgb_quick = lgb.LGBMRegressor(**_lgb_quick_params)
else:
    _lgb_quick = lgb.LGBMClassifier(**_lgb_quick_params)

_lgb_quick.fit(X_en_filtered, y_train)

_explainer_sel = shap.TreeExplainer(_lgb_quick)
_shap_raw = _explainer_sel.shap_values(X_en_filtered)

# Average |SHAP| across classes for multiclass
if TASK == "multiclass" and isinstance(_shap_raw, list):
    _mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in _shap_raw], axis=0)
else:
    _mean_abs_shap = np.abs(_shap_raw).mean(axis=0)

_shap_df = (
    pd.DataFrame({"feature": features_after_en, "mean_abs_shap": _mean_abs_shap})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
print(f"SHAP ranking — retaining top {TOP_K_SHAP}:")
print(_shap_df.head(TOP_K_SHAP).to_string(index=False))

# SHAP bar chart (Plotly)
fig = px.bar(
    _shap_df.head(TOP_K_SHAP), x="mean_abs_shap", y="feature", orientation="h",
    title=f"Top {TOP_K_SHAP} Features by Mean |SHAP| (selection LightGBM)",
    labels={"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"},
)
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.show()

# SHAP beeswarm (native shap — matplotlib)
if TASK == "multiclass" and isinstance(_shap_raw, list):
    shap.summary_plot(_shap_raw[0], X_en_filtered, feature_names=features_after_en,
                      plot_type="dot", show=True)
else:
    shap.summary_plot(_shap_raw, X_en_filtered, feature_names=features_after_en,
                      plot_type="dot", show=True)
```

- [ ] **Step 2: Add selected-feature extraction cell**

```python
# Final selected feature list and helper to extract them from raw data
SELECTED_FEATURES = _shap_df["feature"].iloc[:TOP_K_SHAP].tolist()
print(f"SELECTED_FEATURES ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

# Indices within X_corr_filtered that correspond to SELECTED_FEATURES
_en_feat_idx_in_corr = [i for i, m in enumerate(_en_nonzero) if m]         # EN survivors in corr space
_shap_top_idx_in_en  = [features_after_en.index(f) for f in SELECTED_FEATURES]  # top-K in EN space
_final_idx_in_corr   = [_en_feat_idx_in_corr[i] for i in _shap_top_idx_in_en]    # map back to corr space


def get_selected_X(X_raw: pd.DataFrame) -> np.ndarray:
    """Preprocess X_raw and return only the SHAP-selected features.
    
    NOTE: _preprocessor_fs, _var_mask, _corr_mask, _final_idx_in_corr
    must be fit/defined before calling this function (done in Section 5).
    """
    _proc = _preprocessor_fs.transform(X_raw)
    _var  = _proc[:, _var_mask]
    _corr = _var[:, _corr_mask]
    return _corr[:, _final_idx_in_corr]


X_train_sel = get_selected_X(X_train_raw)
X_test_sel  = get_selected_X(X_test_raw)
print(f"X_train_sel shape : {X_train_sel.shape}")
print(f"X_test_sel  shape : {X_test_sel.shape}")
```

---

### Task 10: ElasticNet model fitting

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add model fitting header cell**

Add markdown cell:
```markdown
## 6. Model Fitting

Both models are trained on the **SHAP-selected features** (`X_train_sel`) with CV-based tuning.  
Final test-set evaluation uses `X_test_sel`.
```

- [ ] **Step 2: Add ElasticNet fitting cell**

```python
# ── 6a. ELASTICNET ───────────────────────────────────────────────────────────
print("=" * 50)
print("ElasticNet")
print("=" * 50)

_L1_RATIOS_MODEL = [0.1, 0.5, 0.7, 0.9, 1.0]

if TASK == "regression":
    en_model = ElasticNetCV(
        l1_ratio=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42
    )
    en_model.fit(X_train_sel, y_train)
    en_test_pred  = en_model.predict(X_test_sel)   # continuous predictions
    en_test_proba = None
    en_test_class = None
    print(f"Best alpha={en_model.alpha_:.5f}  l1_ratio={en_model.l1_ratio_:.2f}")

elif TASK == "binary":
    en_model = LogisticRegressionCV(
        penalty="elasticnet", solver="saga",
        l1_ratios=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42,
    )
    en_model.fit(X_train_sel, y_train)
    en_test_proba = en_model.predict_proba(X_test_sel)[:, 1]  # P(class=1)
    en_test_class = en_model.predict(X_test_sel)
    en_test_pred  = en_test_proba  # used in compute_metrics

else:  # multiclass
    en_model = LogisticRegressionCV(
        penalty="elasticnet", solver="saga",
        l1_ratios=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42,
        multi_class="multinomial",
    )
    en_model.fit(X_train_sel, y_train)
    en_test_proba = en_model.predict_proba(X_test_sel)    # (n_test, n_classes)
    en_test_class = en_model.predict(X_test_sel)
    en_test_pred  = en_test_proba

print("ElasticNet fitted ✓")
```

---

### Task 11: LightGBM model fitting with Optuna

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add Optuna objective cell**

```python
# ── 6b. LIGHTGBM + OPTUNA ────────────────────────────────────────────────────
print("=" * 50)
print("LightGBM + Optuna hyperparameter search")
print("=" * 50)


def _lgb_cv_score(params: dict) -> float:
    """Return mean CV score (higher = better) for given params."""
    _scores = []
    for _tr, _val in cv.split(X_train_sel, y_train):
        _Xtr, _Xval = X_train_sel[_tr], X_train_sel[_val]
        _ytr, _yval = y_train[_tr], y_train[_val]

        if TASK == "regression":
            _m = lgb.LGBMRegressor(**params, verbose=-1)
            _m.fit(_Xtr, _ytr,
                   eval_set=[(_Xval, _yval)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
            _pred = _m.predict(_Xval)
            _score = -float(mean_squared_error(_yval, _pred, squared=False))  # −RMSE

        elif TASK == "binary":
            _m = lgb.LGBMClassifier(**params, verbose=-1)
            _m.fit(_Xtr, _ytr,
                   eval_set=[(_Xval, _yval)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
            _pred = _m.predict_proba(_Xval)[:, 1]
            _score = float(roc_auc_score(_yval, _pred))

        else:  # multiclass
            _m = lgb.LGBMClassifier(**params, verbose=-1)
            _m.fit(_Xtr, _ytr,
                   eval_set=[(_Xval, _yval)],
                   callbacks=[lgb.early_stopping(30, verbose=False)])
            _pred = _m.predict_proba(_Xval)
            _score = float(roc_auc_score(_yval, _pred, multi_class="ovr", average="macro"))

        _scores.append(_score)
    return float(np.mean(_scores))


def _optuna_objective(trial: optuna.Trial) -> float:
    _p = {
        "n_estimators":       trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":      trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":         trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples":  trial.suggest_int("min_child_samples", 5, 50),
        "subsample":          trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":   trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }
    return _lgb_cv_score(_p)


_study = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
_study.optimize(_optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_lgb_params = {**_study.best_params, "random_state": 42}
print(f"\nBest CV score : {_study.best_value:.4f}")
print(f"Best params   : {best_lgb_params}")
```

- [ ] **Step 2: Add LightGBM final fit cell**

```python
# Fit final LightGBM on full training set with best params
if TASK == "regression":
    lgb_model = lgb.LGBMRegressor(**best_lgb_params, verbose=-1)
    lgb_model.fit(X_train_sel, y_train)
    lgb_test_pred  = lgb_model.predict(X_test_sel)
    lgb_test_proba = None
    lgb_test_class = None

elif TASK == "binary":
    lgb_model = lgb.LGBMClassifier(**best_lgb_params, verbose=-1)
    lgb_model.fit(X_train_sel, y_train)
    lgb_test_proba = lgb_model.predict_proba(X_test_sel)[:, 1]
    lgb_test_class = lgb_model.predict(X_test_sel)
    lgb_test_pred  = lgb_test_proba

else:  # multiclass
    lgb_model = lgb.LGBMClassifier(**best_lgb_params, verbose=-1)
    lgb_model.fit(X_train_sel, y_train)
    lgb_test_proba = lgb_model.predict_proba(X_test_sel)
    lgb_test_class = lgb_model.predict(X_test_sel)
    lgb_test_pred  = lgb_test_proba

print("LightGBM final model fitted ✓")
```

---

### Task 12: Evaluation — metrics, plots, comparison table

**Files:**
- Modify: `ml_template.ipynb`

- [ ] **Step 1: Add evaluation header cell**

Add markdown cell:
```markdown
## 7. Evaluation

All metrics are computed on the **held-out test set** (last 20% by date).
```

- [ ] **Step 2: Add metrics computation cell**

```python
# ── 7a. METRICS ──────────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, task: str, model_name: str) -> dict:
    m: dict = {"Model": model_name}
    if task == "regression":
        m["RMSE"] = round(float(mean_squared_error(y_true, y_pred, squared=False)), 4)
        m["MAE"]  = round(float(mean_absolute_error(y_true, y_pred)), 4)
        m["R²"]   = round(float(r2_score(y_true, y_pred)), 4)
    elif task == "binary":
        _cls = (y_pred >= 0.5).astype(int)
        m["AUC-ROC"]  = round(float(roc_auc_score(y_true, y_pred)), 4)
        m["F1"]       = round(float(f1_score(y_true, _cls)), 4)
        m["Log-loss"] = round(float(log_loss(y_true, y_pred)), 4)
        m["Accuracy"] = round(float(accuracy_score(y_true, _cls)), 4)
    else:  # multiclass
        _cls = np.argmax(y_pred, axis=1)
        m["Macro-AUC"] = round(float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")), 4)
        m["Macro-F1"]  = round(float(f1_score(y_true, _cls, average="macro")), 4)
        m["Log-loss"]  = round(float(log_loss(y_true, y_pred)), 4)
        m["Accuracy"]  = round(float(accuracy_score(y_true, _cls)), 4)
    return m


en_metrics  = compute_metrics(y_test, en_test_pred,  TASK, "ElasticNet")
lgb_metrics = compute_metrics(y_test, lgb_test_pred, TASK, "LightGBM")

print("ElasticNet :", {k: v for k, v in en_metrics.items()  if k != "Model"})
print("LightGBM   :", {k: v for k, v in lgb_metrics.items() if k != "Model"})
```

- [ ] **Step 3: Add regression plots cell**

```python
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
if TASK == "regression":
    # Predicted vs actual
    fig = px.scatter(
        x=y_test, y=lgb_test_pred, opacity=0.5,
        labels={"x": "Actual", "y": "Predicted"},
        title="LightGBM — Predicted vs Actual",
    )
    fig.add_shape(type="line",
                  x0=float(y_test.min()), y0=float(y_test.min()),
                  x1=float(y_test.max()), y1=float(y_test.max()),
                  line=dict(color="red", dash="dash"))
    fig.show()

    # Residual distribution
    _resid = y_test - lgb_test_pred
    fig = px.histogram(_resid, nbins=50, title="LightGBM — Residual Distribution",
                       labels={"value": "Residual (actual − predicted)"})
    fig.show()
```

- [ ] **Step 4: Add classification plots cell**

```python
if TASK == "binary":
    # ROC curve
    _fpr, _tpr, _ = roc_curve(y_test, lgb_test_proba)
    _auc = roc_auc_score(y_test, lgb_test_proba)
    fig = px.line(x=_fpr, y=_tpr, labels={"x": "FPR", "y": "TPR"},
                  title=f"LightGBM — ROC Curve (AUC={_auc:.3f})")
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                  line=dict(color="red", dash="dash"))
    fig.show()

    # Precision-Recall curve
    _prec, _rec, _ = precision_recall_curve(y_test, lgb_test_proba)
    fig = px.line(x=_rec, y=_prec, labels={"x": "Recall", "y": "Precision"},
                  title="LightGBM — Precision-Recall Curve")
    fig.show()

    # Confusion matrix
    _cm = confusion_matrix(y_test, lgb_test_class)
    fig = px.imshow(_cm, text_auto=True, title="LightGBM — Confusion Matrix",
                    labels={"x": "Predicted", "y": "Actual"})
    fig.show()

if TASK == "multiclass":
    # Per-class ROC curves
    _n_cls = len(np.unique(y_test))
    _fig_traces = []
    for _cls in range(_n_cls):
        _fpr, _tpr, _ = roc_curve((y_test == _cls).astype(int), lgb_test_proba[:, _cls])
        _auc = roc_auc_score((y_test == _cls).astype(int), lgb_test_proba[:, _cls])
        _fig_traces.append(go.Scatter(x=_fpr, y=_tpr, name=f"Class {_cls} AUC={_auc:.3f}"))
    _fig_roc = go.Figure(_fig_traces)
    _fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
                       line=dict(color="red", dash="dash"))
    _fig_roc.update_layout(title="LightGBM — Per-Class ROC Curves",
                            xaxis_title="FPR", yaxis_title="TPR")
    _fig_roc.show()

    # Confusion matrix
    _cm = confusion_matrix(y_test, lgb_test_class)
    fig = px.imshow(_cm, text_auto=True, title="LightGBM — Confusion Matrix",
                    labels={"x": "Predicted", "y": "Actual"})
    fig.show()
```

- [ ] **Step 5: Add SHAP final model cell**

```python
# SHAP summary for final LightGBM (on test set)
_explainer_final = shap.TreeExplainer(lgb_model)
_shap_final = _explainer_final.shap_values(X_test_sel)

if TASK == "multiclass" and isinstance(_shap_final, list):
    shap.summary_plot(_shap_final[0], X_test_sel,
                      feature_names=SELECTED_FEATURES, plot_type="dot", show=True)
else:
    shap.summary_plot(_shap_final, X_test_sel,
                      feature_names=SELECTED_FEATURES, plot_type="dot", show=True)
```

- [ ] **Step 6: Add model comparison table cell**

```python
# ── 7c. MODEL COMPARISON TABLE ───────────────────────────────────────────────
_comp_df = pd.DataFrame([en_metrics, lgb_metrics]).set_index("Model")
print("\n=== Model Comparison (Test Set) ===")
print(_comp_df.to_string())

# Grouped bar chart
_comp_melted = _comp_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
fig = px.bar(
    _comp_melted, x="Metric", y="Score", color="Model", barmode="group",
    title="Model Comparison — ElasticNet vs LightGBM (Test Set)",
    text_auto=".4f",
)
fig.show()
```

- [ ] **Step 7: Run full notebook end-to-end**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template
uv run jupyter nbconvert --to notebook --execute ml_template.ipynb \
    --output ml_template.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -10
```
Expected: `ml_template.ipynb` saved with all outputs. No error tracebacks.

---

## Self-Review

### Spec coverage
| Spec requirement | Task |
|---|---|
| Synthetic parquet with date + numeric + categorical + 3 targets | Task 2 |
| Config cell with TASK / CV_STRATEGY / TOP_K_SHAP | Task 3 |
| Data loading with Polars | Task 3 |
| EDA: distributions, correlation heatmap, target balance | Task 4 |
| Preprocessing: imputation, scaling, encoding | Task 5 |
| Train/test split (last 20%) | Task 6 |
| CV: kfold + timeseries | Task 6 |
| Filter: variance threshold + correlation filter | Task 7 |
| Embedded: ElasticNet / LassoCV path | Task 8 |
| SHAP: top-K selection + bar chart + beeswarm | Task 9 |
| ElasticNet model (regression / binary / multiclass) | Task 10 |
| LightGBM + Optuna tuning + early stopping | Task 11 |
| Metrics: RMSE/MAE/R² or AUC/F1/logloss/accuracy | Task 12 |
| Plots: residuals / ROC / PR / confusion matrix / SHAP | Task 12 |
| Side-by-side comparison table | Task 12 |
| Polars for data, Plotly Express for charts | throughout |
| No leakage: preprocessor fit on train only | Tasks 5, 7, 9 |

All spec requirements covered. ✅

### Placeholder scan
No TBDs, TODOs, or "similar to Task N" references found. ✅

### Type consistency
- `cv` defined in Task 6, used in Tasks 8, 10, 11 ✅
- `_preprocessor_fs` fit in Task 7, used via `get_selected_X` in Tasks 9, 10, 11 ✅
- `_var_mask`, `_corr_mask`, `_final_idx_in_corr` defined in Tasks 7–9, used in `get_selected_X` ✅
- `X_train_sel`, `X_test_sel` defined in Task 9, used in Tasks 10–12 ✅
- `en_test_pred`, `lgb_test_pred` defined in Tasks 10–11, used in Task 12 ✅
- `lgb_test_proba`, `lgb_test_class` defined in Task 11, used in Task 12 classification plots ✅
- `compute_metrics` defined and used in Task 12 ✅
- `SELECTED_FEATURES` defined in Task 9, used in Task 12 SHAP plot ✅
