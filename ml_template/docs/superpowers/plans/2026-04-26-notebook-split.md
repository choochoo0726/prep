# Notebook Split + TabPFN Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TabPFN as a third model to the master notebook, then generate three task-specific notebooks (regression, binary, multiclass) with all `if TASK ==` branching removed.

**Architecture:** A `build_notebooks.py` script defines all cell content as Python strings keyed by task, assembles each notebook with nbformat, and writes the three files. The master `ml_template.ipynb` is updated with TabPFN first, then the three notebooks are built, executed, and converted to HTML.

**Tech Stack:** Python 3.12, nbformat, tabpfn, uv, jupyter nbconvert

---

## File Map

| File | Action |
|---|---|
| `ml_template.ipynb` | Modify — add TabPFN (imports, 6c cell, metrics, comparison) |
| `build_notebooks.py` | Create — generates the three task-specific notebooks |
| `notebook_regression.ipynb` | Created by build_notebooks.py |
| `notebook_binary.ipynb` | Created by build_notebooks.py |
| `notebook_multiclass.ipynb` | Created by build_notebooks.py |

---

### Task 1: Install TabPFN and update master notebook

**Files:**
- Modify: `pyproject.toml` (via uv add)
- Modify: `ml_template.ipynb` — cells 1, 22 (insert after), 24, 28

- [ ] **Step 1: Install tabpfn**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv add tabpfn
```
Expected: tabpfn added to pyproject.toml and uv.lock.

- [ ] **Step 2: Verify import**

```bash
uv run python -c "from tabpfn import TabPFNClassifier, TabPFNRegressor; print('TabPFN OK')"
```
Expected: `TabPFN OK`

- [ ] **Step 3: Update master notebook with TabPFN**

Run this Python script (save as a temp file and execute, then delete):

```python
import nbformat

nb = nbformat.read("ml_template.ipynb", as_version=4)

# 1. Add tabpfn imports to cell 1 (imports cell)
nb.cells[1].source = nb.cells[1].source + "\nfrom tabpfn import TabPFNClassifier, TabPFNRegressor"

# 2. Insert TabPFN fitting cell after cell 22 (LightGBM final fit)
tabpfn_cell = nbformat.v4.new_code_cell("""\
# ── 6c. TABPFN ───────────────────────────────────────────────────────────────
print("=" * 50)
print("TabPFN (zero-config, in-context learning)")
print("=" * 50)

if TASK == "regression":
    tabpfn_model = TabPFNRegressor()
    tabpfn_model.fit(X_train_sel, y_train)
    tabpfn_test_pred  = tabpfn_model.predict(X_test_sel)
    tabpfn_test_proba = None
    tabpfn_test_class = None
elif TASK == "binary":
    tabpfn_model = TabPFNClassifier()
    tabpfn_model.fit(X_train_sel, y_train)
    tabpfn_test_proba = tabpfn_model.predict_proba(X_test_sel)[:, 1]
    tabpfn_test_class = tabpfn_model.predict(X_test_sel)
    tabpfn_test_pred  = tabpfn_test_proba
else:  # multiclass
    tabpfn_model = TabPFNClassifier()
    tabpfn_model.fit(X_train_sel, y_train)
    tabpfn_test_proba = tabpfn_model.predict_proba(X_test_sel)
    tabpfn_test_class = tabpfn_model.predict(X_test_sel)
    tabpfn_test_pred  = tabpfn_test_proba

print("TabPFN fitted ✓")\
""")
nb.cells.insert(23, tabpfn_cell)

# 3. Update metrics cell (now cell 25 after insert) — add tabpfn_metrics
metrics_idx = next(i for i, c in enumerate(nb.cells) if "en_metrics  = compute_metrics" in c.source)
nb.cells[metrics_idx].source = nb.cells[metrics_idx].source.replace(
    'print("ElasticNet :", {k: v for k, v in en_metrics.items()  if k != "Model"})\nprint("LightGBM   :", {k: v for k, v in lgb_metrics.items() if k != "Model"})',
    'tabpfn_metrics = compute_metrics(y_test, tabpfn_test_pred, TASK, "TabPFN")\n\nprint("ElasticNet :", {k: v for k, v in en_metrics.items()  if k != "Model"})\nprint("LightGBM   :", {k: v for k, v in lgb_metrics.items() if k != "Model"})\nprint("TabPFN     :", {k: v for k, v in tabpfn_metrics.items() if k != "Model"})'
)

# 4. Update comparison table cell — add tabpfn_metrics
comp_idx = next(i for i, c in enumerate(nb.cells) if "_comp_df = pd.DataFrame([en_metrics" in c.source)
nb.cells[comp_idx].source = nb.cells[comp_idx].source.replace(
    "_comp_df = pd.DataFrame([en_metrics, lgb_metrics]).set_index(\"Model\")",
    "_comp_df = pd.DataFrame([en_metrics, lgb_metrics, tabpfn_metrics]).set_index(\"Model\")"
).replace(
    "Model Comparison — ElasticNet vs LightGBM (Test Set)",
    "Model Comparison — ElasticNet vs LightGBM vs TabPFN (Test Set)"
)

nbformat.write(nb, "ml_template.ipynb")
print(f"Master notebook updated: {len(nb.cells)} cells")
```

Save this as `_patch_master.py`, run it, then delete it:
```bash
cd /Users/wangxiao/Desktop/Claude/ml_template
uv run python _patch_master.py
rm _patch_master.py
```
Expected: `Master notebook updated: 30 cells`

- [ ] **Step 4: Re-execute master notebook**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to notebook --execute ml_template.ipynb --output ml_template.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -5
```
Expected: no errors.

- [ ] **Step 5: Reconvert master to HTML**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to html ml_template.ipynb 2>&1 | tail -3
```
Expected: `Writing ... bytes to ml_template.html`

---

### Task 2: Write build_notebooks.py

**Files:**
- Create: `build_notebooks.py`

- [ ] **Step 1: Create build_notebooks.py**

Create `/Users/wangxiao/Desktop/Claude/ml_template/build_notebooks.py`:

```python
"""Generate task-specific ML pipeline notebooks (no if-TASK branching).

Usage:
    uv run python build_notebooks.py

Outputs:
    notebook_regression.ipynb
    notebook_binary.ipynb
    notebook_multiclass.ipynb
"""
import nbformat
from pathlib import Path

C = nbformat.v4.new_code_cell
M = nbformat.v4.new_markdown_cell

# ── shared cells (identical across all tasks) ────────────────────────────────

CELL_IMPORTS = C("""\
import plotly.io as pio
pio.renderers.default = "notebook"

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
from tabpfn import TabPFNClassifier, TabPFNRegressor\
""")

CELL_DATA_LOADING = C("""\
# ── 1. DATA LOADING ─────────────────────────────────────────────────────────
df = pl.read_parquet("data/synthetic.parquet")

print("Shape :", df.shape)
print()
print("Dtypes:")
print(df.schema)
print()
print("Missing values per column:")
print(df.null_count())\
""")

CELL_EDA_HEADER = M("## 2. Exploratory Data Analysis")

CELL_EDA_DISTRIBUTIONS = C("""\
# Numeric feature box plots
_num_cols = [c for c in df.columns if c.startswith("num_")]
_df_box = df.select(_num_cols).to_pandas().melt(var_name="feature", value_name="value")
fig = px.box(_df_box, x="feature", y="value", title="Numeric Feature Distributions")
fig.update_layout(xaxis_tickangle=-45)
fig.show()\
""")

CELL_EDA_CORRELATION = C("""\
# Correlation heatmap (numeric features)
_corr = df.select(_num_cols).to_pandas().corr().round(2)
fig = px.imshow(
    _corr, text_auto=True,
    title="Numeric Feature Correlation Heatmap",
    color_continuous_scale="RdBu_r", zmin=-1, zmax=1,
)
fig.show()\
""")

CELL_PREPROCESSING_HEADER = M("""\
## 3. Preprocessing

The `preprocessor` pipeline is defined here but **fit only on training data** to prevent leakage.

- Numeric: median imputation → `StandardScaler`
- Categorical: mode imputation → `OrdinalEncoder` (handles unseen categories as -1)\
""")

CELL_PREPROCESSING_FEATURES = C("""\
# ── 3. PREPROCESSING ────────────────────────────────────────────────────────
FEATURE_COLS = [c for c in df.columns
                if c not in ("date", "target_reg", "target_bin", "target_multi")]
NUM_COLS = [c for c in FEATURE_COLS if c.startswith("num_")]
CAT_COLS = [c for c in FEATURE_COLS if c.startswith("cat_")]

# Convert to pandas for sklearn
X_all = df.select(FEATURE_COLS).to_pandas()
y_all = df[TARGET_COL].to_numpy()
dates_all = df["date"].to_numpy()

print(f"Features: {len(FEATURE_COLS)} total ({len(NUM_COLS)} numeric, {len(CAT_COLS)} categorical)")\
""")

CELL_PREPROCESSING_PIPELINE = C("""\
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
PROC_FEATURE_NAMES = NUM_COLS + CAT_COLS
print("preprocessor defined (not yet fit)")\
""")

CELL_SPLIT_HEADER = M("""\
## 4. Sample Split & Cross-Validation

| Strategy | Splitter | When to use |
|---|---|---|
| `kfold` | `KFold` / `StratifiedKFold` | i.i.d. data |
| `timeseries` | `TimeSeriesSplit` expanding window | time-ordered panel data |

Data is sorted by `date` before splitting. The **last 20%** is held out as the final test set.\
""")

CELL_SPLIT = C("""\
# ── 4. SAMPLE SPLIT ─────────────────────────────────────────────────────────
_sort_idx = np.argsort(dates_all)
X_all     = X_all.iloc[_sort_idx].reset_index(drop=True)
y_all     = y_all[_sort_idx]
dates_all = dates_all[_sort_idx]

_n = len(X_all)
_cut = int(_n * (1 - TEST_FRAC))
X_train_raw, X_test_raw = X_all.iloc[:_cut], X_all.iloc[_cut:]
y_train,     y_test     = y_all[:_cut],       y_all[_cut:]

print(f"Train : {len(X_train_raw)} rows")
print(f"Test  : {len(X_test_raw)} rows (held out — not used until final evaluation)")\
""")

CELL_FILTER_STAGE = C("""\
# ── 5a. FILTER STAGE ────────────────────────────────────────────────────────
_preprocessor_fs = ColumnTransformer(
    [("num", _num_pipe, NUM_COLS), ("cat", _cat_pipe, CAT_COLS)],
    remainder="drop",
)
X_train_proc = _preprocessor_fs.fit_transform(X_train_raw)

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
            _vi = X_var_filtered[:, _i].var()
            _vj = X_var_filtered[:, _j].var()
            _corr_drop_idx.add(_j if _vi >= _vj else _i)

_corr_mask = np.array([i not in _corr_drop_idx for i in range(len(features_after_var))])
features_after_corr = [f for f, m in zip(features_after_var, _corr_mask) if m]
X_corr_filtered = X_var_filtered[:, _corr_mask]

print(f"Correlation filter: dropped {sum(~_corr_mask)} features → {len(features_after_corr)} remain")
print(f"Remaining: {features_after_corr}")\
""")

CELL_SELECTED_FEATURES = C("""\
# Final selected feature list and helper to extract them from raw data
SELECTED_FEATURES = _shap_df["feature"].iloc[:TOP_K_SHAP].tolist()
print(f"SELECTED_FEATURES ({len(SELECTED_FEATURES)}): {SELECTED_FEATURES}")

_en_feat_idx_in_corr = [i for i, m in enumerate(_en_nonzero) if m]
_shap_top_idx_in_en  = [features_after_en.index(f) for f in SELECTED_FEATURES]
_final_idx_in_corr   = [_en_feat_idx_in_corr[i] for i in _shap_top_idx_in_en]


def get_selected_X(X_raw: pd.DataFrame) -> np.ndarray:
    """Preprocess X_raw and return only the SHAP-selected features."""
    _proc = _preprocessor_fs.transform(X_raw)
    _var  = _proc[:, _var_mask]
    _corr = _var[:, _corr_mask]
    return _corr[:, _final_idx_in_corr]


X_train_sel = get_selected_X(X_train_raw)
X_test_sel  = get_selected_X(X_test_raw)
print(f"X_train_sel shape : {X_train_sel.shape}")
print(f"X_test_sel  shape : {X_test_sel.shape}")\
""")

CELL_MODEL_FITTING_HEADER = M("""\
## 6. Model Fitting

All models are trained on the **SHAP-selected features** (`X_train_sel`).  
Final test-set evaluation uses `X_test_sel`.\
""")

CELL_LGBM_OPTUNA_PARAMS = C("""\
def _optuna_objective(trial: optuna.Trial) -> float:
    _p = {
        "n_estimators":      trial.suggest_int("n_estimators", 100, 1000),
        "learning_rate":     trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves":        trial.suggest_int("num_leaves", 16, 128),
        "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
        "subsample":         trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree":  trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "random_state": 42,
    }
    return _lgb_cv_score(_p)

_study = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
_study.optimize(_optuna_objective, n_trials=N_OPTUNA_TRIALS, show_progress_bar=True)

best_lgb_params = {**_study.best_params, "random_state": 42}
print(f"\\nBest CV score : {_study.best_value:.4f}")
print(f"Best params   : {best_lgb_params}")\
""")

CELL_EVAL_HEADER = M("## 7. Evaluation\n\nAll metrics are computed on the **held-out test set** (last 20% by date).")

CELL_SHAP_FINAL_REGRESSION_BINARY = C("""\
# SHAP summary for final LightGBM (on test set)
_explainer_final = shap.TreeExplainer(lgb_model)
_shap_final = _explainer_final.shap_values(X_test_sel)
shap.summary_plot(_shap_final, X_test_sel,
                  feature_names=SELECTED_FEATURES, plot_type="dot", show=True)\
""")

CELL_SHAP_FINAL_MULTICLASS = C("""\
# SHAP summary for final LightGBM (on test set)
_explainer_final = shap.TreeExplainer(lgb_model)
_shap_final = _explainer_final.shap_values(X_test_sel)
if isinstance(_shap_final, list):
    shap.summary_plot(_shap_final[0], X_test_sel,
                      feature_names=SELECTED_FEATURES, plot_type="dot", show=True)
else:
    shap.summary_plot(_shap_final, X_test_sel,
                      feature_names=SELECTED_FEATURES, plot_type="dot", show=True)\
""")

# ── task-specific cells ──────────────────────────────────────────────────────

def cell_title(task: str) -> nbformat.notebooknode.NotebookNode:
    titles = {
        "regression":  "Regression Pipeline",
        "binary":      "Binary Classification Pipeline",
        "multiclass":  "Multiclass Classification Pipeline",
    }
    t = titles[task]
    return M(f"""\
# {t}

An interview-ready end-to-end ML pipeline for **{task}** tasks.

**Models:** ElasticNet · LightGBM (Optuna-tuned) · TabPFN (zero-config)  
**Feature selection:** Variance filter → Correlation filter → ElasticNet path → SHAP top-K  
**CV strategies:** Set `CV_STRATEGY = "kfold"` or `"timeseries"` in the Config cell.\
""")


def cell_config(task: str) -> nbformat.notebooknode.NotebookNode:
    target_map = {"regression": "target_reg", "binary": "target_bin", "multiclass": "target_multi"}
    target_col = target_map[task]
    return C(f"""\
# ── CONFIG ──────────────────────────────────────────────────────────────────
TASK       = "{task}"
TARGET_COL = "{target_col}"

# CV_STRATEGY: "kfold" | "timeseries"
CV_STRATEGY = "kfold"

TOP_K_SHAP      = 10
N_SPLITS        = 5
VARIANCE_THRESH = 0.01
CORR_THRESH     = 0.90
TEST_FRAC       = 0.20
N_OPTUNA_TRIALS = 50

print(f"TASK={{TASK!r}}  TARGET_COL={{TARGET_COL!r}}  CV_STRATEGY={{CV_STRATEGY!r}}")\
""")


def cell_eda_target(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        return C("""\
# Target distribution
fig = px.histogram(df.to_pandas(), x=TARGET_COL, nbins=50,
                   title=f"Target Distribution: {TARGET_COL}")
fig.show()\
""")
    else:
        return C("""\
# Target class balance
_vc = df[TARGET_COL].value_counts().sort("count", descending=True).to_pandas()
fig = px.bar(_vc, x=TARGET_COL, y="count",
             title=f"Target Class Balance: {TARGET_COL}")
fig.show()\
""")


def cell_cv_splitter(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        kfold_line = 'cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)'
        kfold_print = 'print(f"CV strategy : KFold ({N_SPLITS} folds)")'
    else:
        kfold_line = 'cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)'
        kfold_print = 'print(f"CV strategy : StratifiedKFold ({N_SPLITS} folds)")'

    return C(f"""\
# CV splitter
if CV_STRATEGY == "timeseries":
    cv = TimeSeriesSplit(n_splits=N_SPLITS)
    print(f"CV strategy : TimeSeriesSplit (expanding window, {{N_SPLITS}} splits)")
else:
    {kfold_line}
    {kfold_print}

print("\\nFold sizes:")
for i, (tr_idx, val_idx) in enumerate(cv.split(X_train_raw, y_train)):
    print(f"  Fold {{i+1}}: train={{len(tr_idx):4d}}  val={{len(val_idx):4d}}")\
""")


def cell_elasticnet_path(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        fit_block = """\
_en_sel = ElasticNetCV(l1_ratio=_L1_RATIOS, cv=cv, max_iter=5000, random_state=42)
_en_sel.fit(X_corr_filtered, y_train)
_en_coef = np.abs(_en_sel.coef_)
print(f"ElasticNetCV  alpha={_en_sel.alpha_:.5f}  l1_ratio={_en_sel.l1_ratio_:.2f}")\
"""
    elif task == "binary":
        fit_block = """\
_en_sel = LogisticRegressionCV(
    penalty="elasticnet", solver="saga",
    l1_ratios=_L1_RATIOS, cv=cv, max_iter=5000, random_state=42,
)
_en_sel.fit(X_corr_filtered, y_train)
_en_coef = np.abs(_en_sel.coef_[0])
print("LogisticRegressionCV (elasticnet) fitted")\
"""
    else:  # multiclass
        fit_block = """\
_en_sel = LogisticRegressionCV(
    penalty="elasticnet", solver="saga",
    l1_ratios=_L1_RATIOS, cv=cv, max_iter=5000, random_state=42,
    multi_class="multinomial",
)
_en_sel.fit(X_corr_filtered, y_train)
_en_coef = np.abs(_en_sel.coef_).mean(axis=0)
print("LogisticRegressionCV (elasticnet, multinomial) fitted")\
"""

    return C(f"""\
# ── 5b. EMBEDDED STAGE: ELASTICNET / LASSO PATH ─────────────────────────────
_L1_RATIOS = [0.1, 0.5, 0.7, 0.9, 1.0]

{fit_block}

_en_nonzero = _en_coef != 0
features_after_en = [f for f, m in zip(features_after_corr, _en_nonzero) if m]
X_en_filtered = X_corr_filtered[:, _en_nonzero]

print(f"ElasticNet path: zeroed {{sum(~_en_nonzero)}} features → {{len(features_after_en)}} remain")
print(f"Remaining: {{features_after_en}}")

_coef_df = pd.DataFrame({{"feature": features_after_corr, "coef": _en_coef}})
_coef_df = _coef_df.sort_values("coef", ascending=False)
fig = px.bar(
    _coef_df, x="feature", y="coef",
    title="ElasticNet Coefficients (|coef|, sorted)",
    color="coef", color_continuous_scale="Blues",
)
fig.update_layout(xaxis_tickangle=-45)
fig.show()\
""")


def cell_shap_stage(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        lgb_line = "_lgb_quick = lgb.LGBMRegressor(**_lgb_quick_params)"
        shap_mean = "_mean_abs_shap = np.abs(_shap_raw).mean(axis=0)"
        beeswarm  = "shap.summary_plot(_shap_raw, X_en_filtered, feature_names=features_after_en, plot_type=\"dot\", show=True)"
    else:
        lgb_line  = "_lgb_quick = lgb.LGBMClassifier(**_lgb_quick_params)"
        if task == "multiclass":
            shap_mean = """\
if isinstance(_shap_raw, list):
    _mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in _shap_raw], axis=0)
else:
    _mean_abs_shap = np.abs(_shap_raw).mean(axis=0)\
"""
            beeswarm = """\
if isinstance(_shap_raw, list):
    shap.summary_plot(_shap_raw[0], X_en_filtered, feature_names=features_after_en, plot_type="dot", show=True)
else:
    shap.summary_plot(_shap_raw, X_en_filtered, feature_names=features_after_en, plot_type="dot", show=True)\
"""
        else:
            shap_mean = "_mean_abs_shap = np.abs(_shap_raw).mean(axis=0)"
            beeswarm  = "shap.summary_plot(_shap_raw, X_en_filtered, feature_names=features_after_en, plot_type=\"dot\", show=True)"

    return C(f"""\
# ── 5c. SHAP STAGE ───────────────────────────────────────────────────────────
_lgb_quick_params = dict(n_estimators=200, learning_rate=0.05, num_leaves=31,
                         random_state=42, verbose=-1)
{lgb_line}
_lgb_quick.fit(X_en_filtered, y_train)

_explainer_sel = shap.TreeExplainer(_lgb_quick)
_shap_raw = _explainer_sel.shap_values(X_en_filtered)

{shap_mean}

_shap_df = (
    pd.DataFrame({{"feature": features_after_en, "mean_abs_shap": _mean_abs_shap}})
    .sort_values("mean_abs_shap", ascending=False)
    .reset_index(drop=True)
)
print(f"SHAP ranking — retaining top {{TOP_K_SHAP}}:")
print(_shap_df.head(TOP_K_SHAP).to_string(index=False))

fig = px.bar(
    _shap_df.head(TOP_K_SHAP), x="mean_abs_shap", y="feature", orientation="h",
    title=f"Top {{TOP_K_SHAP}} Features by Mean |SHAP| (selection LightGBM)",
    labels={{"mean_abs_shap": "Mean |SHAP value|", "feature": "Feature"}},
)
fig.update_layout(yaxis=dict(autorange="reversed"))
fig.show()

{beeswarm}\
""")


def cell_elasticnet_model(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        body = """\
en_model = ElasticNetCV(
    l1_ratio=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42
)
en_model.fit(X_train_sel, y_train)
en_test_pred  = en_model.predict(X_test_sel)
en_test_proba = None
en_test_class = None
print(f"Best alpha={en_model.alpha_:.5f}  l1_ratio={en_model.l1_ratio_:.2f}")\
"""
    elif task == "binary":
        body = """\
en_model = LogisticRegressionCV(
    penalty="elasticnet", solver="saga",
    l1_ratios=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42,
)
en_model.fit(X_train_sel, y_train)
en_test_proba = en_model.predict_proba(X_test_sel)[:, 1]
en_test_class = en_model.predict(X_test_sel)
en_test_pred  = en_test_proba\
"""
    else:
        body = """\
en_model = LogisticRegressionCV(
    penalty="elasticnet", solver="saga",
    l1_ratios=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42,
    multi_class="multinomial",
)
en_model.fit(X_train_sel, y_train)
en_test_proba = en_model.predict_proba(X_test_sel)
en_test_class = en_model.predict(X_test_sel)
en_test_pred  = en_test_proba\
"""

    return C(f"""\
# ── 6a. ELASTICNET ───────────────────────────────────────────────────────────
print("=" * 50)
print("ElasticNet")
print("=" * 50)

_L1_RATIOS_MODEL = [0.1, 0.5, 0.7, 0.9, 1.0]

{body}

print("ElasticNet fitted ✓")\
""")


def cell_lgbm_cv_score(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        score_block = """\
        _m = lgb.LGBMRegressor(**params, verbose=-1)
        _m.fit(_Xtr, _ytr,
               eval_set=[(_Xval, _yval)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        _pred = _m.predict(_Xval)
        _score = -float(np.sqrt(mean_squared_error(_yval, _pred)))  # -RMSE\
"""
    elif task == "binary":
        score_block = """\
        _m = lgb.LGBMClassifier(**params, verbose=-1)
        _m.fit(_Xtr, _ytr,
               eval_set=[(_Xval, _yval)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        _pred = _m.predict_proba(_Xval)[:, 1]
        _score = float(roc_auc_score(_yval, _pred))\
"""
    else:
        score_block = """\
        _m = lgb.LGBMClassifier(**params, verbose=-1)
        _m.fit(_Xtr, _ytr,
               eval_set=[(_Xval, _yval)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        _pred = _m.predict_proba(_Xval)
        _score = float(roc_auc_score(_yval, _pred, multi_class="ovr", average="macro"))\
"""

    return C(f"""\
# ── 6b. LIGHTGBM + OPTUNA ────────────────────────────────────────────────────
print("=" * 50)
print("LightGBM + Optuna hyperparameter search")
print("=" * 50)


def _lgb_cv_score(params: dict) -> float:
    _scores = []
    for _tr, _val in cv.split(X_train_sel, y_train):
        _Xtr, _Xval = X_train_sel[_tr], X_train_sel[_val]
        _ytr, _yval = y_train[_tr], y_train[_val]
{score_block}
        _scores.append(_score)
    return float(np.mean(_scores))\
""")


def cell_lgbm_final(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        body = """\
lgb_model = lgb.LGBMRegressor(**best_lgb_params, verbose=-1)
lgb_model.fit(X_train_sel, y_train)
lgb_test_pred  = lgb_model.predict(X_test_sel)
lgb_test_proba = None
lgb_test_class = None\
"""
    elif task == "binary":
        body = """\
lgb_model = lgb.LGBMClassifier(**best_lgb_params, verbose=-1)
lgb_model.fit(X_train_sel, y_train)
lgb_test_proba = lgb_model.predict_proba(X_test_sel)[:, 1]
lgb_test_class = lgb_model.predict(X_test_sel)
lgb_test_pred  = lgb_test_proba\
"""
    else:
        body = """\
lgb_model = lgb.LGBMClassifier(**best_lgb_params, verbose=-1)
lgb_model.fit(X_train_sel, y_train)
lgb_test_proba = lgb_model.predict_proba(X_test_sel)
lgb_test_class = lgb_model.predict(X_test_sel)
lgb_test_pred  = lgb_test_proba\
"""

    return C(f"""\
# Fit final LightGBM on full training set with best params
{body}
print("LightGBM final model fitted ✓")\
""")


def cell_tabpfn(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        body = """\
tabpfn_model = TabPFNRegressor()
tabpfn_model.fit(X_train_sel, y_train)
tabpfn_test_pred  = tabpfn_model.predict(X_test_sel)
tabpfn_test_proba = None
tabpfn_test_class = None\
"""
    elif task == "binary":
        body = """\
tabpfn_model = TabPFNClassifier()
tabpfn_model.fit(X_train_sel, y_train)
tabpfn_test_proba = tabpfn_model.predict_proba(X_test_sel)[:, 1]
tabpfn_test_class = tabpfn_model.predict(X_test_sel)
tabpfn_test_pred  = tabpfn_test_proba\
"""
    else:
        body = """\
tabpfn_model = TabPFNClassifier()
tabpfn_model.fit(X_train_sel, y_train)
tabpfn_test_proba = tabpfn_model.predict_proba(X_test_sel)
tabpfn_test_class = tabpfn_model.predict(X_test_sel)
tabpfn_test_pred  = tabpfn_test_proba\
"""

    return C(f"""\
# ── 6c. TABPFN ───────────────────────────────────────────────────────────────
print("=" * 50)
print("TabPFN (zero-config, in-context learning)")
print("=" * 50)

{body}
print("TabPFN fitted ✓")\
""")


def cell_metrics(task: str) -> nbformat.notebooknode.NotebookNode:
    if task == "regression":
        fn_body = """\
    m["RMSE"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    m["MAE"]  = round(float(mean_absolute_error(y_true, y_pred)), 4)
    m["R\\u00b2"]   = round(float(r2_score(y_true, y_pred)), 4)\
"""
    elif task == "binary":
        fn_body = """\
    _cls = (y_pred >= 0.5).astype(int)
    m["AUC-ROC"]  = round(float(roc_auc_score(y_true, y_pred)), 4)
    m["F1"]       = round(float(f1_score(y_true, _cls)), 4)
    m["Log-loss"] = round(float(log_loss(y_true, y_pred)), 4)
    m["Accuracy"] = round(float(accuracy_score(y_true, _cls)), 4)\
"""
    else:
        fn_body = """\
    _cls = np.argmax(y_pred, axis=1)
    m["Macro-AUC"] = round(float(roc_auc_score(y_true, y_pred, multi_class="ovr", average="macro")), 4)
    m["Macro-F1"]  = round(float(f1_score(y_true, _cls, average="macro")), 4)
    m["Log-loss"]  = round(float(log_loss(y_true, y_pred)), 4)
    m["Accuracy"]  = round(float(accuracy_score(y_true, _cls)), 4)\
"""

    return C(f"""\
# ── 7a. METRICS ──────────────────────────────────────────────────────────────

def compute_metrics(y_true, y_pred, model_name: str) -> dict:
    m = {{"Model": model_name}}
{fn_body}
    return m


en_metrics     = compute_metrics(y_test, en_test_pred,     "ElasticNet")
lgb_metrics    = compute_metrics(y_test, lgb_test_pred,    "LightGBM")
tabpfn_metrics = compute_metrics(y_test, tabpfn_test_pred, "TabPFN")

print("ElasticNet :", {{k: v for k, v in en_metrics.items()     if k != "Model"}})
print("LightGBM   :", {{k: v for k, v in lgb_metrics.items()    if k != "Model"}})
print("TabPFN     :", {{k: v for k, v in tabpfn_metrics.items() if k != "Model"}})\
""")


def cell_plots(task: str) -> list:
    """Returns a list of plot cells (varies by task)."""
    if task == "regression":
        return [C("""\
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
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
fig.show()\
""")]
    elif task == "binary":
        return [C("""\
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
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
fig.show()\
""")]
    else:  # multiclass
        return [C("""\
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
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
fig.show()\
""")]


def cell_comparison_table() -> nbformat.notebooknode.NotebookNode:
    return C("""\
# ── 7c. MODEL COMPARISON TABLE ───────────────────────────────────────────────
_comp_df = pd.DataFrame([en_metrics, lgb_metrics, tabpfn_metrics]).set_index("Model")
print("\\n=== Model Comparison (Test Set) ===")
print(_comp_df.to_string())

_comp_melted = _comp_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
fig = px.bar(
    _comp_melted, x="Metric", y="Score", color="Model", barmode="group",
    title="Model Comparison — ElasticNet vs LightGBM vs TabPFN (Test Set)",
    text_auto=".4f",
)
fig.show()\
""")


# ── notebook assembly ────────────────────────────────────────────────────────

def build_notebook(task: str) -> nbformat.notebooknode.NotebookNode:
    shap_final_cell = CELL_SHAP_FINAL_MULTICLASS if task == "multiclass" else CELL_SHAP_FINAL_REGRESSION_BINARY

    cells = [
        cell_title(task),
        CELL_IMPORTS,
        cell_config(task),
        CELL_DATA_LOADING,
        CELL_EDA_HEADER,
        CELL_EDA_DISTRIBUTIONS,
        CELL_EDA_CORRELATION,
        cell_eda_target(task),
        CELL_PREPROCESSING_HEADER,
        CELL_PREPROCESSING_FEATURES,
        CELL_PREPROCESSING_PIPELINE,
        CELL_SPLIT_HEADER,
        CELL_SPLIT,
        cell_cv_splitter(task),
        M("## 5. Feature Selection Pipeline\n\n"
          "Three progressive stages, all fit on **training data only**:\n\n"
          "| Stage | Method | Removes |\n|---|---|---|\n"
          "| **5a Filter** | Variance threshold + correlation filter | Zero-variance noise; redundant correlated features |\n"
          "| **5b Embedded** | ElasticNet / Lasso coefficient paths | Features zeroed by L1 regularization |\n"
          "| **5c SHAP** | LightGBM SHAP mean\\|value\\| ranking | Bottom features by contribution; keeps top-K |"),
        CELL_FILTER_STAGE,
        cell_elasticnet_path(task),
        cell_shap_stage(task),
        CELL_SELECTED_FEATURES,
        CELL_MODEL_FITTING_HEADER,
        cell_elasticnet_model(task),
        cell_lgbm_cv_score(task),
        CELL_LGBM_OPTUNA_PARAMS,
        cell_lgbm_final(task),
        cell_tabpfn(task),
        CELL_EVAL_HEADER,
        cell_metrics(task),
        *cell_plots(task),
        shap_final_cell,
        cell_comparison_table(),
    ]

    nb = nbformat.v4.new_notebook()
    nb.cells = cells
    return nb


if __name__ == "__main__":
    for task in ("regression", "binary", "multiclass"):
        nb = build_notebook(task)
        path = Path(f"notebook_{task}.ipynb")
        nbformat.write(nb, path)
        print(f"Written {len(nb.cells)} cells → {path}")
```

- [ ] **Step 2: Run build_notebooks.py**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run python build_notebooks.py
```
Expected:
```
Written 30 cells → notebook_regression.ipynb
Written 30 cells → notebook_binary.ipynb
Written 30 cells → notebook_multiclass.ipynb
```
(Cell count may vary slightly — within ±2 is fine.)

- [ ] **Step 3: Verify files exist**

```bash
ls -lh notebook_*.ipynb
```
Expected: three .ipynb files, each > 20KB.

---

### Task 3: Execute all three notebooks and convert to HTML

**Files:**
- Modify: `notebook_regression.ipynb`, `notebook_binary.ipynb`, `notebook_multiclass.ipynb`
- Create: `notebook_regression.html`, `notebook_binary.html`, `notebook_multiclass.html`

- [ ] **Step 1: Execute regression notebook**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to notebook --execute notebook_regression.ipynb --output notebook_regression.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -5
```
Expected: no error tracebacks.

- [ ] **Step 2: Execute binary notebook**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to notebook --execute notebook_binary.ipynb --output notebook_binary.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -5
```
Expected: no error tracebacks.

- [ ] **Step 3: Execute multiclass notebook**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to notebook --execute notebook_multiclass.ipynb --output notebook_multiclass.ipynb --ExecutePreprocessor.timeout=600 2>&1 | tail -5
```
Expected: no error tracebacks.

- [ ] **Step 4: Convert all three to HTML**

```bash
cd /Users/wangxiao/Desktop/Claude/ml_template && uv run jupyter nbconvert --to html notebook_regression.ipynb notebook_binary.ipynb notebook_multiclass.ipynb 2>&1 | tail -6
```
Expected: three .html files written.

- [ ] **Step 5: Verify outputs**

```bash
ls -lh notebook_*.html
```
Expected: three HTML files, each > 1MB (Plotly charts embedded).

---

## Self-Review

### Spec coverage
| Requirement | Task |
|---|---|
| `uv add tabpfn` | Task 1 Step 1 |
| TabPFN imports in master | Task 1 Step 3 |
| TabPFN Section 6c cell in master (all 3 TASK branches) | Task 1 Step 3 |
| `tabpfn_metrics` in master metrics cell | Task 1 Step 3 |
| TabPFN in master comparison table | Task 1 Step 3 |
| Master notebook re-executed | Task 1 Step 4 |
| Master HTML updated | Task 1 Step 5 |
| `build_notebooks.py` generates 3 notebooks | Task 2 |
| Config cell hardcoded TASK + TARGET_COL, no `_TARGET_MAP` | Task 2 `cell_config()` |
| `if TASK ==` guards removed (CV, EN path, SHAP, EN model, LGB, metrics, plots) | Task 2 all `cell_*` functions |
| TabPFN in all 3 task-specific notebooks | Task 2 `cell_tabpfn()` |
| 3-model comparison table in all notebooks | Task 2 `cell_comparison_table()` |
| All 3 notebooks executed | Task 3 Steps 1-3 |
| All 3 notebooks converted to HTML | Task 3 Steps 4-5 |

All spec requirements covered. ✅

### Placeholder scan
No TBDs or incomplete steps found. ✅

### Type consistency
- `compute_metrics` in task-specific notebooks takes `(y_true, y_pred, model_name)` — 3 args (no `task` param needed since task is hardcoded). Called consistently in all three notebooks. ✅
- `tabpfn_test_pred`, `tabpfn_test_proba`, `tabpfn_test_class` produced by `cell_tabpfn()`, consumed by `cell_metrics()` — consistent. ✅
- `lgb_test_proba`, `lgb_test_class` produced by `cell_lgbm_final()`, consumed by `cell_plots()` — consistent for binary/multiclass. ✅
- `lgb_model` produced by `cell_lgbm_final()`, consumed by `CELL_SHAP_FINAL_*` — consistent. ✅
