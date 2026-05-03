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
import lightgbm as lgb\
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

X_all = df.select(FEATURE_COLS).to_pandas()
y_all = df[TARGET_COL].to_numpy()
dates_all = df["date"].to_numpy()

print(f"Features: {len(FEATURE_COLS)} total ({len(NUM_COLS)} numeric, {len(CAT_COLS)} categorical)")\
""")

CELL_PREPROCESSING_PIPELINE = C("""\
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
print(f"Test  : {len(X_test_raw)} rows (held out)")\
""")

CELL_FEATURE_SELECTION_HEADER = M("""\
## 5. Feature Selection Pipeline

Three progressive stages, all fit on **training data only**:

| Stage | Method | Removes |
|---|---|---|
| **5a Filter** | Variance threshold + correlation filter | Zero-variance noise; redundant correlated features |
| **5b Embedded** | ElasticNet / Lasso coefficient paths | Features zeroed by L1 regularization |
| **5c SHAP** | LightGBM SHAP mean\\|value\\| ranking | Bottom features by contribution; keeps top-K |\
""")

CELL_FILTER_STAGE = C("""\
# ── 5a. FILTER STAGE ────────────────────────────────────────────────────────
_preprocessor_fs = ColumnTransformer(
    [("num", _num_pipe, NUM_COLS), ("cat", _cat_pipe, CAT_COLS)],
    remainder="drop",
)
X_train_proc = _preprocessor_fs.fit_transform(X_train_raw)

_var_sel = VarianceThreshold(threshold=VARIANCE_THRESH)
_var_sel.fit(X_train_proc)
_var_mask = _var_sel.get_support()
features_after_var = [f for f, m in zip(PROC_FEATURE_NAMES, _var_mask) if m]
print(f"Variance filter : dropped {sum(~_var_mask)} features → {len(features_after_var)} remain")

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

CELL_SELECTED_FEATURES = C('''\
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
''')

CELL_MODEL_FITTING_HEADER = M("""\
## 6. Model Fitting

Both models are trained on the **SHAP-selected features** (`X_train_sel`).
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

# ── task-specific cell factories ─────────────────────────────────────────────

def cell_title(task: str):
    titles = {
        "regression": "Regression Pipeline",
        "binary":     "Binary Classification Pipeline",
        "multiclass": "Multiclass Classification Pipeline",
    }
    t = titles[task]
    return M(f"""\
# {t}

An interview-ready end-to-end ML pipeline for **{task}** tasks.

**Models:** ElasticNet (CV-tuned) · LightGBM (Optuna-tuned)
**Feature selection:** Variance filter → Correlation filter → ElasticNet path → SHAP top-K
**CV strategies:** Set `CV_STRATEGY = "kfold"` or `"timeseries"` in the Config cell.\
""")


def cell_config(task: str):
    target_map = {
        "regression": "target_reg",
        "binary":     "target_bin",
        "multiclass": "target_multi",
    }
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


def cell_eda_target(task: str):
    if task == "regression":
        return C("""\
fig = px.histogram(df.to_pandas(), x=TARGET_COL, nbins=50,
                   title=f"Target Distribution: {TARGET_COL}")
fig.show()\
""")
    else:
        return C("""\
_vc = df[TARGET_COL].value_counts().sort("count", descending=True).to_pandas()
fig = px.bar(_vc, x=TARGET_COL, y="count",
             title=f"Target Class Balance: {TARGET_COL}")
fig.show()\
""")


def cell_cv_splitter(task: str):
    if task == "regression":
        kfold_line  = "    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)"
        kfold_print = '    print(f"CV strategy : KFold ({N_SPLITS} folds)")'
    else:
        kfold_line  = "    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)"
        kfold_print = '    print(f"CV strategy : StratifiedKFold ({N_SPLITS} folds)")'

    return C(f"""\
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


def cell_elasticnet_path(task: str):
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
    else:
        fit_block = """\
_en_sel = LogisticRegressionCV(
    penalty="elasticnet", solver="saga",
    l1_ratios=_L1_RATIOS, cv=cv, max_iter=5000, random_state=42,
)
_en_sel.fit(X_corr_filtered, y_train)
_en_coef = np.abs(_en_sel.coef_).mean(axis=0)
print("LogisticRegressionCV (elasticnet, multiclass) fitted")\
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


def cell_shap_stage(task: str):
    lgb_line = (
        "_lgb_quick = lgb.LGBMRegressor(**_lgb_quick_params)"
        if task == "regression"
        else "_lgb_quick = lgb.LGBMClassifier(**_lgb_quick_params)"
    )

    if task == "multiclass":
        shap_mean = """\
if isinstance(_shap_raw, list):
    _mean_abs_shap = np.mean([np.abs(sv).mean(axis=0) for sv in _shap_raw], axis=0)
else:
    _mean_abs_shap = np.abs(_shap_raw).mean(axis=0)\
"""
        beeswarm = """\
if isinstance(_shap_raw, list):
    shap.summary_plot(_shap_raw[0], X_en_filtered, feature_names=features_after_en,
                      plot_type="dot", show=True)
else:
    shap.summary_plot(_shap_raw, X_en_filtered, feature_names=features_after_en,
                      plot_type="dot", show=True)\
"""
    else:
        shap_mean = "_mean_abs_shap = np.abs(_shap_raw).mean(axis=0)"
        beeswarm  = ('shap.summary_plot(_shap_raw, X_en_filtered, feature_names=features_after_en,\n'
                     '                  plot_type="dot", show=True)')

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


def cell_elasticnet_model(task: str):
    if task == "regression":
        body = """\
en_model = ElasticNetCV(l1_ratio=_L1_RATIOS_MODEL, cv=cv, max_iter=5000, random_state=42)
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


def cell_lgbm_cv_score(task: str):
    if task == "regression":
        score_block = """\
        _m = lgb.LGBMRegressor(**params, verbose=-1)
        _m.fit(_Xtr, _ytr, eval_set=[(_Xval, _yval)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        _pred  = _m.predict(_Xval)
        _score = -float(np.sqrt(mean_squared_error(_yval, _pred)))\
"""
    elif task == "binary":
        score_block = """\
        _m = lgb.LGBMClassifier(**params, verbose=-1)
        _m.fit(_Xtr, _ytr, eval_set=[(_Xval, _yval)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        _pred  = _m.predict_proba(_Xval)[:, 1]
        _score = float(roc_auc_score(_yval, _pred))\
"""
    else:
        score_block = """\
        _m = lgb.LGBMClassifier(**params, verbose=-1)
        _m.fit(_Xtr, _ytr, eval_set=[(_Xval, _yval)],
               callbacks=[lgb.early_stopping(30, verbose=False)])
        _pred  = _m.predict_proba(_Xval)
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


def cell_lgbm_final(task: str):
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
{body}
print("LightGBM final model fitted ✓")\
""")


def cell_metrics(task: str):
    if task == "regression":
        fn_body = """\
    m["RMSE"] = round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4)
    m["MAE"]  = round(float(mean_absolute_error(y_true, y_pred)), 4)
    m["R2"]   = round(float(r2_score(y_true, y_pred)), 4)\
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


en_metrics  = compute_metrics(y_test, en_test_pred,  "ElasticNet")
lgb_metrics = compute_metrics(y_test, lgb_test_pred, "LightGBM")

print("ElasticNet :", {{k: v for k, v in en_metrics.items()  if k != "Model"}})
print("LightGBM   :", {{k: v for k, v in lgb_metrics.items() if k != "Model"}})\
""")


def cell_plots(task: str):
    if task == "regression":
        return [C("""\
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
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

_resid = y_test - lgb_test_pred
fig = px.histogram(_resid, nbins=50, title="LightGBM — Residual Distribution",
                   labels={"value": "Residual (actual − predicted)"})
fig.show()\
""")]
    elif task == "binary":
        return [C("""\
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
_fpr, _tpr, _ = roc_curve(y_test, lgb_test_proba)
_auc = roc_auc_score(y_test, lgb_test_proba)
fig = px.line(x=_fpr, y=_tpr, labels={"x": "FPR", "y": "TPR"},
              title=f"LightGBM — ROC Curve (AUC={_auc:.3f})")
fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1,
              line=dict(color="red", dash="dash"))
fig.show()

_prec, _rec, _ = precision_recall_curve(y_test, lgb_test_proba)
fig = px.line(x=_rec, y=_prec, labels={"x": "Recall", "y": "Precision"},
              title="LightGBM — Precision-Recall Curve")
fig.show()

_cm = confusion_matrix(y_test, lgb_test_class)
fig = px.imshow(_cm, text_auto=True, title="LightGBM — Confusion Matrix",
                labels={"x": "Predicted", "y": "Actual"})
fig.show()\
""")]
    else:
        return [C("""\
# ── 7b. PLOTS ────────────────────────────────────────────────────────────────
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

_cm = confusion_matrix(y_test, lgb_test_class)
fig = px.imshow(_cm, text_auto=True, title="LightGBM — Confusion Matrix",
                labels={"x": "Predicted", "y": "Actual"})
fig.show()\
""")]


def cell_shap_final(task: str):
    if task == "multiclass":
        body = """\
if isinstance(_shap_final, list):
    shap.summary_plot(_shap_final[0], X_test_sel,
                      feature_names=SELECTED_FEATURES, plot_type="dot", show=True)
else:
    shap.summary_plot(_shap_final, X_test_sel,
                      feature_names=SELECTED_FEATURES, plot_type="dot", show=True)\
"""
    else:
        body = ('shap.summary_plot(_shap_final, X_test_sel,\n'
                '                  feature_names=SELECTED_FEATURES, plot_type="dot", show=True)')

    return C(f"""\
_explainer_final = shap.TreeExplainer(lgb_model)
_shap_final = _explainer_final.shap_values(X_test_sel)
{body}\
""")


def cell_comparison_table():
    return C("""\
# ── 7c. MODEL COMPARISON TABLE ───────────────────────────────────────────────
_comp_df = pd.DataFrame([en_metrics, lgb_metrics]).set_index("Model")
print("\\n=== Model Comparison (Test Set) ===")
print(_comp_df.to_string())

_comp_melted = _comp_df.reset_index().melt(id_vars="Model", var_name="Metric", value_name="Score")
fig = px.bar(
    _comp_melted, x="Metric", y="Score", color="Model", barmode="group",
    title="Model Comparison — ElasticNet vs LightGBM (Test Set)",
    text_auto=".4f",
)
fig.show()\
""")


# ── notebook assembly ─────────────────────────────────────────────────────────

def build_notebook(task: str):
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
        CELL_FEATURE_SELECTION_HEADER,
        CELL_FILTER_STAGE,
        cell_elasticnet_path(task),
        cell_shap_stage(task),
        CELL_SELECTED_FEATURES,
        CELL_MODEL_FITTING_HEADER,
        cell_elasticnet_model(task),
        cell_lgbm_cv_score(task),
        CELL_LGBM_OPTUNA_PARAMS,
        cell_lgbm_final(task),
        CELL_EVAL_HEADER,
        cell_metrics(task),
        *cell_plots(task),
        cell_shap_final(task),
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
