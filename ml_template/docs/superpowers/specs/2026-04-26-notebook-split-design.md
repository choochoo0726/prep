# Notebook Split — Design Spec
**Date:** 2026-04-26  
**Goal:** Split the master ML template into three task-specific notebooks (regression, binary classification, multiclass classification), each with `if TASK ==` guards removed and TabPFN added as a third model.

---

## 1. Files

| File | Action |
|---|---|
| `ml_template.ipynb` | Unchanged — remains the master all-in-one reference |
| `notebook_regression.ipynb` | New — regression-only, hardcoded `TASK="regression"` |
| `notebook_binary.ipynb` | New — binary classification, hardcoded `TASK="binary"` |
| `notebook_multiclass.ipynb` | New — multiclass classification, hardcoded `TASK="multiclass"` |
| `build_notebooks.py` | New — script that generates all three notebooks using nbformat |

---

## 2. TabPFN Addition (applied to all notebooks including master)

Before generating the split notebooks, TabPFN is added to `ml_template.ipynb`:

- `uv add tabpfn`
- Imports cell: add `from tabpfn import TabPFNClassifier, TabPFNRegressor`
- New cell after LightGBM final fit (Section 6c): fit TabPFN on `X_train_sel` / `y_train`, produce `tabpfn_test_pred`, `tabpfn_test_proba`, `tabpfn_test_class`
- Metrics cell: add `tabpfn_metrics = compute_metrics(y_test, tabpfn_test_pred, TASK, "TabPFN")`
- Comparison table cell: include `tabpfn_metrics` alongside `en_metrics` and `lgb_metrics`

---

## 3. Per-Notebook Cell Transformations

Each task-specific notebook is built by `build_notebooks.py` with the following cell content (hardcoded, no branching):

### Config cell
- `TASK` hardcoded to the task string
- `TARGET_COL` hardcoded directly (no `_TARGET_MAP` dict)
- All other config variables (`TOP_K_SHAP`, `N_SPLITS`, etc.) unchanged

### Title markdown cell
- "Regression Pipeline", "Binary Classification Pipeline", or "Multiclass Classification Pipeline"

### CV splitter cell
- Regression: `KFold(n_splits=N_SPLITS, shuffle=True, random_state=42)`
- Binary/Multiclass: `StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=42)`
- `TimeSeriesSplit` branch is kept via the `CV_STRATEGY` config variable (user can still set `CV_STRATEGY="timeseries"`)

### ElasticNet path cell (5b)
- Regression: `ElasticNetCV` only
- Binary: `LogisticRegressionCV(penalty="elasticnet", multi_class="auto")` only
- Multiclass: `LogisticRegressionCV(penalty="elasticnet", multi_class="multinomial")` only
- Single-path coefficient extraction (no `if _en_sel.coef_.ndim > 1`)

### ElasticNet fitting cell (6a)
- Regression: `ElasticNetCV` only, sets `en_test_pred` as continuous
- Binary: `LogisticRegressionCV` only, sets `en_test_proba` and `en_test_class`
- Multiclass: `LogisticRegressionCV(multi_class="multinomial")` only, sets `en_test_proba` and `en_test_class`

### LightGBM CV score function (_lgb_cv_score)
- Regression: `-RMSE` scoring only
- Binary: `roc_auc_score` only
- Multiclass: `roc_auc_score(..., multi_class="ovr", average="macro")` only

### LightGBM final fit cell
- Single branch matching the task

### TabPFN cell (6c) — new cell in all notebooks
- Regression: `TabPFNRegressor().fit(X_train_sel, y_train)`, `tabpfn_test_pred = model.predict(X_test_sel)`
- Binary: `TabPFNClassifier().fit(...)`, `tabpfn_test_proba = model.predict_proba(X_test_sel)[:, 1]`
- Multiclass: `TabPFNClassifier().fit(...)`, `tabpfn_test_proba = model.predict_proba(X_test_sel)`
- No hyperparameter tuning (TabPFN is zero-config)

### compute_metrics function
- Single task branch only

### Plots cells (7b)
- Regression notebook: predicted vs actual scatter + residual histogram (no classification cells)
- Binary notebook: ROC curve + PR curve + confusion matrix (no regression cells)
- Multiclass notebook: per-class ROC curves + confusion matrix (no regression cells)

### SHAP cell
- Identical across all notebooks (works for all task types already)

### Comparison table cell (7c)
- All three notebooks: `pd.DataFrame([en_metrics, lgb_metrics, tabpfn_metrics])` — three-model comparison

---

## 4. build_notebooks.py

A standalone Python script that:
1. Defines cell content for each section as a dict keyed by task
2. Creates each notebook using `nbformat.v4`
3. Writes `notebook_regression.ipynb`, `notebook_binary.ipynb`, `notebook_multiclass.ipynb`
4. Does NOT execute the notebooks (execution is done separately)

---

## 5. Master Notebook Update

`ml_template.ipynb` is updated with TabPFN (Section 6c + metrics + comparison table) but otherwise unchanged. The `if TASK ==` branching is preserved as the all-in-one reference.

---

## 6. Execution & Verification

After generation, each notebook is executed with:
```bash
uv run jupyter nbconvert --to notebook --execute <notebook>.ipynb --output <notebook>.ipynb --ExecutePreprocessor.timeout=600
```
And converted to HTML:
```bash
uv run jupyter nbconvert --to html <notebook>.ipynb
```
