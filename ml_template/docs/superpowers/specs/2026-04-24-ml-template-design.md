# ML Template Notebook — Design Spec
**Date:** 2026-04-24  
**Goal:** A reusable interview-ready notebook covering the full ML pipeline for regression, binary classification, and multiclass classification.

---

## 1. File Structure

```
ml_template/
├── generate_data.py              # synthetic data generator
├── data/
│   └── synthetic.parquet         # output of generate_data.py
└── ml_template.ipynb             # main modeling notebook
```

---

## 2. Data Generation (`generate_data.py`)

Produces `data/synthetic.parquet` with ~2000 rows and the following columns:

| Column group | Details |
|---|---|
| `date` | daily dates, used for time-series splits |
| `num_1` … `num_10` | numeric features; some correlated pairs, some pure noise |
| `cat_1` … `cat_5` | categorical features; low and high cardinality mix |
| `target_reg` | continuous target (regression) |
| `target_bin` | binary target 0/1 (binary classification) |
| `target_multi` | multiclass target 0/1/2 (multiclass classification) |

---

## 3. Notebook Config Cell

At the top of `ml_template.ipynb`, one cell controls all branching:

```python
TASK         = "regression"   # "regression" | "binary" | "multiclass"
TARGET_COL   = "target_reg"   # set automatically based on TASK, or override
CV_STRATEGY  = "kfold"        # "kfold" | "timeseries"
TOP_K_SHAP   = 10             # number of features to retain after SHAP selection
N_SPLITS     = 5              # folds for both CV strategies
```

---

## 4. Notebook Sections

### Section 0 — Config
Single cell as described above.

### Section 1 — Data Loading
- Read `data/synthetic.parquet` with Polars
- Print shape, dtypes, missing value counts
- Auto-set `TARGET_COL` based on `TASK` if not overridden

### Section 2 — EDA
- Distribution plots for numeric features (Plotly Express)
- Correlation heatmap
- Target distribution / class balance check (task-appropriate)

### Section 3 — Preprocessing
- Numeric: median imputation → `StandardScaler`
- Categorical: mode imputation → `OrdinalEncoder` (or `OneHotEncoder` for low-cardinality)
- All transformers fit on training data only (no leakage)
- Wrapped in `sklearn.Pipeline`

### Section 4 — Sample Split
Two strategies, selected by `CV_STRATEGY`:

| Strategy | Implementation | Use case |
|---|---|---|
| `kfold` | `StratifiedKFold` (classification) or `KFold` (regression), 5 folds | i.i.d. data |
| `timeseries` | `TimeSeriesSplit` with expanding window, 5 splits; data sorted by `date` | panel / time-ordered data |

A held-out final test set (last 20% of data, or last time split) is reserved before any CV.

### Section 5 — Feature Selection Pipeline

Applied to training folds only. Three progressive stages:

**5a. Filter Stage**
- Variance threshold: drop features with variance < 0.01
- Correlation filter: for each correlated pair (|corr| > 0.90), drop the feature with lower variance

**5b. Embedded Stage (ElasticNet path)**
- Regression: `ElasticNetCV` with `l1_ratio` grid `[0.1, 0.5, 0.9, 1.0]`
- Classification: `LogisticRegressionCV` with `penalty="elasticnet"`, same `l1_ratio` grid, `solver="saga"`
- Retain features with non-zero coefficients at the selected alpha
- Plot coefficient paths

**5c. SHAP Stage**
- Fit a lightweight LightGBM on ElasticNet-selected features
- Compute SHAP values (`shap.TreeExplainer`)
- Rank features by mean |SHAP|; retain top `TOP_K_SHAP`
- Display SHAP summary bar plot (Plotly) and beeswarm plot (shap native)

### Section 6 — Model Fitting

**6a. ElasticNet**
- Regression: `ElasticNetCV`
- Binary classification: `LogisticRegressionCV(penalty="elasticnet", solver="saga")`
- Multiclass: same, with `multi_class="multinomial"`
- Fit on training folds using the SHAP-selected feature set

**6b. LightGBM**
- `LGBMRegressor` / `LGBMClassifier`
- Hyperparameter tuning via `optuna` (50 trials): `num_leaves`, `learning_rate`, `n_estimators`, `min_child_samples`, `subsample`
- Early stopping on validation fold within each CV split
- Fit on training folds using the SHAP-selected feature set

### Section 7 — Evaluation

**7a. Per-model metrics (on held-out test set)**

| Task | Metrics |
|---|---|
| Regression | RMSE, MAE, R² |
| Binary classification | AUC-ROC, F1, log-loss, accuracy |
| Multiclass classification | macro AUC, macro F1, log-loss, accuracy |

**7b. Plots**

| Task | Plots |
|---|---|
| Regression | Predicted vs actual scatter, residual distribution |
| Binary | ROC curve, precision-recall curve, confusion matrix |
| Multiclass | Per-class ROC curves, confusion matrix |
| All | SHAP summary plot for final LightGBM model |

**7c. Model comparison table**
Side-by-side DataFrame comparing ElasticNet vs LightGBM across all metrics for the active task. Displayed as a styled table.

---

## 5. Libraries

| Purpose | Library |
|---|---|
| Data wrangling | `polars` |
| ML models | `scikit-learn`, `lightgbm` |
| SHAP | `shap` |
| Hyperparameter tuning | `optuna` |
| Visualization | `plotly.express`, `shap` (beeswarm) |
| Serialization | `pyarrow` (parquet read/write) |

---

## 6. Design Decisions & Notes

- **No leakage**: all preprocessing transformers and feature selectors are fit inside CV folds, never on the full dataset.
- **Task branching**: `if TASK == "..."` guards all task-specific cells so the notebook runs end-to-end with a single config change.
- **SHAP for feature selection**: SHAP is run on a quick LightGBM fit (not the tuned final model) to keep the selection stage fast; the tuned final model then trains only on the selected features.
- **Polars at boundaries**: data loading and wrangling use Polars; `.to_pandas()` conversion happens at the sklearn Pipeline boundary only.
- **Plotly for all custom plots**: all custom charts use `plotly.express`; SHAP beeswarm uses the native `shap.plots.beeswarm` (matplotlib-based) since no Plotly equivalent exists.
