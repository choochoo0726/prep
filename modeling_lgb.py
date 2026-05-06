# %%
import plotly.io as pio
pio.renderers.default = "notebook"
pio.templates.default = "plotly_white"

import lightgbm as lgb

import polars as pl
import polars.selectors as S
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNetCV
from sklearn.model_selection import train_test_split, cross_val_score, KFold, StratifiedKFold, TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, accuracy_score

# %% [markdown]
# # load data

# %%
df = pl.read_parquet("ml_template/data/synthetic.parquet")

# %% [markdown]
# ## EDA

# %%
df.with_columns(S.numeric().fill_nan(None)).describe()

# %%
num_cols = [item for item in df.columns if item.startswith('num')]
cat_cols = [item for item in df.columns if item.startswith('cat')]

# %%
df_box = df.select(num_cols).to_pandas().melt(var_name='feature', value_name='value')
px.box(df_box, x='feature', y='value', title='Box Plot of Numerical Features', width=800, height=500)

# %%
df.select(num_cols).to_pandas().corr().pipe(
    px.imshow,
    title="Correlation Heatmap of Numerical Features",
    text_auto=".2f",
    color_continuous_scale="blues",
)


# %%
px.histogram(df.to_pandas(), x='target_reg', title='Distribution of Target Variable', nbins=50, width=800, height=500)

# %% [markdown]
# # data processing

# %%
num_pipeline = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_pipeline = Pipeline(
    [
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "encoder",
            OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
        ),
    ]
)

# %%
preproc = ColumnTransformer(
    transformers=[
        ('num', num_pipeline, num_cols),
        ('cat', cat_pipeline, cat_cols)
    ]
)

# %%
proc_feature_names = num_cols + cat_cols
print(proc_feature_names)

# %% [markdown]
# # sample split

# %%
X_train, X_test, y_train, y_test = train_test_split(
    df.select(S.exclude(["target_reg", "target_bin", "target_multi"])).to_pandas(),
    df["target_reg"].to_pandas(),
    test_size=0.2,
    random_state=42,
)

# %%
X_train_proc = preproc.fit_transform(X_train.reset_index(drop=True))
y_train_proc = y_train.reset_index(drop=True)

# %% [markdown]
# # LightGBM + Optuna 

# %%
import shap
import optuna

# %%
import warnings
warnings.filterwarnings("ignore")

# %%
def compute_metrics(y_true, y_pred, model_name):
    m = {"model": model_name}
    m["MAE"] = mean_absolute_error(y_true, y_pred)
    m["RMSE"] = np.sqrt(mean_squared_error(y_true, y_pred))
    m["R2"] = r2_score(y_true, y_pred)
    return m

# %% [markdown]
# ## full model

# %%
cv = KFold(n_splits=5, shuffle=True, random_state=42)

# %%
def lgb_cv_score(params):
    scores = []
    for train_idx, val_idx in cv.split(X_train_proc, y_train_proc):
        X_tr, X_val = X_train_proc[train_idx], X_train_proc[val_idx]
        y_tr, y_val = y_train_proc[train_idx], y_train_proc[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds = model.predict(X_val)
        score = -float(np.sqrt(mean_squared_error(y_val, preds)))
        scores.append(score)
        
    return np.mean(scores)

# %%
def optuna_objective(trial):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42
    }
    return lgb_cv_score(params)

# %%
study = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), study_name='lgb_test')
study.optimize(optuna_objective, n_trials=40, show_progress_bar=True)

# %%
best_lgb_params = {**study.best_params, 'random_state': 42}

# %%
best_lgb_params

# %%
study.best_value

# %%
lgm_model = lgb.LGBMRegressor(**best_lgb_params, verbose=-1)
lgm_model.fit(X_train_proc, y_train_proc)

# %%
y_train_pred = lgm_model.predict(X_train_proc)

# %%
from optuna.visualization import plot_optimization_history, plot_param_importances

# %%
best_lgb_params

# %%
optuna.visualization.plot_param_importances(study)

# %%
plot_optimization_history(study)

# %%
print(compute_metrics(y_train_proc, y_train_pred, "LightGBM"))

# %%
y_train_resid = y_train_proc - y_train_pred
px.histogram(y_train_resid, nbins=50, title="Residuals Distribution", width=800, height=500)

# %% [markdown]
# # feature selection

# %%
lgb_best_explainer = shap.TreeExplainer(lgm_model)
lgb_best_shap_values = lgb_best_explainer.shap_values(X_train_proc)

# %%
shap.summary_plot(lgb_best_shap_values, X_train_proc, feature_names=proc_feature_names,
                  plot_type="dot", show=True)

# %%
mean_abs_shap = np.abs(lgb_best_shap_values).mean(axis=0)
mae_shap_df = pd.DataFrame(
    {"feature": proc_feature_names, "mean_abs_shap": mean_abs_shap}
).sort_values("mean_abs_shap", ascending=False)

# %%
fig = px.bar(mae_shap_df, x="mean_abs_shap", y="feature", orientation="h", title="Mean Absolute SHAP Values (Test Set)", width=900, height=500)
fig.update_layout(yaxis=dict(autorange="reversed"))

# %%
mae_shap_df = mae_shap_df.assign(ratio=lambda t: t['mean_abs_shap'] / t['mean_abs_shap'].sum(),
                   cum_ratio=lambda t: t['ratio'].cumsum()).reset_index(drop=True)
px.line(mae_shap_df, x="feature", y="cum_ratio", title="Cumulative Ratio of Mean Absolute SHAP Values", width=800, height=500)

# %%
selected_features = mae_shap_df.head(7).feature.tolist()

# %%
selected_features

# %% [markdown]
# ## retrain

# %%
def lgb_cv_score(params, X, y, cv):
    scores = []
    # Use the passed X and y instead of global variables
    for train_idx, val_idx in cv.split(X, y):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr,
            y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
        preds = model.predict(X_val)
        # Note: RMSE is usually minimized; if maximizing, negative RMSE is correct
        score = -float(np.sqrt(mean_squared_error(y_val, preds)))
        scores.append(score)
    return np.mean(scores)

# Add X, y, and cv as parameters here
def optuna_objective(trial, X, y, cv):
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1),
        'num_leaves': trial.suggest_int('num_leaves', 16, 128),
        'subsample': trial.suggest_float('subsample', 0.5, 1.0),
        'min_child_samples': trial.suggest_int('min_child_samples', 5, 100),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
        'random_state': 42,
        'verbosity': -1 # Keeps the output clean
    }
    # Pass them down to the score function
    return lgb_cv_score(params, X, y, cv)

# %%
shap_top_feature_idx = [proc_feature_names.index(f) for f in selected_features]

# %%
X_test_proc = preproc.transform(X_test.reset_index(drop=True))

# %%
X_train_proc_selected = X_train_proc[:, shap_top_feature_idx]
X_test_proc_selected = X_test_proc[:, shap_top_feature_idx]

# %%
study2 = optuna.create_study(direction='maximize', sampler=optuna.samplers.TPESampler(seed=42), study_name='lgb_selected')

# %%
study2.optimize(lambda t: optuna_objective(t, X_train_proc_selected, y_train_proc, cv), n_trials=50, show_progress_bar=True)

# %%
plot_param_importances(study2)


# %%
plot_optimization_history(study2)

# %%
study2.best_trial

# %%
best_model_params2 = {**study2.best_params, 'random_state': 42}

# %%
best_model_params2

# %% [markdown]
# ## final model

# %%
lgb_model_best2 = lgb.LGBMRegressor(**best_model_params2, verbose=-1)

# %%
lgb_model_best2.fit(X_train_proc_selected, y_train_proc)

# %%
explainer3 = shap.TreeExplainer(lgb_model_best2)
shap_values3 = explainer3.shap_values(X_test_proc_selected)
shap.summary_plot(shap_values3, X_test_proc_selected, feature_names=selected_features)

# %%
lgb_test2_pred = lgb_model_best2.predict(X_test_proc_selected)

# %%
compute_metrics(y_test, lgb_test2_pred, "LightGBM")

# %%
lgb2_resid = y_test - lgb_test2_pred

# %%
px.histogram(lgb2_resid, nbins=50, title="LightGBM (Selected Features) Residuals", width=800, height=500)

# %%
fig = px.scatter(
    x=y_test,
    y=lgb_test2_pred,
    title="Predicted vs Actual (LightGBM Selected Features)",
    labels={"x": "Actual", "y": "Predicted"},
    opacity=0.6,
    width=700,
    height=600,
)

fig.add_shape(
    type="line",
    x0=y_test.min(),
    y0=y_test.min(),
    x1=y_test.max(),
    y1=y_test.max(),
    line=dict(color="red", dash="dash"),
)
fig.show()


