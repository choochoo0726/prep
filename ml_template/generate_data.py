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

    # --- Targets (based on clean pre-NaN arrays; targets have no missing values by design) ---
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
    _output_dir = Path(__file__).parent / "data"
    _output_dir.mkdir(exist_ok=True)
    df = generate_synthetic_data()
    _out_path = _output_dir / "synthetic.parquet"
    df.write_parquet(_out_path)
    print(f"Saved {len(df)} rows × {len(df.columns)} columns to {_out_path}")
    print(df.head(3))
