from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


OUTPUT_DIR = Path("outputs")


FEATURE_COLUMNS = [
    "lag_1",
    "lag_4",
    "roll_4",
    "roll_8",
    "sales_std_4",
    "recent_trend",
    "growth_ratio",
    "lag1_vs_lag4",
    "promo_flag",
    "promo_uplift",
    "week_sin",
    "week_cos",
    "brand_code",
    "sku_type_code",
    "channel_code",
]


TARGET_COLUMN = "actual_4w"


def _split_feature_table(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = feature_df[feature_df["decision_week"] <= 36].copy()
    validation_df = feature_df[feature_df["decision_week"].between(37, 48)].copy()
    test_df = feature_df[feature_df["decision_week"].between(49, 56)].copy()
    return train_df, validation_df, test_df


def _build_models() -> dict[str, object]:
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=26,
            max_depth=8,
            random_state=42,
            n_jobs=1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.03,
            l2_regularization=0.05,
            max_leaf_nodes=63,
            random_state=42,
        ),
    }


def _add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    result = df.copy()
    result["forecast_4w"] = result["forecast_4w"].clip(lower=0).round().astype(int)
    result["actual_4w"] = result["actual_4w"].round().astype(int)
    result["error"] = result["forecast_4w"] - result["actual_4w"]
    result["abs_error"] = result["error"].abs()
    result["ape"] = np.where(
        result["actual_4w"] > 0,
        result["abs_error"] / result["actual_4w"],
        np.where(result["abs_error"] == 0, 0.0, np.nan),
    )
    result["bias"] = result["error"]
    result["hit_20_flag"] = result["ape"].fillna(np.inf) <= 0.20
    return result


def _summarize_metrics(result_df: pd.DataFrame) -> dict[str, float]:
    actual_sum = result_df["actual_4w"].sum()
    abs_error_sum = result_df["abs_error"].sum()
    error_sum = result_df["error"].sum()

    if actual_sum == 0:
        wape = 0.0 if abs_error_sum == 0 else np.nan
        bias_pct = 0.0 if error_sum == 0 else np.nan
    else:
        wape = abs_error_sum / actual_sum
        bias_pct = error_sum / actual_sum

    return {
        "wape": wape,
        "forecast_accuracy": 1 - wape if pd.notna(wape) else np.nan,
        "bias_pct": bias_pct,
        "hit_rate_20": result_df["hit_20_flag"].mean(),
        "mae": result_df["abs_error"].mean(),
    }


def _predict_raw(model: object, source_df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(source_df[FEATURE_COLUMNS]), a_min=0, a_max=None)


def _predict_with_metrics(
    model: object,
    source_df: pd.DataFrame,
    calibration_factor: float = 1.0,
) -> pd.DataFrame:
    result_df = source_df.copy()
    result_df["forecast_4w"] = _predict_raw(model, result_df) * calibration_factor
    return _add_row_metrics(result_df)


def _calculate_calibration_factor(model: object, validation_df: pd.DataFrame) -> float:
    raw_forecast_sum = _predict_raw(model, validation_df).sum()
    actual_sum = validation_df[TARGET_COLUMN].sum()
    if raw_forecast_sum <= 0:
        return 1.0
    return float(np.clip(actual_sum / raw_forecast_sum, 0.85, 1.20))


def train_and_select_forecast_model() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(OUTPUT_DIR / "01_feature_table.csv")
    feature_df = feature_df.dropna(subset=[TARGET_COLUMN]).copy()

    train_df, validation_df, test_df = _split_feature_table(feature_df)
    models = _build_models()
    summary_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        fitted_models[model_name] = model

        validation_result = _predict_with_metrics(model, validation_df)
        metrics = _summarize_metrics(validation_result)
        summary_rows.append(
            {
                "model_name": model_name,
                "split": "validation",
                "wape": metrics["wape"],
                "forecast_accuracy": metrics["forecast_accuracy"],
                "bias_pct": metrics["bias_pct"],
                "hit_rate_20": metrics["hit_rate_20"],
                "mae": metrics["mae"],
                "calibration_factor": 1.0,
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    best_model_name = summary_df.sort_values("wape", ascending=True).iloc[0]["model_name"]
    final_model = fitted_models[best_model_name]
    calibration_factor = _calculate_calibration_factor(final_model, validation_df)

    test_result = _predict_with_metrics(final_model, test_df, calibration_factor=calibration_factor)
    test_metrics = _summarize_metrics(test_result)
    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(
                [
                    {
                        "model_name": best_model_name,
                        "split": "test",
                        "wape": test_metrics["wape"],
                        "forecast_accuracy": test_metrics["forecast_accuracy"],
                        "bias_pct": test_metrics["bias_pct"],
                        "hit_rate_20": test_metrics["hit_rate_20"],
                        "mae": test_metrics["mae"],
                        "calibration_factor": calibration_factor,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary_df["selected_final_model"] = summary_df["model_name"].eq(best_model_name)
    summary_df.loc[
        (summary_df["model_name"] == best_model_name) & (summary_df["split"] == "validation"),
        "calibration_factor",
    ] = calibration_factor

    result_columns = [
        "decision_week",
        "brand",
        "finished_good_sku",
        "sales_channel",
        "promo_flag",
        "forecast_4w",
        "actual_4w",
        "error",
        "abs_error",
        "ape",
        "bias",
        "hit_20_flag",
    ]
    forecast_result = test_result[result_columns].sort_values(
        ["decision_week", "brand", "finished_good_sku", "sales_channel"]
    )

    forecast_result.to_csv(OUTPUT_DIR / "02_forecast_result.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "model_selection_summary.csv", index=False, encoding="utf-8-sig")
    return forecast_result


if __name__ == "__main__":
    train_and_select_forecast_model()
