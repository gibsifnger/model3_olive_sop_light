from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
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


KEY_COLUMNS = ["brand", "finished_good_sku", "sales_channel"]


def _add_history_features(group: pd.DataFrame) -> pd.DataFrame:
    group = group.sort_values("week").copy()
    sales = group["sales_qty"]

    group["lag_1"] = sales.shift(1)
    group["lag_4"] = sales.shift(4)
    group["roll_4"] = sales.shift(1).rolling(window=4, min_periods=1).mean()
    group["roll_8"] = sales.shift(1).rolling(window=8, min_periods=1).mean()
    group["sales_std_4"] = sales.shift(1).rolling(window=4, min_periods=2).std()
    group["actual_4w"] = sales.shift(-1).rolling(window=4, min_periods=4).sum().shift(-3)

    return group


def _add_codes(feature_df: pd.DataFrame) -> pd.DataFrame:
    feature_df = feature_df.copy()
    feature_df["brand_code"] = pd.Categorical(feature_df["brand"]).codes
    feature_df["sku_type_code"] = pd.Categorical(feature_df["sku_type"]).codes
    feature_df["channel_code"] = pd.Categorical(feature_df["sales_channel"]).codes
    return feature_df


def build_feature_table() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sales_history = pd.read_csv(DATA_DIR / "sales_history.csv")
    sku_master = pd.read_csv(DATA_DIR / "sku_master.csv")

    feature_parts = []
    for key_values, group in sales_history.groupby(KEY_COLUMNS, sort=False):
        enriched = _add_history_features(group).copy()
        for column, value in zip(KEY_COLUMNS, key_values):
            enriched[column] = value
        feature_parts.append(enriched)

    feature_df = pd.concat(feature_parts, ignore_index=True)
    feature_df = feature_df.merge(
        sku_master[["brand", "finished_good_sku", "sku_type"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    feature_df["decision_week"] = feature_df["week"]
    feature_df["week_sin"] = np.sin(2 * np.pi * feature_df["decision_week"] / 52)
    feature_df["week_cos"] = np.cos(2 * np.pi * feature_df["decision_week"] / 52)
    feature_df = _add_codes(feature_df)

    feature_df["sales_std_4"] = feature_df["sales_std_4"].fillna(0)
    feature_df[["lag_1", "lag_4", "roll_4", "roll_8"]] = feature_df[
        ["lag_1", "lag_4", "roll_4", "roll_8"]
    ].fillna(0)
    feature_df["recent_trend"] = feature_df["roll_4"] - feature_df["roll_8"]
    feature_df["growth_ratio"] = feature_df["roll_4"] / np.maximum(feature_df["roll_8"], 1)
    feature_df["lag1_vs_lag4"] = feature_df["lag_1"] / np.maximum(feature_df["lag_4"], 1)

    feature_df = feature_df[feature_df["decision_week"].between(1, 56)].copy()
    feature_df = feature_df[
        [
            "decision_week",
            "brand",
            "finished_good_sku",
            "sales_channel",
            *FEATURE_COLUMNS,
            "actual_4w",
        ]
    ].sort_values(["decision_week", "brand", "finished_good_sku", "sales_channel"])

    feature_df.to_csv(OUTPUT_DIR / "01_feature_table.csv", index=False, encoding="utf-8-sig")
    return feature_df


if __name__ == "__main__":
    build_feature_table()
