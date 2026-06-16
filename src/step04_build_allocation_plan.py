from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


SKU_KEYS = ["decision_week", "brand", "finished_good_sku"]


def _calculate_priority(allocation_df: pd.DataFrame) -> pd.DataFrame:
    allocation_df = allocation_df.copy()
    if "promo_flag" not in allocation_df.columns:
        allocation_df["promo_flag"] = 0

    allocation_df["demand_share"] = np.where(
        allocation_df["total_forecast_4w"] > 0,
        allocation_df["forecast_4w"] / allocation_df["total_forecast_4w"],
        0.0,
    )
    allocation_df["priority_score"] = (
        0.35 * allocation_df["demand_share"]
        + 0.15 * allocation_df["promo_flag"]
        + 0.20 * allocation_df["service_penalty_weight"]
        + 0.18 * allocation_df["strategic_weight"]
        + 0.12 * allocation_df["margin_weight"]
    )
    return allocation_df


def _allocate_one_sku(group: pd.DataFrame) -> pd.DataFrame:
    group = group.copy()
    available_inventory = float(group["available_inventory"].iloc[0])
    shortage_qty = float(group["shortage_qty"].iloc[0])

    if shortage_qty <= 0:
        group["allocation_qty"] = group["forecast_4w"]
        group["allocation_reason"] = "No shortage: full forecast covered"
        return group

    remaining_inventory = max(0.0, available_inventory)
    ordered_idx = group.sort_values(
        ["priority_score", "forecast_4w"],
        ascending=[False, False],
    ).index
    group["allocation_qty"] = 0.0

    for idx in ordered_idx:
        channel_forecast = float(group.at[idx, "forecast_4w"])
        allocated = min(channel_forecast, remaining_inventory)
        group.at[idx, "allocation_qty"] = allocated
        remaining_inventory -= allocated
        if remaining_inventory <= 0:
            break

    group["allocation_reason"] = np.where(
        group["allocation_qty"] >= group["forecast_4w"],
        "Shortage: high priority or sufficient remaining inventory",
        "Shortage: constrained by SKU common inventory",
    )
    return group


def _build_shortage_summary(allocation_df: pd.DataFrame) -> pd.DataFrame:
    allocation_df = allocation_df.copy()
    allocation_df["lead_time_weighted_qty"] = allocation_df["lead_time_week"] * allocation_df["forecast_4w"]
    summary = (
        allocation_df.groupby(SKU_KEYS, as_index=False)
        .agg(
            total_forecast_4w=("forecast_4w", "sum"),
            available_inventory=("available_inventory", "first"),
            shortage_qty=("shortage_qty", "first"),
            total_allocation_qty=("allocation_qty", "sum"),
            total_unfulfilled_qty=("unfulfilled_qty", "sum"),
            max_priority_score=("priority_score", "max"),
            lead_time_weighted_qty=("lead_time_weighted_qty", "sum"),
        )
    )
    summary["lead_time_week"] = np.where(
        summary["total_forecast_4w"] > 0,
        summary["lead_time_weighted_qty"] / summary["total_forecast_4w"],
        4.0,
    )
    summary = summary.drop(columns=["lead_time_weighted_qty"])
    summary["shortage_flag"] = summary["shortage_qty"] > 0
    summary["sku_fill_rate"] = np.where(
        summary["total_forecast_4w"] > 0,
        summary["total_allocation_qty"] / summary["total_forecast_4w"],
        1.0,
    )
    return summary


def build_allocation_plan() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_result = pd.read_csv(OUTPUT_DIR / "02_forecast_result.csv")
    inventory_snapshot = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
    channel_master = pd.read_csv(DATA_DIR / "channel_master.csv")

    forecast_cols = [
        "decision_week",
        "brand",
        "finished_good_sku",
        "sales_channel",
        "forecast_4w",
    ]
    if "promo_flag" in forecast_result.columns:
        forecast_cols.append("promo_flag")

    allocation_df = forecast_result[forecast_cols].copy()
    allocation_df = allocation_df.merge(
        inventory_snapshot[["brand", "finished_good_sku", "available_inventory"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )
    allocation_df = allocation_df.merge(channel_master, on="sales_channel", how="left")

    allocation_df["available_inventory"] = allocation_df["available_inventory"].fillna(0)
    allocation_df["forecast_4w"] = allocation_df["forecast_4w"].clip(lower=0)
    allocation_df["total_forecast_4w"] = allocation_df.groupby(SKU_KEYS)["forecast_4w"].transform("sum")
    allocation_df["shortage_qty"] = (
        allocation_df["total_forecast_4w"] - allocation_df["available_inventory"]
    ).clip(lower=0)
    allocation_df = _calculate_priority(allocation_df)

    allocation_parts = []
    for key_values, group in allocation_df.groupby(SKU_KEYS, sort=False):
        enriched = _allocate_one_sku(group).copy()
        for column, value in zip(SKU_KEYS, key_values):
            enriched[column] = value
        allocation_parts.append(enriched)
    allocation_df = pd.concat(allocation_parts, ignore_index=True)
    allocation_df["allocation_qty"] = allocation_df["allocation_qty"].clip(lower=0)
    allocation_df["unfulfilled_qty"] = (allocation_df["forecast_4w"] - allocation_df["allocation_qty"]).clip(lower=0)
    allocation_df["allocation_fill_rate"] = np.where(
        allocation_df["forecast_4w"] > 0,
        allocation_df["allocation_qty"] / allocation_df["forecast_4w"],
        1.0,
    )
    for column in [
        "forecast_4w",
        "total_forecast_4w",
        "available_inventory",
        "shortage_qty",
        "allocation_qty",
        "unfulfilled_qty",
    ]:
        allocation_df[column] = allocation_df[column].round().astype(int)

    plan_columns = [
        "decision_week",
        "brand",
        "finished_good_sku",
        "sales_channel",
        "forecast_4w",
        "total_forecast_4w",
        "available_inventory",
        "shortage_qty",
        "demand_share",
        "priority_score",
        "allocation_qty",
        "unfulfilled_qty",
        "allocation_fill_rate",
        "allocation_reason",
    ]
    allocation_plan = allocation_df[plan_columns].sort_values(
        ["decision_week", "brand", "finished_good_sku", "priority_score"],
        ascending=[True, True, True, False],
    )
    shortage_summary = _build_shortage_summary(allocation_df).sort_values(
        ["decision_week", "brand", "finished_good_sku"]
    )
    for column in [
        "total_forecast_4w",
        "available_inventory",
        "shortage_qty",
        "total_allocation_qty",
        "total_unfulfilled_qty",
    ]:
        shortage_summary[column] = shortage_summary[column].round().astype(int)

    shortage_summary.to_csv(OUTPUT_DIR / "03_sku_shortage_summary.csv", index=False, encoding="utf-8-sig")
    allocation_plan.to_csv(OUTPUT_DIR / "04_allocation_plan.csv", index=False, encoding="utf-8-sig")
    return allocation_plan


if __name__ == "__main__":
    build_allocation_plan()
