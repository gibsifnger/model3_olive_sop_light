from pathlib import Path
import math

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def _ceil_to_multiple(value: float, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(math.ceil(value / multiple) * multiple)


def _recommended_order_qty(row: pd.Series, selected_action: str) -> int:
    if selected_action in {"hold", "hold_with_inbound_monitoring", "reduce_review"}:
        return 0

    weekly_forecast = max(row["total_forecast_4w"] / 4, 1)
    if selected_action == "order_2w_cover":
        base_qty = max(row["net_requirement"], weekly_forecast * 2)
    elif selected_action == "order_with_timing_check":
        base_qty = row["net_requirement"] + weekly_forecast
    else:
        base_qty = row["net_requirement"]

    return _ceil_to_multiple(max(base_qty, row["moq"]), int(row["box_multiple"]))


def _decide_one_row(row: pd.Series) -> dict[str, object]:
    shortage_qty = row.get("shortage_qty", max(row["total_forecast_4w"] - row["available_inventory"], 0))
    inbound_covers_shortage = shortage_qty > 0 and row["available_inventory"] + row["inbound_qty_4w"] >= row["total_forecast_4w"]
    short_shelf_life = row["shelf_life_week"] <= 52
    reduce_review_candidate = row["sku_type"] in {"slow", "basic", "bundle"}
    excessive_cover_limit = 8 if short_shelf_life else 12

    if (
        row["net_requirement"] <= 0
        and row["inventory_cover_week"] >= 8
        and reduce_review_candidate
    ):
        selected_action = "reduce_review"
        reason_1 = "Net requirement is covered"
        reason_2 = "Inventory cover is high for SKU type with overstock risk"
    elif row["net_requirement"] <= 0 and inbound_covers_shortage:
        selected_action = "hold_with_inbound_monitoring"
        reason_1 = "Current shortage is expected to be covered by inbound within 4 weeks"
        reason_2 = "Monitor inbound execution before placing a new order"
    elif row["net_requirement"] <= 0:
        selected_action = "hold"
        reason_1 = "Available inventory plus inbound covers 4-week forecast"
        reason_2 = "No incremental replenishment required"
    elif row["inventory_cover_week"] < row["lead_time_week"]:
        selected_action = "order_with_timing_check"
        reason_1 = "Inventory cover is shorter than lead time"
        reason_2 = "Order is needed and timing risk should be checked"
    elif row["net_requirement"] <= row["moq"]:
        selected_action = "order_moq"
        reason_1 = "Net requirement is positive but below MOQ"
        reason_2 = "Order minimum quantity to satisfy supplier constraint"
    else:
        selected_action = "order_2w_cover"
        reason_1 = "Net requirement exceeds MOQ"
        reason_2 = "Order enough to restore about 2 weeks of cover"

    recommended_qty = _recommended_order_qty(row, selected_action)
    projected_cover_after_order = (
        row["available_inventory"] + row["inbound_qty_4w"] + recommended_qty
    ) / max(row["total_forecast_4w"] / 4, 1)

    warnings = []
    if row["inventory_cover_week"] < row["lead_time_week"]:
        warnings.append("cover_below_lead_time")
    if short_shelf_life and row["inventory_cover_week"] >= excessive_cover_limit:
        warnings.append("short_shelf_life_high_cover")
    if projected_cover_after_order > excessive_cover_limit and reduce_review_candidate:
        warnings.append("possible_overstock_or_expiry")
    if selected_action.startswith("order") and recommended_qty < row["moq"]:
        warnings.append("below_moq")

    gate_status = "pass"
    if selected_action in {"order_with_timing_check", "reduce_review"}:
        gate_status = "review"
    if "possible_overstock_or_expiry" in warnings:
        gate_status = "review"

    return {
        "recommended_order_qty": recommended_qty,
        "selected_action": selected_action,
        "gate_status": gate_status,
        "warning": "; ".join(warnings) if warnings else "none",
        "reason_1": reason_1,
        "reason_2": reason_2,
    }


def decide_replenishment_action() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    shortage_summary = pd.read_csv(OUTPUT_DIR / "03_sku_shortage_summary.csv")
    sku_master = pd.read_csv(DATA_DIR / "sku_master.csv")
    inbound_plan = pd.read_csv(DATA_DIR / "inbound_plan.csv")

    decision_df = shortage_summary.merge(
        sku_master[
            [
                "brand",
                "finished_good_sku",
                "sku_type",
                "moq",
                "box_multiple",
                "shelf_life_week",
            ]
        ],
        on=["brand", "finished_good_sku"],
        how="left",
    )
    decision_df = decision_df.merge(
        inbound_plan[["brand", "finished_good_sku", "inbound_qty_4w"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    decision_df["inbound_qty_4w"] = decision_df["inbound_qty_4w"].fillna(0)
    decision_df["lead_time_week"] = decision_df["lead_time_week"].fillna(4)
    decision_df["net_requirement"] = (
        decision_df["total_forecast_4w"]
        - decision_df["available_inventory"]
        - decision_df["inbound_qty_4w"]
    )
    decision_df["inventory_cover_week"] = decision_df["available_inventory"] / np.maximum(
        decision_df["total_forecast_4w"] / 4,
        1,
    )

    action_rows = decision_df.apply(_decide_one_row, axis=1, result_type="expand")
    decision_df = pd.concat([decision_df, action_rows], axis=1)

    output_columns = [
        "decision_week",
        "brand",
        "finished_good_sku",
        "total_forecast_4w",
        "available_inventory",
        "inbound_qty_4w",
        "net_requirement",
        "inventory_cover_week",
        "recommended_order_qty",
        "selected_action",
        "gate_status",
        "warning",
        "reason_1",
        "reason_2",
    ]
    replenishment_decision = decision_df[output_columns].sort_values(
        ["decision_week", "brand", "finished_good_sku"]
    )
    for column in [
        "total_forecast_4w",
        "available_inventory",
        "inbound_qty_4w",
        "net_requirement",
        "recommended_order_qty",
    ]:
        replenishment_decision[column] = replenishment_decision[column].round().astype(int)
    replenishment_decision.to_csv(
        OUTPUT_DIR / "05_replenishment_decision.csv",
        index=False,
        encoding="utf-8-sig",
    )
    return replenishment_decision


if __name__ == "__main__":
    decide_replenishment_action()
