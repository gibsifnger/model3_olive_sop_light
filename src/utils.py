from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")


def build_summary_table() -> pd.DataFrame:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_result = pd.read_csv(OUTPUT_DIR / "02_forecast_result.csv")
    allocation_plan = pd.read_csv(OUTPUT_DIR / "04_allocation_plan.csv")
    replenishment_decision = pd.read_csv(OUTPUT_DIR / "05_replenishment_decision.csv")
    model_summary = pd.read_csv(OUTPUT_DIR / "model_selection_summary.csv")

    selected_rows = model_summary[model_summary["selected_final_model"] == True]
    selected_model = selected_rows["model_name"].iloc[0]

    validation_row = model_summary[
        (model_summary["split"] == "validation") & (model_summary["model_name"] == selected_model)
    ].iloc[0]
    test_row = model_summary[
        (model_summary["split"] == "test") & (model_summary["model_name"] == selected_model)
    ].iloc[0]

    sku_level = allocation_plan[
        ["decision_week", "brand", "finished_good_sku", "shortage_qty"]
    ].drop_duplicates()
    order_actions = replenishment_decision[
        replenishment_decision["selected_action"].isin(
            ["order_moq", "order_2w_cover", "order_with_timing_check"]
        )
    ]

    summary_table = pd.DataFrame(
        [
            {
                "selected_model": selected_model,
                "calibration_factor": test_row["calibration_factor"],
                "validation_wape": validation_row["wape"],
                "test_wape": test_row["wape"],
                "test_forecast_accuracy": test_row["forecast_accuracy"],
                "test_bias_pct": test_row["bias_pct"],
                "test_hit_rate_20": test_row["hit_rate_20"],
                "test_mae": test_row["mae"],
                "total_sku_count": sku_level[["brand", "finished_good_sku"]].drop_duplicates().shape[0],
                "total_sku_channel_count": allocation_plan[
                    ["decision_week", "brand", "finished_good_sku", "sales_channel"]
                ].drop_duplicates().shape[0],
                "shortage_sku_count": sku_level[sku_level["shortage_qty"] > 0].shape[0],
                "avg_allocation_fill_rate": allocation_plan["allocation_fill_rate"].mean(),
                "total_unfulfilled_qty": allocation_plan["unfulfilled_qty"].sum(),
                "order_action_count": len(order_actions),
                "timing_check_count": (
                    replenishment_decision["selected_action"] == "order_with_timing_check"
                ).sum(),
                "reduce_review_count": (
                    replenishment_decision["selected_action"] == "reduce_review"
                ).sum(),
            }
        ]
    )

    summary_table.to_csv(OUTPUT_DIR / "06_summary_table.csv", index=False, encoding="utf-8-sig")
    return summary_table
