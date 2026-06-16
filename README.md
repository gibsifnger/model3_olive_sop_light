# K-Beauty S&OP Replenishment MVP

## 1. Project Overview

This project builds a lightweight S&OP replenishment MVP for a high-growth K-beauty brand company.

The core flow is:

```text
channel-level demand forecast
-> shared SKU inventory shortage check
-> channel allocation priority and allocation quantity
-> SKU-level replenishment action
-> forecast accuracy and decision trace
```

This is not designed as a pure forecasting benchmark. It is designed to show how a 4-week forecast can become an executable inventory allocation and replenishment decision.

## 2. Business Problem

Fast-growing beauty brands often sell the same finished good SKU across several sales channels such as D2C, domestic retail, Amazon US, and Japan offline. Inventory is usually managed as shared finished goods inventory at the brand x SKU level, while demand signals arrive by sales channel.

The business question is:

```text
If channel forecasts exceed shared available inventory, which channel should receive inventory first,
and what replenishment action should the SKU owner take?
```

## 3. Decision Row Unit

Forecast and allocation rows are defined at:

```text
decision_week x brand x finished_good_sku x sales_channel
```

Replenishment action rows are defined at:

```text
decision_week x brand x finished_good_sku
```

Important definitions:

- `sales_channel` is a selling channel, not a purchasing channel.
- `finished_good_sku` is a sellable finished product SKU, not a raw material or component.
- Inventory is shared at `brand x finished_good_sku`, not by sales channel.
- `actual_4w` is used only for model training and accuracy evaluation.
- Allocation and replenishment decisions use `forecast_4w`, not `actual_4w`.

## 4. Data Schema

Synthetic input files are generated under `data/`.

- `sales_history.csv`: weekly sales history by brand, finished good SKU, and sales channel
- `sku_master.csv`: SKU type, MOQ, box multiple, shelf life, and cost fields
- `channel_master.csv`: channel priority weights, lead time, and allocation minimum rate
- `inventory_snapshot.csv`: shared available inventory by brand x finished good SKU
- `inbound_plan.csv`: inbound quantity plan by brand x finished good SKU

The synthetic data includes realistic noise:

- SKU-specific demand behavior for hero, new, slow, global, bundle, and basic SKUs
- channel-specific patterns for D2C, Domestic_Retail, Amazon_US, and Japan_Offline
- promotion uplift variability
- viral spikes
- intermittent slow-moving SKU demand
- stockout censoring where observed sales can be lower than true demand

## 5. Pipeline Structure

The pipeline is orchestrated by `run_pipeline.py`.

```text
1. generate_all_input_data()
2. build_feature_table()
3. train_and_select_forecast_model()
4. build_allocation_plan()
5. decide_replenishment_action()
6. build_summary_table()
```

Each step writes an intermediate CSV to either `data/` or `outputs/`.

## 6. Forecast Accuracy Design

The feature table is built without future leakage.

For decision week `t`, features use only values available through week `t`, such as:

- `lag_1`
- `lag_4`
- `roll_4`
- `roll_8`
- `sales_std_4`
- promo features
- calendar seasonality
- brand, SKU type, and channel codes

The target is:

```text
actual_4w = sales_qty from t+1 through t+4
```

The time split is:

- train: `decision_week <= 36`
- validation: `37 <= decision_week <= 48`
- test: `49 <= decision_week <= 56`
- weeks 57 to 60 are used only to calculate `actual_4w`

Two models are compared:

- `RandomForestRegressor`
- `HistGradientBoostingRegressor`

Validation WAPE selects the final model. Test metrics are reported only for the selected model.

Metrics:

- WAPE
- Forecast Accuracy
- Bias %
- Hit Rate +/-20%
- MAE

## 7. Allocation Logic

Allocation is calculated from `forecast_4w`.

For each `decision_week x brand x finished_good_sku`:

```text
total_forecast_4w = sum(channel forecast_4w)
shortage_qty = max(total_forecast_4w - available_inventory, 0)
```

If there is no shortage, every sales channel receives its forecast quantity.

If there is a shortage, channels are sorted by `priority_score`, and available inventory is allocated until the shared SKU inventory is exhausted.

The priority score reflects:

- demand share
- promotion flag
- service penalty weight
- strategic weight
- margin weight

The allocation output includes:

- `allocation_qty`
- `unfulfilled_qty`
- `allocation_fill_rate`
- `allocation_reason`

## 8. Replenishment Action Logic

Replenishment is decided at the SKU level.

Core calculations:

```text
net_requirement = total_forecast_4w - available_inventory - inbound_qty_4w
inventory_cover_week = available_inventory / max(total_forecast_4w / 4, 1)
```

The decision uses:

- MOQ
- box multiple
- shelf life
- lead time
- shortage status
- inbound coverage

Possible actions:

- `hold`
- `order_moq`
- `order_2w_cover`
- `order_with_timing_check`
- `hold_with_inbound_monitoring`
- `reduce_review`

Recommended order quantity is rounded up to the SKU box multiple and must satisfy MOQ when an order action is selected.

## 9. Output Files

Pipeline outputs are written under `outputs/`.

- `01_feature_table.csv`: model feature table and `actual_4w` target
- `02_forecast_result.csv`: test forecast result with row-level accuracy fields
- `03_sku_shortage_summary.csv`: SKU-level forecast, inventory, shortage, and fill-rate summary
- `04_allocation_plan.csv`: channel-level allocation plan and decision trace
- `05_replenishment_decision.csv`: SKU-level replenishment action
- `06_summary_table.csv`: portfolio-level KPI summary
- `model_selection_summary.csv`: validation model comparison and selected model test metrics

## 10. How to Run

Install dependencies:

```powershell
pip install -r requirements.txt
```

Run the full pipeline:

```powershell
python run_pipeline.py
```

On this local Windows setup, the virtual environment command is:

```powershell
.\.venv\Scripts\python.exe run_pipeline.py
```

## 11. Portfolio Interpretation

This project is a synthetic-data MVP. The main value is not the forecast accuracy number itself.

The focus is to validate the decision structure:

```text
time-based Forecast Accuracy
-> forecast_4w
-> shared SKU shortage
-> channel allocation priority
-> allocation quantity
-> replenishment action
-> summary KPI
```

This project is synthetic data based, so the emphasis is not the absolute forecast accuracy value. The emphasis is defining Forecast Accuracy through time-ordered backtesting and validating the decision architecture that connects `forecast_4w` to allocation priority and replenishment action.
