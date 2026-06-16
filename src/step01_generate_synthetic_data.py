from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RANDOM_SEED = 42


def _build_sku_master() -> pd.DataFrame:
    rows = [
        ("GlowLab", "GL_HERO_SERUM", "hero", 500, 50, 104, 8.5, 18.0, 0.08, 3.2),
        ("GlowLab", "GL_NEW_AMPOULE", "new", 300, 30, 78, 7.2, 15.0, 0.07, 2.8),
        ("PureLeaf", "PL_SLOW_MASK", "slow", 200, 20, 52, 3.6, 7.5, 0.04, 1.4),
        ("PureLeaf", "PL_GLOBAL_CREAM", "global", 400, 40, 104, 6.8, 16.0, 0.06, 2.5),
        ("AquaMuse", "AM_BUNDLE_KIT", "bundle", 250, 25, 52, 11.5, 22.0, 0.10, 4.0),
        ("AquaMuse", "AM_BASIC_TONER", "basic", 350, 35, 104, 4.4, 9.0, 0.05, 1.6),
    ]
    columns = [
        "brand",
        "finished_good_sku",
        "sku_type",
        "moq",
        "box_multiple",
        "shelf_life_week",
        "unit_cost",
        "stockout_cost",
        "holding_cost",
        "disposal_cost",
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_channel_master() -> pd.DataFrame:
    rows = [
        ("D2C", "owned_online", 1.15, 1.25, 1.40, 1, 0.18),
        ("Domestic_Retail", "domestic_wholesale", 1.05, 1.05, 0.95, 2, 0.22),
        ("Amazon_US", "marketplace_global", 1.45, 1.35, 1.20, 4, 0.20),
        ("Japan_Offline", "export_offline", 1.30, 1.15, 0.90, 5, 0.25),
    ]
    columns = [
        "sales_channel",
        "channel_type",
        "service_penalty_weight",
        "strategic_weight",
        "margin_weight",
        "lead_time_week",
        "allocation_min_rate",
    ]
    return pd.DataFrame(rows, columns=columns)


def _sku_base_demand(sku_type: str) -> float:
    return {
        "hero": 520,
        "new": 230,
        "slow": 65,
        "global": 210,
        "bundle": 150,
        "basic": 260,
    }[sku_type]


def _channel_multiplier(sku_type: str, sales_channel: str, week: int) -> float:
    base = {
        "D2C": 1.05,
        "Domestic_Retail": 0.85,
        "Amazon_US": 0.75,
        "Japan_Offline": 0.55,
    }[sales_channel]

    if sku_type == "global" and sales_channel in {"Amazon_US", "Japan_Offline"}:
        base *= 1.0 + 0.012 * week
    if sku_type == "bundle" and sales_channel in {"D2C", "Domestic_Retail"}:
        base *= 1.15
    if sku_type == "slow" and sales_channel in {"Amazon_US", "Japan_Offline"}:
        base *= 0.55
    if sku_type == "new" and week <= 12:
        base *= np.interp(week, [1, 12], [0.45, 1.20])
    if week >= 50:
        base *= 0.95

    return base


def _promo_probability(sku_type: str, sales_channel: str) -> float:
    sku_effect = {
        "hero": 0.22,
        "new": 0.16,
        "slow": 0.08,
        "global": 0.14,
        "bundle": 0.30,
        "basic": 0.12,
    }[sku_type]
    channel_effect = {
        "D2C": 0.04,
        "Domestic_Retail": 0.12,
        "Amazon_US": 0.06,
        "Japan_Offline": 0.02,
    }[sales_channel]
    return min(sku_effect + channel_effect, 0.55)


def _make_channel_demand(
    rng: np.random.Generator,
    week: int,
    sku_type: str,
    sales_channel: str,
) -> tuple[int, int, float]:
    base = _sku_base_demand(sku_type) * _channel_multiplier(sku_type, sales_channel, week)
    seasonality = 1.0 + 0.10 * np.sin(2 * np.pi * week / 13) + 0.06 * np.cos(2 * np.pi * week / 26)
    trend = 1.0 + (0.003 * week if sku_type in {"hero", "basic"} else 0.0)

    noise_sigma = {
        "hero": 0.24,
        "new": 0.55 if week <= 12 else 0.30,
        "slow": 0.65,
        "global": 0.32,
        "bundle": 0.42,
        "basic": 0.18,
    }[sku_type]
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma)

    promo_flag = int(rng.random() < _promo_probability(sku_type, sales_channel))
    promo_uplift = 0.0
    if promo_flag:
        uplift_base = {
            "hero": 0.45,
            "new": 0.28,
            "slow": 0.18,
            "global": 0.25,
            "bundle": 0.65,
            "basic": 0.20,
        }[sku_type]
        channel_promo = 0.18 if sales_channel == "Domestic_Retail" else 0.08
        promo_uplift = max(0.05, rng.normal(uplift_base + channel_promo, 0.18))

    viral_probability = 0.0
    if sku_type == "hero" and sales_channel in {"D2C", "Amazon_US"}:
        viral_probability = 0.035
    elif sku_type == "global" and sales_channel == "Amazon_US":
        viral_probability = 0.030
    elif sku_type == "new" and sales_channel == "D2C" and week <= 16:
        viral_probability = 0.025

    viral_multiplier = rng.uniform(1.8, 3.6) if rng.random() < viral_probability else 1.0

    intermittent_factor = 1.0
    if sku_type == "slow":
        intermittent_factor = 0.0 if rng.random() < 0.35 else rng.uniform(0.35, 1.35)

    true_demand = base * seasonality * trend * noise * (1.0 + promo_uplift) * viral_multiplier * intermittent_factor
    true_demand = max(0, int(round(true_demand)))
    return true_demand, promo_flag, round(promo_uplift, 3)


def _apply_stockout_censoring(
    rng: np.random.Generator,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    censored_rows = []

    for _, group in demand_df.groupby(["week", "brand", "finished_good_sku"], sort=False):
        total_demand = group["true_demand"].sum()
        shortage_event = rng.random() < 0.18
        if shortage_event and total_demand > 0:
            available_for_week = int(round(total_demand * rng.uniform(0.58, 0.93)))
        else:
            available_for_week = int(round(total_demand * rng.uniform(0.98, 1.18)))

        available_for_week = max(0, available_for_week)
        allocation_weight = group["true_demand"].to_numpy(dtype=float)
        allocation_weight = allocation_weight * rng.uniform(0.85, 1.15, size=len(group))
        allocation_weight_sum = allocation_weight.sum()

        if allocation_weight_sum <= 0:
            sales_qty = np.zeros(len(group), dtype=int)
        elif available_for_week >= total_demand:
            sales_qty = group["true_demand"].to_numpy(dtype=int)
        else:
            raw_alloc = available_for_week * allocation_weight / allocation_weight_sum
            sales_qty = np.minimum(group["true_demand"].to_numpy(dtype=int), np.floor(raw_alloc).astype(int))
            remaining = available_for_week - sales_qty.sum()
            if remaining > 0:
                gaps = group["true_demand"].to_numpy(dtype=int) - sales_qty
                candidate_idx = np.where(gaps > 0)[0]
                if len(candidate_idx) > 0:
                    add_idx = rng.choice(candidate_idx, size=min(remaining, len(candidate_idx)), replace=False)
                    sales_qty[add_idx] += 1

        out = group.copy()
        out["sales_qty"] = sales_qty
        censored_rows.append(out)

    return pd.concat(censored_rows, ignore_index=True)


def _build_sales_history(
    rng: np.random.Generator,
    sku_master: pd.DataFrame,
    channel_master: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for week in range(1, 61):
        for sku in sku_master.itertuples(index=False):
            for channel in channel_master["sales_channel"]:
                true_demand, promo_flag, promo_uplift = _make_channel_demand(
                    rng=rng,
                    week=week,
                    sku_type=sku.sku_type,
                    sales_channel=channel,
                )
                rows.append(
                    {
                        "week": week,
                        "brand": sku.brand,
                        "finished_good_sku": sku.finished_good_sku,
                        "sales_channel": channel,
                        "true_demand": true_demand,
                        "promo_flag": promo_flag,
                        "promo_uplift": promo_uplift,
                    }
                )

    demand_df = pd.DataFrame(rows)
    sales_df = _apply_stockout_censoring(rng, demand_df)
    return sales_df[
        [
            "week",
            "brand",
            "finished_good_sku",
            "sales_channel",
            "sales_qty",
            "promo_flag",
            "promo_uplift",
        ]
    ].sort_values(["week", "brand", "finished_good_sku", "sales_channel"])


def _build_inventory_snapshot(
    rng: np.random.Generator,
    sales_history: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> pd.DataFrame:
    recent_sales = (
        sales_history[sales_history["week"].between(53, 60)]
        .groupby(["brand", "finished_good_sku"], as_index=False)["sales_qty"]
        .sum()
    )
    recent_sales["avg_4w_sales"] = recent_sales["sales_qty"] / 2.0
    recent_sales = recent_sales.merge(
        sku_master[["brand", "finished_good_sku", "sku_type"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    rows = []
    for row in recent_sales.itertuples(index=False):
        coverage_factor = {
            "hero": rng.uniform(1.25, 1.70),
            "new": rng.uniform(0.25, 0.45),
            "slow": rng.uniform(2.20, 3.00),
            "global": rng.uniform(0.55, 0.75),
            "bundle": rng.uniform(0.82, 0.95),
            "basic": rng.uniform(2.00, 2.70),
        }[row.sku_type]
        on_hand = int(round(row.avg_4w_sales * coverage_factor))
        blocked = int(round(on_hand * rng.uniform(0.01, 0.08)))
        available = max(0, on_hand - blocked)
        rows.append(
            {
                "decision_week": 61,
                "brand": row.brand,
                "finished_good_sku": row.finished_good_sku,
                "on_hand_inventory": on_hand,
                "blocked_inventory": blocked,
                "available_inventory": available,
            }
        )

    return pd.DataFrame(rows)


def _build_inbound_plan(
    rng: np.random.Generator,
    sales_history: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> pd.DataFrame:
    recent_sales = (
        sales_history[sales_history["week"].between(49, 60)]
        .groupby(["brand", "finished_good_sku"], as_index=False)["sales_qty"]
        .sum()
    )
    recent_sales["avg_4w_sales"] = recent_sales["sales_qty"] / 3.0
    plan = recent_sales.merge(
        sku_master[["brand", "finished_good_sku", "box_multiple"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    rows = []
    for row in plan.itertuples(index=False):
        base = row.avg_4w_sales
        sku_type = sku_master.loc[
            sku_master["finished_good_sku"].eq(row.finished_good_sku),
            "sku_type",
        ].iloc[0]
        inbound_factor = {
            "hero": rng.uniform(0.20, 0.45),
            "new": rng.uniform(0.00, 0.12),
            "slow": rng.uniform(0.00, 0.05),
            "global": rng.uniform(0.55, 0.85),
            "bundle": rng.uniform(0.08, 0.18),
            "basic": rng.uniform(0.00, 0.08),
        }[sku_type]
        inbound_1w = _round_to_multiple(base * inbound_factor * rng.uniform(0.05, 0.25), row.box_multiple)
        inbound_2w = _round_to_multiple(base * inbound_factor * rng.uniform(0.25, 0.55), row.box_multiple)
        inbound_4w = _round_to_multiple(base * inbound_factor, row.box_multiple)
        rows.append(
            {
                "decision_week": 61,
                "brand": row.brand,
                "finished_good_sku": row.finished_good_sku,
                "inbound_qty_1w": inbound_1w,
                "inbound_qty_2w": inbound_2w,
                "inbound_qty_4w": inbound_4w,
            }
        )

    return pd.DataFrame(rows)


def _round_to_multiple(value: float, multiple: int) -> int:
    if value <= 0:
        return 0
    return int(round(value / multiple) * multiple)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def generate_all_input_data() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    sku_master = _build_sku_master()
    channel_master = _build_channel_master()
    sales_history = _build_sales_history(rng, sku_master, channel_master)
    inventory_snapshot = _build_inventory_snapshot(rng, sales_history, sku_master)
    inbound_plan = _build_inbound_plan(rng, sales_history, sku_master)

    _save_csv(sales_history, DATA_DIR / "sales_history.csv")
    _save_csv(sku_master, DATA_DIR / "sku_master.csv")
    _save_csv(channel_master, DATA_DIR / "channel_master.csv")
    _save_csv(inventory_snapshot, DATA_DIR / "inventory_snapshot.csv")
    _save_csv(inbound_plan, DATA_DIR / "inbound_plan.csv")


if __name__ == "__main__":
    generate_all_input_data()
