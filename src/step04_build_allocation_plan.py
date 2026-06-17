"""
[FILE PURPOSE]
- SKU 공통 가용재고가 4주 예측수요를 모두 충족하지 못할 때 판매채널별 우선순위에 따라 재고를 배분한다.
- 단순 forecast 결과를 실제 공급 제약 하의 allocation plan과 SKU 결품 요약으로 변환한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차

[INPUT]
- outputs/02_forecast_result.csv: SKU·채널별 forecast_4w
- data/inventory_snapshot.csv: SKU별 available_inventory
- data/channel_master.csv: 채널별 서비스 패널티, 전략 중요도, 마진 중요도, 리드타임

[OUTPUT]
- outputs/03_sku_shortage_summary.csv: SKU 단위 결품수량, fill rate, 평균 리드타임
- outputs/04_allocation_plan.csv: 채널별 배분수량, 미충족수량, 배분 사유로그

[현업 적용 시 교체 대상]
- WMS 가용재고, 채널별 SLA/우선순위, 수출·내수 리드타임, 영업 전략 가중치, 채널별 최소 공급 정책으로 교체한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


SKU_KEYS = ["decision_week", "brand", "finished_good_sku"]  # SKU 공통 재고를 공유하는 배분 판단 단위


def _calculate_priority(allocation_df: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 채널별 배분 우선순위 산정
    # [현업 의미] 제한된 SKU 재고를 어떤 판매채널에 먼저 공급할지 결정하기 위한 우선순위 점수를 만든다.
    # [판단 기준] 수요 비중, 프로모션 여부, 서비스 패널티, 전략 중요도, 마진 중요도
    # [산출물] demand_share, priority_score
    # [수정 포인트] 실무 적용 시 핵심 거래처 SLA, 페널티 계약, 영업 우선순위, 마진율, 국가별 공급전략을 반영한다.
    # ============================================================
    allocation_df = allocation_df.copy()
    # 프로모션 정보가 없는 예측 결과도 일반 판매수요로 배분할 수 있도록 기본값을 적용한다.
    if "promo_flag" not in allocation_df.columns:
        allocation_df["promo_flag"] = 0  # 프로모션 여부

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
    # ============================================================
    # [BLOCK] SKU별 제한재고 배분
    # [현업 의미] SKU 단위 가용재고가 부족할 때 우선순위가 높은 채널부터 forecast를 충족시킨다.
    # [판단 기준] 가용재고, 결품수량, priority_score, 채널별 forecast_4w
    # [산출물] allocation_qty, allocation_reason
    # [수정 포인트] 실무 적용 시 채널별 최소 보장 물량, 부분출고 정책, 국가별 선적 컷오프, 거래처 페널티를 추가한다.
    # ============================================================
    group = group.copy()
    available_inventory = float(group["available_inventory"].iloc[0])  # 채널 배분에 사용할 SKU 공통 가용재고
    shortage_qty = float(group["shortage_qty"].iloc[0])  # 4주 forecast 대비 부족한 SKU 결품수량

    # 가용재고가 총 forecast를 커버하면 채널별 예측수요를 그대로 공급한다.
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
        # 남은 가용재고가 없으면 추가 채널 배분은 결품으로 남긴다.
        if remaining_inventory <= 0:
            break

    group["allocation_reason"] = np.where(  # 채널별 배분 결과를 설명하는 사유로그
        group["allocation_qty"] >= group["forecast_4w"],
        "Shortage: high priority or sufficient remaining inventory",
        "Shortage: constrained by SKU common inventory",
    )
    return group


def _build_shortage_summary(allocation_df: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] SKU 단위 결품 요약
    # [현업 의미] 채널별 배분 결과를 SKU 단위로 집계해 발주 판단에 필요한 결품 규모와 평균 리드타임을 산출한다.
    # [판단 기준] 총 forecast, 가용재고, 결품수량, 미충족수량, 채널별 forecast 가중 리드타임
    # [산출물] outputs/03_sku_shortage_summary.csv 입력용 summary
    # [수정 포인트] 실무 적용 시 공급사 리드타임, 생산 리드타임, 선적/통관 리드타임을 SKU별로 세분화한다.
    # ============================================================
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
    # ============================================================
    # [BLOCK] 재고 제약 기반 채널 배분 계획 생성
    # [현업 의미] 예측수요를 현재 가용재고와 비교해 결품 여부를 확인하고, 채널 우선순위에 따라 공급 가능 물량을 배분한다.
    # [판단 기준] forecast_4w, available_inventory, shortage_qty, priority_score, allocation_fill_rate
    # [산출물] outputs/03_sku_shortage_summary.csv, outputs/04_allocation_plan.csv
    # [수정 포인트] 실무 적용 시 실시간 ATP, 거래처별 확정오더, 안전재고, 채널별 최소 공급률을 반영한다.
    # ============================================================
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

    allocation_df["available_inventory"] = allocation_df["available_inventory"].fillna(0)  # 현재 배분 가능한 SKU별 가용재고
    allocation_df["forecast_4w"] = allocation_df["forecast_4w"].clip(lower=0)
    allocation_df["total_forecast_4w"] = allocation_df.groupby(SKU_KEYS)["forecast_4w"].transform("sum")  # SKU 단위 4주 총 예측수요
    allocation_df["shortage_qty"] = (
        allocation_df["total_forecast_4w"] - allocation_df["available_inventory"]
    ).clip(lower=0)  # 가용재고로 커버하지 못하는 SKU 단위 결품수량
    allocation_df = _calculate_priority(allocation_df)

    allocation_parts = []
    for key_values, group in allocation_df.groupby(SKU_KEYS, sort=False):
        enriched = _allocate_one_sku(group).copy()
        for column, value in zip(SKU_KEYS, key_values):
            enriched[column] = value
        allocation_parts.append(enriched)
    allocation_df = pd.concat(allocation_parts, ignore_index=True)
    allocation_df["allocation_qty"] = allocation_df["allocation_qty"].clip(lower=0)
    allocation_df["unfulfilled_qty"] = (allocation_df["forecast_4w"] - allocation_df["allocation_qty"]).clip(lower=0)  # 채널별 미충족수량
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
