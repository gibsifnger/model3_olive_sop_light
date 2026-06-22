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
    # [WHY] 공통 SKU 재고가 부족할 때 단순 수요비례 배분만으로는 행사·서비스 손실·전략·마진 차이를 반영할 수 없다.
    # [ASSUMPTION] 0.35·0.15·0.20·0.18·0.12 가중치는 실제 최적화 결과가 아닌 synthetic 정책 가정이다.
    # [DESIGN LOGIC] 수요비중을 가장 크게 두고 프로모션, 서비스 패널티, 전략 중요도, 마진을 합산해 해석 가능한 우선순위 점수를 만든다.
    # [DATA LINEAGE] demand_share와 priority_score는 outputs/04_allocation_plan.csv에 직접 저장되고 03_sku_shortage_summary.csv의 max score에 간접 반영된다.
    # [REAL DATA REPLACEMENT] SLA 위약금, 거래처 등급, 확정 프로모션, 공헌이익, 국가·영업 전략과 정책 승인값으로 교체해야 한다.
    # [INTERVIEW CHECK] 가중치 합이 1인 것은 설명 편의를 위한 정책 설계이며 실제 적용 시 민감도·공정성·부서 합의 검증이 필요하다.
    # ============================================================
    allocation_df = allocation_df.copy()
    # 프로모션 정보가 없는 예측 결과도 일반 판매수요로 배분할 수 있도록 기본값을 적용한다.
    if "promo_flag" not in allocation_df.columns:
        allocation_df["promo_flag"] = 0  # 프로모션 여부

    allocation_df["demand_share"] = np.where(
        allocation_df["total_forecast_4w"] > 0,
        allocation_df["forecast_4w"] / allocation_df["total_forecast_4w"],
        0.0,
    )  # SKU 총수요 중 채널이 차지하는 비중으로 기본 공급 필요도를 표현한다.
    allocation_df["priority_score"] = (
        0.35 * allocation_df["demand_share"]  # 실제 수요 비중을 가장 크게 반영해 물량 규모와 배분량의 정합성을 확보
        + 0.15 * allocation_df["promo_flag"]  # 확정 행사 중 결품으로 인한 판촉비 손실을 배분 우선순위에 반영
        + 0.20 * allocation_df["service_penalty_weight"]  # 채널 결품 시 SLA·고객경험 손실을 반영
        + 0.18 * allocation_df["strategic_weight"]  # 회사가 전략적으로 보호할 채널 중요도를 반영
        + 0.12 * allocation_df["margin_weight"]  # 제한재고의 수익성 기여도를 보조 기준으로 반영
    )  # 수요·행사·서비스·전략·마진을 종합한 채널 배분 우선순위
    return allocation_df


def _allocate_one_sku(group: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] SKU별 제한재고 배분
    # [현업 의미] SKU 단위 가용재고가 부족할 때 우선순위가 높은 채널부터 forecast를 충족시킨다.
    # [판단 기준] 가용재고, 결품수량, priority_score, 채널별 forecast_4w
    # [산출물] allocation_qty, allocation_reason
    # [수정 포인트] 실무 적용 시 채널별 최소 보장 물량, 부분출고 정책, 국가별 선적 컷오프, 거래처 페널티를 추가한다.
    # [WHY] 부족한 SKU 재고를 채널별 실행 가능한 공급수량으로 변환해야 결품 영향과 후속 발주 필요량을 확인할 수 있다.
    # [ASSUMPTION] 채널별 최소 배분율보다 priority score 순차 충족을 우선하며 수량 분할과 선적 제약이 없다고 가정한다.
    # [DESIGN LOGIC] shortage가 없으면 전량 공급하고, 부족하면 priority score와 forecast가 높은 채널부터 잔여재고를 소진한다.
    # [DATA LINEAGE] allocation_qty와 allocation_reason은 outputs/04_allocation_plan.csv에 직접 저장되고 03_sku_shortage_summary.csv의 충족량에 집계된다.
    # [REAL DATA REPLACEMENT] 확정오더, 최소 공급률, 채널별 case pack, 선적 컷오프, 부분출고·공정성 정책을 반영해야 한다.
    # [INTERVIEW CHECK] channel_master의 allocation_min_rate가 현재 배분식에 직접 사용되지 않는 점은 실제 적용 전 정책 확인이 필요한 설계 한계다.
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
        channel_forecast = float(group.at[idx, "forecast_4w"])  # 현재 우선순위 채널이 향후 4주에 필요로 하는 수량
        allocated = min(channel_forecast, remaining_inventory)  # 채널 수요와 잔여 가용재고 중 작은 수량만 실제 공급
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
    # [WHY] 채널별 allocation 결과를 구매 판단 단위인 SKU로 집계해야 총 부족량·충족률·대표 리드타임을 Step 5에 전달할 수 있다.
    # [ASSUMPTION] 채널 lead_time을 forecast 수량으로 가중한 평균이 SKU의 replenishment 리드타임을 대표한다고 가정한다.
    # [DESIGN LOGIC] SKU 키로 forecast·allocation·unfulfilled를 합산하고 리드타임은 수요가 큰 채널의 영향을 더 크게 반영한다.
    # [DATA LINEAGE] 집계 결과는 outputs/03_sku_shortage_summary.csv에 직접 저장되고 outputs/05_replenishment_decision.csv에 직접 입력된다.
    # [REAL DATA REPLACEMENT] 공급사·생산·운송·통관 리드타임, 변동성, 납기준수율과 SKU sourcing route로 교체해야 한다.
    # [INTERVIEW CHECK] 판매채널 리드타임과 조달 리드타임이 동일하지 않을 수 있어 실제 적용 시 정의 확인이 필요하다고 설명해야 한다.
    # ============================================================
    allocation_df = allocation_df.copy()
    allocation_df["lead_time_weighted_qty"] = allocation_df["lead_time_week"] * allocation_df["forecast_4w"]  # 수요 규모가 큰 채널의 납기 영향을 더 크게 반영하는 가중 리드타임 분자
    summary = (
        allocation_df.groupby(SKU_KEYS, as_index=False)
        .agg(
            total_forecast_4w=("forecast_4w", "sum"),  # SKU 전체 채널의 향후 4주 총 예측수요
            available_inventory=("available_inventory", "first"),  # 모든 채널이 공동으로 사용하는 SKU 가용재고
            shortage_qty=("shortage_qty", "first"),  # 총 예측수요 대비 SKU 공통 재고 부족수량
            total_allocation_qty=("allocation_qty", "sum"),  # 채널에 실제 배분한 SKU 총수량
            total_unfulfilled_qty=("unfulfilled_qty", "sum"),  # 재고 제약으로 공급하지 못한 SKU 총수량
            max_priority_score=("priority_score", "max"),  # 해당 SKU에서 가장 우선 보호해야 하는 채널의 점수
            lead_time_weighted_qty=("lead_time_weighted_qty", "sum"),  # 수요가중 평균 리드타임 계산용 합계
        )
    )
    summary["lead_time_week"] = np.where(
        summary["total_forecast_4w"] > 0,
        summary["lead_time_weighted_qty"] / summary["total_forecast_4w"],
        4.0,
    )
    summary = summary.drop(columns=["lead_time_weighted_qty"])
    summary["shortage_flag"] = summary["shortage_qty"] > 0  # SKU 단위로 채널 배분 전 공급 부족이 존재하는지 표시
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
    # [WHY] unconstrained forecast를 현재 가용재고와 비교해 실제 공급 가능한 계획과 미충족 수요로 전환해야 S&OP 공급회의에 사용할 수 있다.
    # [ASSUMPTION] inventory snapshot이 forecast 기준시점과 일치하고 inbound는 현재 allocation 가능 재고에 포함하지 않는다고 가정한다.
    # [DESIGN LOGIC] SKU 총 forecast와 available inventory로 shortage를 계산한 뒤 채널 priority에 따라 allocation하고 SKU 요약을 별도로 생성한다.
    # [DATA LINEAGE] outputs/02_forecast_result.csv, data/inventory_snapshot.csv, channel_master.csv를 읽어 03_sku_shortage_summary.csv와 04_allocation_plan.csv를 직접 생성한다.
    # [REAL DATA REPLACEMENT] 실시간 ATP, 예약재고, 확정오더, 안전재고, 입고 가용일, 채널별 최소 공급 및 승인정책이 필요하다.
    # [INTERVIEW CHECK] forecast와 재고 snapshot의 기준시각 불일치가 shortage를 왜곡할 수 있으므로 cut-off 정합성을 설명해야 한다.
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_result = pd.read_csv(OUTPUT_DIR / "02_forecast_result.csv")
    inventory_snapshot = pd.read_csv(DATA_DIR / "inventory_snapshot.csv")
    channel_master = pd.read_csv(DATA_DIR / "channel_master.csv")

    forecast_cols = [
        "decision_week",  # 배분 판단 기준 주차
        "brand",  # 브랜드별 공급계획 구분 단위
        "finished_good_sku",  # 공통 재고를 공유하는 완제품 단위
        "sales_channel",  # 제한재고 배분 대상 채널
        "forecast_4w",  # 채널별 향후 4주 공급 필요수량
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
    allocation_df["forecast_4w"] = allocation_df["forecast_4w"].clip(lower=0)  # 음수 예측이 공급 가능량을 왜곡하지 않도록 배분 수요 하한을 0으로 설정
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
    allocation_df["allocation_qty"] = allocation_df["allocation_qty"].clip(lower=0)  # 채널별 실제 공급계획 수량의 하한을 0으로 제한
    allocation_df["unfulfilled_qty"] = (allocation_df["forecast_4w"] - allocation_df["allocation_qty"]).clip(lower=0)  # 채널별 미충족수량
    allocation_df["allocation_fill_rate"] = np.where(
        allocation_df["forecast_4w"] > 0,
        allocation_df["allocation_qty"] / allocation_df["forecast_4w"],
        1.0,
    )  # 채널 예측수요 중 실제 배분으로 충족한 비율
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
        "decision_week",  # 배분 의사결정 기준 주차
        "brand",  # 브랜드별 공급계획 구분 단위
        "finished_good_sku",  # 재고 제약을 적용한 완제품 단위
        "sales_channel",  # 배분 결과를 확인할 판매채널
        "forecast_4w",  # 채널별 향후 4주 예측수요
        "total_forecast_4w",  # SKU 전체 채널의 향후 4주 총 예측수요
        "available_inventory",  # 채널에 공동 배분할 수 있는 SKU 가용재고
        "shortage_qty",  # SKU 총수요 대비 가용재고 부족수량
        "demand_share",  # SKU 총수요에서 채널이 차지하는 비중
        "priority_score",  # 제한재고 배분 순서를 정하는 종합점수
        "allocation_qty",  # 해당 채널에 최종 배분한 수량
        "unfulfilled_qty",  # 채널 예측수요 중 공급하지 못한 수량
        "allocation_fill_rate",  # 채널 수요 대비 배분 충족률
        "allocation_reason",  # 전량 또는 제한 배분의 원인을 설명하는 사유로그
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
