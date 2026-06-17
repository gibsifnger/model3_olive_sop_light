"""
[FILE PURPOSE]
- SKU 단위 결품 요약, 입고예정, 구매조건을 결합해 최종 발주/보류/감량검토 액션을 결정한다.
- 예측수요와 재고 제약을 MOQ, 박스배수, 리드타임, 유통기한 리스크를 반영한 구매 실행 판단으로 전환한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 의사결정 주차

[INPUT]
- outputs/03_sku_shortage_summary.csv: SKU별 총 forecast, 가용재고, 결품수량, fill rate, 리드타임
- data/sku_master.csv: MOQ, 박스배수, 유통기한, SKU 유형
- data/inbound_plan.csv: 향후 4주 입고예정

[OUTPUT]
- outputs/05_replenishment_decision.csv: 추천 발주수량, 선택 액션, gate 상태, warning, 사유로그

[현업 적용 시 교체 대상]
- ERP 구매조건, 공급사 MOQ/박스배수, SAP MM 발주잔량, 입고예정, 품목별 유통기한, 안전재고 및 발주 승인 기준으로 교체한다.
"""

from pathlib import Path
import math

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


def _ceil_to_multiple(value: float, multiple: int) -> int:
    # 공급사 박스배수보다 작은 단위로 발주할 수 없으므로 추천 발주수량을 구매 가능 단위로 올림 처리한다.
    if value <= 0:
        return 0
    return int(math.ceil(value / multiple) * multiple)


def _recommended_order_qty(row: pd.Series, selected_action: str) -> int:
    # ============================================================
    # [BLOCK] 추천 발주수량 산정
    # [현업 의미] 선택된 replenishment action에 맞춰 MOQ와 박스배수를 만족하는 실행 가능한 발주수량을 계산한다.
    # [판단 기준] selected_action, net_requirement, 주간 forecast, MOQ, 박스배수
    # [산출물] recommended_order_qty
    # [수정 포인트] 실무 적용 시 안전재고, 목표 커버주수, 공급사별 주문 캘린더, 컨테이너 단위 제약을 반영한다.
    # ============================================================
    # 보류 또는 감량 검토 액션은 신규 구매 실행보다 재고 모니터링이 우선이므로 발주수량을 0으로 둔다.
    if selected_action in {"hold", "hold_with_inbound_monitoring", "reduce_review"}:
        return 0

    weekly_forecast = max(row["total_forecast_4w"] / 4, 1)
    # 리드타임 리스크가 낮고 순소요가 MOQ보다 충분하면 약 2주 커버를 회복하는 수준으로 주문한다.
    if selected_action == "order_2w_cover":
        base_qty = max(row["net_requirement"], weekly_forecast * 2)
    # 재고 커버가 리드타임보다 짧으면 입고 전 결품을 줄이기 위해 순소요에 1주 buffer를 더한다.
    elif selected_action == "order_with_timing_check":
        base_qty = row["net_requirement"] + weekly_forecast
    else:
        base_qty = row["net_requirement"]

    return _ceil_to_multiple(max(base_qty, row["moq"]), int(row["box_multiple"]))  # MOQ와 박스배수를 반영한 발주수량


def _decide_one_row(row: pd.Series) -> dict[str, object]:
    # ============================================================
    # [BLOCK] SKU별 replenishment action 결정
    # [현업 의미] 예측수요, 가용재고, 입고예정, 리드타임, MOQ, 유통기한 리스크를 종합해 구매 실행 액션을 부여한다.
    # [판단 기준] net_requirement, inventory_cover_week, lead_time_week, inbound_qty_4w, MOQ, SKU 유형, 유통기한
    # [산출물] selected_action, gate_status, warning, reason_1, reason_2
    # [수정 포인트] 실무 적용 시 안전재고, 품절 패널티, 공급사별 납기신뢰도, 폐기 리스크, 구매 승인 workflow를 반영한다.
    # ============================================================
    shortage_qty = row.get("shortage_qty", max(row["total_forecast_4w"] - row["available_inventory"], 0))  # SKU 단위 결품수량
    inbound_covers_shortage = shortage_qty > 0 and row["available_inventory"] + row["inbound_qty_4w"] >= row["total_forecast_4w"]  # 4주 입고예정으로 결품을 해소할 수 있는지 판단
    short_shelf_life = row["shelf_life_week"] <= 52
    reduce_review_candidate = row["sku_type"] in {"slow", "basic", "bundle"}
    excessive_cover_limit = 8 if short_shelf_life else 12  # 유통기한이 짧은 SKU는 더 낮은 커버주수부터 과재고 리스크로 본다.

    # 순소요가 없고 재고 커버가 높은 저회전/기본/번들 SKU는 신규 발주보다 감량 검토를 우선한다.
    if (
        row["net_requirement"] <= 0
        and row["inventory_cover_week"] >= 8
        and reduce_review_candidate
    ):
        selected_action = "reduce_review"
        reason_1 = "Net requirement is covered"
        reason_2 = "Inventory cover is high for SKU type with overstock risk"
    # 현재는 부족하지만 입고예정이 4주 forecast를 커버하면 추가 발주보다 입고 이행 모니터링을 우선한다.
    elif row["net_requirement"] <= 0 and inbound_covers_shortage:
        selected_action = "hold_with_inbound_monitoring"
        reason_1 = "Current shortage is expected to be covered by inbound within 4 weeks"
        reason_2 = "Monitor inbound execution before placing a new order"
    # 가용재고와 입고예정으로 향후 4주 수요를 커버할 수 있으면 신규 발주는 보류한다.
    elif row["net_requirement"] <= 0:
        selected_action = "hold"
        reason_1 = "Available inventory plus inbound covers 4-week forecast"
        reason_2 = "No incremental replenishment required"
    # 재고 커버주수가 리드타임보다 짧으면 입고 전 결품 가능성이 있으므로 타이밍 점검 액션을 부여한다.
    elif row["inventory_cover_week"] < row["lead_time_week"]:
        selected_action = "order_with_timing_check"
        reason_1 = "Inventory cover is shorter than lead time"
        reason_2 = "Order is needed and timing risk should be checked"
    # 순소요가 MOQ보다 작아도 공급사 최소 발주수량을 만족해야 하므로 MOQ 발주 액션을 부여한다.
    elif row["net_requirement"] <= row["moq"]:
        selected_action = "order_moq"
        reason_1 = "Net requirement is positive but below MOQ"
        reason_2 = "Order minimum quantity to satisfy supplier constraint"
    # 순소요가 MOQ를 초과하면 결품 해소와 단기 커버 회복을 위해 2주 커버 기준 발주를 적용한다.
    else:
        selected_action = "order_2w_cover"
        reason_1 = "Net requirement exceeds MOQ"
        reason_2 = "Order enough to restore about 2 weeks of cover"

    recommended_qty = _recommended_order_qty(row, selected_action)
    projected_cover_after_order = (
        row["available_inventory"] + row["inbound_qty_4w"] + recommended_qty
    ) / max(row["total_forecast_4w"] / 4, 1)

    warnings = []
    # 리드타임보다 재고 커버가 짧으면 발주 여부와 별개로 납기 리스크를 경고한다.
    if row["inventory_cover_week"] < row["lead_time_week"]:
        warnings.append("cover_below_lead_time")
    # 유통기한이 짧은 SKU가 과도한 커버를 보유하면 폐기 리스크를 경고한다.
    if short_shelf_life and row["inventory_cover_week"] >= excessive_cover_limit:
        warnings.append("short_shelf_life_high_cover")
    # 발주 후 커버가 과도하게 높아질 수 있는 저회전성 SKU는 과재고 검토 대상으로 둔다.
    if projected_cover_after_order > excessive_cover_limit and reduce_review_candidate:
        warnings.append("possible_overstock_or_expiry")
    # 발주 액션인데 MOQ 미만이면 공급사 주문 제약 위반 가능성을 경고한다.
    if selected_action.startswith("order") and recommended_qty < row["moq"]:
        warnings.append("below_moq")

    gate_status = "pass"  # 자동 실행 가능한 기본 gate 상태
    # 납기 타이밍 또는 감량 판단이 필요한 액션은 구매/SCM 담당자 검토 gate로 보낸다.
    if selected_action in {"order_with_timing_check", "reduce_review"}:
        gate_status = "review"
    # 과재고·폐기 가능성이 있으면 수량 자체보다 리스크 검토가 우선이므로 review gate로 전환한다.
    if "possible_overstock_or_expiry" in warnings:
        gate_status = "review"

    return {
        "recommended_order_qty": recommended_qty,  # 최종 추천 발주수량
        "selected_action": selected_action,  # 구매 실행/보류/검토 액션 판단 결과
        "gate_status": gate_status,  # 자동 실행 또는 담당자 검토 필요 여부
        "warning": "; ".join(warnings) if warnings else "none",  # 구매 담당자가 확인해야 할 리스크 사유로그
        "reason_1": reason_1,  # 1차 액션 판단 사유로그
        "reason_2": reason_2,  # 보조 액션 판단 사유로그
    }


def decide_replenishment_action() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] replenishment decision 테이블 생성
    # [현업 의미] 배분 후 남은 SKU 단위 결품과 구매조건을 연결해 실행 가능한 발주 의사결정 테이블을 만든다.
    # [판단 기준] 4주 forecast, 가용재고, 입고예정, 순소요, 재고 커버주수, MOQ, 박스배수, 리드타임
    # [산출물] outputs/05_replenishment_decision.csv
    # [수정 포인트] 실무 적용 시 구매오더 잔량, 생산 가능일, 공급사 휴무, 안전재고, 결재권자 승인 기준을 추가한다.
    # ============================================================
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

    decision_df["inbound_qty_4w"] = decision_df["inbound_qty_4w"].fillna(0)  # 향후 4주 입고예정 수량
    decision_df["lead_time_week"] = decision_df["lead_time_week"].fillna(4)  # 입고 전 결품 가능성을 판단하는 평균 리드타임
    decision_df["net_requirement"] = (
        decision_df["total_forecast_4w"]
        - decision_df["available_inventory"]
        - decision_df["inbound_qty_4w"]
    )  # forecast에서 가용재고와 입고예정을 차감한 추가 발주 필요수량
    decision_df["inventory_cover_week"] = decision_df["available_inventory"] / np.maximum(
        decision_df["total_forecast_4w"] / 4,
        1,
    )  # 현재 가용재고가 forecast 기준 몇 주를 버틸 수 있는지 보는 커버주수

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
