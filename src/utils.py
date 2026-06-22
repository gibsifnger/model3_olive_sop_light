"""
[FILE PURPOSE]
- 예측, 배분, 발주 액션 결과를 포트폴리오 설명용 KPI 요약 테이블로 집계한다.
- 모델 성과와 SCM 실행 결과를 한 장의 summary로 연결해 S&OP 의사결정 흐름을 보여준다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차 결과를 전체 포트폴리오 수준으로 요약

[INPUT]
- outputs/02_forecast_result.csv: 예측 결과
- outputs/04_allocation_plan.csv: 채널별 배분 결과
- outputs/05_replenishment_decision.csv: SKU별 발주 액션
- outputs/model_selection_summary.csv: 모델 선택 및 성과 지표

[OUTPUT]
- outputs/06_summary_table.csv: 예측 성과, 결품 SKU 수, fill rate, 미충족수량, 발주 액션 수 요약

[현업 적용 시 교체 대상]
- 회사 표준 S&OP KPI, 서비스레벨, 결품 금액, 주문 실행률, 발주 승인 상태, 재고금액 지표로 확장한다.
"""

from pathlib import Path

import pandas as pd


OUTPUT_DIR = Path("outputs")


def build_summary_table() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] S&OP 의사결정 KPI 요약
    # [현업 의미] 예측모델 성과와 배분·발주 실행 결과를 하나의 경영/면접 설명용 테이블로 요약한다.
    # [판단 기준] 선택 모델, calibration factor, WAPE, forecast accuracy, 결품 SKU 수, fill rate, 미충족수량, 발주/검토 액션 수
    # [산출물] outputs/06_summary_table.csv
    # [수정 포인트] 실무 적용 시 매출 영향, 결품 금액, 재고금액, 폐기예상금액, 채널별 서비스레벨 KPI를 추가한다.
    # [WHY] 예측모델 성과와 공급·구매 실행 결과를 한 화면에서 연결해야 모델 개선이 실제 서비스 수준과 action에 미친 영향을 설명할 수 있다.
    # [ASSUMPTION] 전체 SKU·채널 단순 평균 및 건수 집계가 포트폴리오 수준 요약에 충분하다고 가정한다.
    # [DESIGN LOGIC] 선택 모델·정확도·bias와 shortage·fill rate·미충족·발주·검토 action을 단일 row KPI로 구성한다.
    # [DATA LINEAGE] outputs/02_forecast_result.csv, 04_allocation_plan.csv, 05_replenishment_decision.csv, model_selection_summary.csv를 읽어 06_summary_table.csv를 직접 생성한다.
    # [REAL DATA REPLACEMENT] 매출·마진·재고금액·서비스 수준·폐기·PO 실행률과 조직별 KPI 정의로 확장해야 한다.
    # [INTERVIEW CHECK] 예측 정확도 개선이 자동으로 결품 감소를 보장하지 않으므로 공급 제약 KPI와 함께 봐야 한다고 설명해야 한다.
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    forecast_result = pd.read_csv(OUTPUT_DIR / "02_forecast_result.csv")
    allocation_plan = pd.read_csv(OUTPUT_DIR / "04_allocation_plan.csv")
    replenishment_decision = pd.read_csv(OUTPUT_DIR / "05_replenishment_decision.csv")
    model_summary = pd.read_csv(OUTPUT_DIR / "model_selection_summary.csv")

    selected_rows = model_summary[model_summary["selected_final_model"] == True]  # 검증 WAPE 기준으로 최종 채택된 모델 행
    selected_model = selected_rows["model_name"].iloc[0]  # 배분·발주 forecast를 생성한 최종 모델명

    validation_row = model_summary[
        (model_summary["split"] == "validation") & (model_summary["model_name"] == selected_model)
    ].iloc[0]
    test_row = model_summary[
        (model_summary["split"] == "test") & (model_summary["model_name"] == selected_model)
    ].iloc[0]

    sku_level = allocation_plan[
        ["decision_week", "brand", "finished_good_sku", "shortage_qty"]  # 채널 중복 없이 SKU 결품 건수를 세기 위한 기준 컬럼
    ].drop_duplicates()
    order_actions = replenishment_decision[
        replenishment_decision["selected_action"].isin(
            ["order_moq", "order_2w_cover", "order_with_timing_check"]
        )  # 실제 발주 실행 또는 납기 타이밍 점검이 필요한 액션만 집계한다.
    ]

    summary_table = pd.DataFrame(
        [
            {
                "selected_model": selected_model,  # 검증 성과 기준으로 최종 채택한 수요예측 모델
                "calibration_factor": test_row["calibration_factor"],  # 검증 총수요 편향을 조정한 forecast 보정 배율
                "validation_wape": validation_row["wape"],  # 모델 선택 구간의 총수요 대비 절대오차율
                "test_wape": test_row["wape"],  # 최근 테스트 구간의 총수요 대비 절대오차율
                "test_forecast_accuracy": test_row["forecast_accuracy"],  # 테스트 WAPE를 정확도 관점으로 변환한 값
                "test_bias_pct": test_row["bias_pct"],  # 과대·과소예측으로 재고 위험 방향을 보는 편향률
                "test_hit_rate_20": test_row["hit_rate_20"],  # 오차 20% 이내인 SKU·채널 예측 비중
                "test_mae": test_row["mae"],  # SKU·채널당 평균 예측 수량 오차
                "total_sku_count": sku_level[["brand", "finished_good_sku"]].drop_duplicates().shape[0],  # 공급계획 대상 SKU 수
                "total_sku_channel_count": allocation_plan[
                    ["decision_week", "brand", "finished_good_sku", "sales_channel"]
                ].drop_duplicates().shape[0],
                "shortage_sku_count": sku_level[sku_level["shortage_qty"] > 0].shape[0],  # 4주 예측수요를 현재고로 충족하지 못한 SKU 수
                "avg_allocation_fill_rate": allocation_plan["allocation_fill_rate"].mean(),  # 채널별 평균 수요 충족률
                "total_unfulfilled_qty": allocation_plan["unfulfilled_qty"].sum(),  # 제한재고로 공급하지 못한 총수량
                "order_action_count": len(order_actions),  # 신규 발주 또는 납기 점검이 필요한 SKU 수
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
