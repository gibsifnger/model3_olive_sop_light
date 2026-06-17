"""
[FILE PURPOSE]
- 전체 S&OP 의사결정 파이프라인을 정해진 순서로 실행한다.
- 수요 데이터 생성부터 예측, 재고 제약 기반 배분, 발주 액션, 요약 지표까지 한 번에 연결한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차

[INPUT]
- data/ 및 outputs/에 저장되는 단계별 CSV 산출물

[OUTPUT]
- outputs/01_feature_table.csv부터 outputs/06_summary_table.csv까지의 의사결정 산출물

[현업 적용 시 교체 대상]
- Synthetic 입력 생성 단계는 POS/출고실적, ERP 품목마스터, 재고 스냅샷, 입고예정, 공급사 MOQ/박스배수 기준으로 교체한다.
"""

from src.step01_generate_synthetic_data import generate_all_input_data
from src.step02_build_features import build_feature_table
from src.step03_train_forecast_model import train_and_select_forecast_model
from src.step04_build_allocation_plan import build_allocation_plan
from src.step05_decide_replenishment_action import decide_replenishment_action
from src.utils import build_summary_table


def main() -> None:
    # ============================================================
    # [BLOCK] S&OP 의사결정 파이프라인 실행 순서
    # [현업 의미] 예측 수요를 재고 제약과 발주 액션으로 연결하는 월간/주간 S&OP 판단 흐름을 고정한다.
    # [판단 기준] 수요예측, 가용재고, 채널 우선순위, 결품수량, MOQ, 입고예정, 발주 액션 기준
    # [산출물] feature, forecast, allocation, replenishment, summary CSV
    # [수정 포인트] 실무 적용 시 회사의 배치 스케줄, 데이터 적재 순서, 승인 프로세스에 맞춰 실행 단위를 조정한다.
    # ============================================================
    generate_all_input_data()
    build_feature_table()
    train_and_select_forecast_model()
    build_allocation_plan()
    decide_replenishment_action()
    build_summary_table()


if __name__ == "__main__":
    main()
