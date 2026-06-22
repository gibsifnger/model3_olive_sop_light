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
    # [WHY] 수요예측이 독립 분석으로 끝나지 않고 재고 제약·채널 배분·구매 액션까지 같은 기준정보로 이어져야 S&OP 의사결정으로 사용할 수 있다.
    # [ASSUMPTION] 별도 스케줄러와 승인 시스템이 없으므로 모든 단계를 한 프로세스에서 순차 실행하고 앞 단계 CSV가 정상 생성된다고 가정한다.
    # [DESIGN LOGIC] 입력 생성 → feature → forecast → allocation → replenishment → KPI 순서를 고정해 각 단계가 직전 산출물을 소비하도록 설계했다.
    # [DATA LINEAGE] data/*.csv를 생성한 뒤 outputs/01_feature_table.csv부터 outputs/06_summary_table.csv까지 순차적으로 직접 생성한다.
    # [REAL DATA REPLACEMENT] 실무에서는 ERP·WMS·POS 적재 완료 여부, 배치 스케줄, 데이터 품질 gate, 담당자 승인 상태와 연결해야 한다.
    # [INTERVIEW CHECK] 예측 정확도만 제시하지 않고 예측 결과가 어떤 순서로 배분과 발주 판단에 연결되는지 설명해야 한다.
    # ============================================================
    generate_all_input_data()  # 판매·품목·채널·재고·입고예정 기준정보를 동일한 의사결정 시점으로 준비한다.
    build_feature_table()  # 과거 판매를 4주 수요예측에 필요한 추세·변동성·행사 신호로 변환한다.
    train_and_select_forecast_model()  # 검증 WAPE가 가장 낮은 모델로 SKU·채널별 4주 수요를 확정한다.
    build_allocation_plan()  # 확정 예측수요를 SKU 공통 가용재고와 비교해 채널 우선순위대로 배분한다.
    decide_replenishment_action()  # 배분 후 부족분과 입고예정·MOQ·리드타임을 결합해 구매 액션을 결정한다.
    build_summary_table()  # 예측 성과와 결품·배분·발주 결과를 S&OP 핵심지표로 요약한다.


if __name__ == "__main__":
    main()
