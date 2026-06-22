"""
[FILE PURPOSE]
- 구매·SCM 의사결정에 필요한 기준정보와 과거 판매·재고·입고예정 데이터를 Synthetic 형태로 생성한다.
- 단순 예제 데이터가 아니라, 수요예측 이후 MOQ, 박스배수, 리드타임, 가용재고, 입고예정을 반영할 수 있는 S&OP 입력 구조를 모델링한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차

[INPUT]
- 별도 외부 입력 없이 SKU 유형, 채널 속성, 판매 변동성, 재고 커버 기준을 임시 파라미터로 사용한다.

[OUTPUT]
- data/sales_history.csv: 주차·SKU·채널별 판매실적, 프로모션 여부, 프로모션 uplift
- data/sku_master.csv: SKU별 MOQ, 박스배수, 유통기한, 비용 기준
- data/channel_master.csv: 채널별 서비스 패널티, 전략 중요도, 리드타임, 최소 배분율
- data/inventory_snapshot.csv: 의사결정 주차 기준 재고 스냅샷
- data/inbound_plan.csv: 향후 입고예정 수량

[현업 적용 시 교체 대상]
- POS/출고실적, ERP 품목마스터, SAP MM 구매조건, 공급사 MOQ/박스배수, 유통기한, 입고예정, WMS 재고 스냅샷, 프로모션 캘린더로 교체한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
RANDOM_SEED = 42  # 동일한 포트폴리오 결과를 재현해 모델·배분·발주 비교 기준을 고정하는 난수값


def _build_sku_master() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] SKU 구매·재고 기준 마스터
    # [LOGIC TYPE] Business Logic Feature + Business Grain Design
    # [현업 의미] SKU별 발주 제약과 재고 리스크 기준을 정의해 이후 발주/과재고 판단의 기준값으로 사용한다.
    # [판단 기준] MOQ, 박스배수, 유통기한, 단가, 결품비용, 보관비용, 폐기비용
    # [산출물] sku_master.csv의 SKU별 구매조건 및 비용 기준 컬럼
    # [수정 포인트] ERP 품목마스터, 공급사 계약조건, 원가정보, 폐기/보관 비용 기준으로 교체한다.
    # [WHY] 수요가 같아도 MOQ·박스배수·유통기한·재고비용에 따라 실행 가능한 발주수량과 과재고 위험이 달라지므로 SKU 기준정보가 필요하다.
    # [ASSUMPTION] 모든 row의 구매조건과 비용값은 실제 계약값이 아닌 포트폴리오 시나리오용 synthetic assumption이다.
    # [DESIGN LOGIC] 히어로·신제품·저회전·글로벌·번들·기본품의 서로 다른 발주 및 재고 위험이 후속 액션에서 드러나도록 값을 차등 설정했다.
    # [DATA LINEAGE] data/sku_master.csv에 직접 저장되고 outputs/01_feature_table.csv의 sku_type 및 outputs/05_replenishment_decision.csv의 발주 제약에 간접 반영된다.
    # [REAL DATA REPLACEMENT] ERP 품목마스터, 공급사 계약서, 실제 원가, MOQ·포장단위, 유통기한, 재고보유·폐기 비용으로 교체해야 한다.
    # [INTERVIEW CHECK] 비용값이 회계 검증된 실제 금액이 아니라 SKU별 리스크 차이를 표현하기 위한 가정임을 분명히 설명해야 한다.
    # ============================================================
    rows = [
        ("GlowLab", "GL_HERO_SERUM", "hero", 500, 50, 104, 8.5, 18.0, 0.08, 3.2),  # 핵심 매출 SKU로 결품 손실이 커 MOQ·결품비용을 높게 설정
        ("GlowLab", "GL_NEW_AMPOULE", "new", 300, 30, 78, 7.2, 15.0, 0.07, 2.8),  # 초기 수요 불확실성을 고려해 히어로 SKU보다 작은 발주 단위로 설정
        ("PureLeaf", "PL_SLOW_MASK", "slow", 200, 20, 52, 3.6, 7.5, 0.04, 1.4),  # 저회전·단기 유통기한 특성상 소량 발주와 폐기 위험 관리가 필요한 품목
        ("PureLeaf", "PL_GLOBAL_CREAM", "global", 400, 40, 104, 6.8, 16.0, 0.06, 2.5),  # 수출 성장과 긴 채널 리드타임을 고려해 결품비용을 높게 둔 글로벌 품목
        ("AquaMuse", "AM_BUNDLE_KIT", "bundle", 250, 25, 52, 11.5, 22.0, 0.10, 4.0),  # 행사 매출과 구성품 재고 부담이 커 결품·보관·폐기비용을 가장 높게 설정
        ("AquaMuse", "AM_BASIC_TONER", "basic", 350, 35, 104, 4.4, 9.0, 0.05, 1.6),  # 안정 수요 기본품으로 중간 발주 단위와 상대적으로 낮은 리스크 비용을 설정
    ]
    columns = [
        "brand",  # 수요·재고·구매계획을 구분하는 브랜드 단위
        "finished_good_sku",  # MOQ와 재고 제약을 적용하는 완제품 식별자
        "sku_type",  # 히어로·신제품·저회전 등 수요와 재고 정책을 구분하는 품목 유형
        "moq",  # 공급사 최소 발주수량
        "box_multiple",  # 박스/카톤 단위 발주 배수
        "shelf_life_week",  # 유통기한 기반 폐기 리스크 판단 기준
        "unit_cost",  # 발주금액과 재고자산 부담 산정에 사용할 완제품 단가
        "stockout_cost",  # 결품 발생 시 판매기회손실/서비스 리스크 가중치
        "holding_cost",  # 보관비·자금묶임 부담 가중치
        "disposal_cost",  # 유통기한 경과/폐기 리스크 가중치
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_channel_master() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 판매채널 우선순위 기준 마스터
    # [LOGIC TYPE] Business Logic Feature + Business Grain Design
    # [현업 의미] 제한된 재고를 어느 채널에 먼저 배분할지 판단하기 위한 채널별 서비스·전략 기준을 정의한다.
    # [판단 기준] 서비스 패널티, 전략 중요도, 마진 중요도, 채널 리드타임, 최소 배분율
    # [산출물] channel_master.csv의 채널별 배분 우선순위 기준 컬럼
    # [수정 포인트] 채널별 SLA, 영업 전략, 수익성, 수출/국내 리드타임, 계약상 최소 공급 조건으로 교체한다.
    # [WHY] SKU 재고가 부족할 때 모든 채널을 동일하게 감액하면 서비스 손실·전략 중요도·수익성 차이를 반영할 수 없으므로 채널 기준정보가 필요하다.
    # [ASSUMPTION] 가중치·리드타임·최소 배분율은 실제 계약이나 실측값이 아니라 채널 특성을 구분하기 위한 synthetic assumption이다.
    # [DESIGN LOGIC] 자사몰은 전략·마진, Amazon_US는 결품 패널티, 수출 오프라인은 리드타임과 최소 공급 부담이 드러나도록 상대값을 배치했다.
    # [DATA LINEAGE] data/channel_master.csv에 직접 저장되고 outputs/04_allocation_plan.csv의 priority_score와 outputs/03_sku_shortage_summary.csv의 가중 리드타임에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 채널 SLA, 거래처 페널티, 실현 마진, 영업 우선순위, 주문-납품 리드타임, 계약상 최소 공급률로 교체해야 한다.
    # [INTERVIEW CHECK] 가중치는 절대적 정답이 아니라 정책 파라미터이며 실제 적용 전 영업·SCM 합의와 민감도 검증이 필요하다고 설명해야 한다.
    # ============================================================
    rows = [
        ("D2C", "owned_online", 1.15, 1.25, 1.40, 1, 0.18),  # 자사몰은 브랜드 경험·고객 유지·마진 기여가 커 전략·마진 가중치를 높게 설정
        ("Domestic_Retail", "domestic_wholesale", 1.05, 1.05, 0.95, 2, 0.22),  # 국내 리테일은 안정 공급과 최소 물량 보장이 중요하나 마진 우선순위는 낮게 설정
        ("Amazon_US", "marketplace_global", 1.45, 1.35, 1.20, 4, 0.20),  # 글로벌 플랫폼 결품 시 랭킹·리뷰 손실이 커 서비스·전략 가중치를 높게 설정
        ("Japan_Offline", "export_offline", 1.30, 1.15, 0.90, 5, 0.25),  # 수출 오프라인은 긴 리드타임과 계약 공급 부담을 반영해 최소 배분율을 높게 설정
    ]
    columns = [
        "sales_channel",  # 수요예측과 제한재고 배분을 구분하는 판매채널
        "channel_type",  # 자사몰·도매·글로벌 마켓·수출 오프라인 등 운영 유형
        "service_penalty_weight",  # 결품 시 고객 서비스·거래처 SLA 손실을 수치화한 가중치
        "strategic_weight",  # 회사가 우선 공급해야 하는 채널의 전략 중요도
        "margin_weight",  # 제한재고 배분 시 채널 수익성을 반영하는 마진 가중치
        "lead_time_week",  # 채널 공급까지 필요한 평균 리드타임
        "allocation_min_rate",  # 채널별 최소 공급률 기준
    ]
    return pd.DataFrame(rows, columns=columns)


def _sku_base_demand(sku_type: str) -> float:
    # [LOGIC TYPE] Business Logic Feature
    # [WHY] SKU 유형별 판매 규모 차이를 먼저 정의해야 채널·시즌·행사 효과를 적용한 synthetic 수요가 현실적인 상대 규모를 갖는다.
    # [ASSUMPTION] 유형별 65~520의 주간 기본 수요는 실제 판매실적이 아닌 시나리오 생성용 synthetic assumption이다.
    # [DESIGN LOGIC] 히어로는 가장 높고 저회전품은 가장 낮게 두며, 신제품·글로벌·번들·기본품은 중간 수준으로 차등화했다.
    # [DATA LINEAGE] 반환값은 직접 저장되지 않고 true_demand와 sales_qty를 거쳐 data/sales_history.csv 및 모든 outputs CSV에 간접 반영된다.
    # [REAL DATA REPLACEMENT] SKU별 정상 판매주차의 POS·출고실적, 품절 보정수요, 신제품 유사품 실적 또는 베이스라인 forecast로 교체해야 한다.
    # [INTERVIEW CHECK] 기본 수요가 예측모델의 결과가 아니라 synthetic 데이터 생성의 출발점이며 실제 적용 시 추정 방식 검증이 필요함을 설명해야 한다.
    return {
        "hero": 520,  # 핵심 매출 SKU의 높은 주간 기본 수요
        "new": 230,  # 출시 초기 확산 전제를 둔 신제품 기본 수요
        "slow": 65,  # 저회전 품목의 낮고 간헐적인 기본 수요
        "global": 210,  # 해외 채널 성장 가능성을 반영할 글로벌 품목 기본 수요
        "bundle": 150,  # 행사 의존도가 높은 번들 품목 기본 수요
        "basic": 260,  # 변동성이 낮은 상시 운영 기본품 수요
    }[sku_type]


def _channel_multiplier(sku_type: str, sales_channel: str, week: int) -> float:
    # [LOGIC TYPE] Business Logic Feature
    # [WHY] 동일 SKU도 채널 도달력·고객군·해외 성장·출시 단계에 따라 판매속도가 달라지는 현업 상황을 수요 생성에 반영한다.
    # [ASSUMPTION] 채널 기본 배율과 글로벌 성장률, 번들 상승률, 저회전 감액률, 신제품 ramp-up은 모두 synthetic assumption이다.
    # [DESIGN LOGIC] 채널 기본 규모 위에 특정 SKU 유형과 채널의 상호작용 및 주차 조건을 곱해 단순 평균으로 설명되지 않는 수요 패턴을 만든다.
    # [DATA LINEAGE] 배율은 직접 저장되지 않고 data/sales_history.csv의 true_demand·sales_qty와 outputs/01_feature_table.csv 이후 결과에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 채널별 판매비중, 입점점포 수, 국가별 성장률, 출시 후 주차별 판매곡선, 실제 영업계획으로 교체해야 한다.
    # [INTERVIEW CHECK] 성장률과 배율은 인과 추정값이 아니며 실제 적용 시 채널·SKU별 근거와 기간 안정성을 확인해야 한다.
    base = {
        "D2C": 1.05,  # 자사몰의 상대적으로 높은 직접 판매력을 반영
        "Domestic_Retail": 0.85,  # 국내 도매의 안정적이지만 자사몰보다 낮은 수요 수준
        "Amazon_US": 0.75,  # 글로벌 플랫폼의 초기 채널 규모를 반영
        "Japan_Offline": 0.55,  # 수출 오프라인의 제한된 판매점 커버리지를 반영
    }[sales_channel]

    # 글로벌 전용 품목은 해외 채널에서 주차가 지날수록 유통망 확장에 따른 성장 수요를 반영한다.
    if sku_type == "global" and sales_channel in {"Amazon_US", "Japan_Offline"}:
        base *= 1.0 + 0.012 * week
    # 번들은 행사 구성과 진열 운영이 쉬운 자사몰·국내 리테일에서 수요가 더 높다고 가정한다.
    if sku_type == "bundle" and sales_channel in {"D2C", "Domestic_Retail"}:
        base *= 1.15
    # 저회전 품목은 해외 채널의 긴 보충주기와 제한된 인지도를 고려해 수요를 낮춘다.
    if sku_type == "slow" and sales_channel in {"Amazon_US", "Japan_Offline"}:
        base *= 0.55
    # 신제품은 출시 초기 12주 동안 인지도와 유통 커버리지가 점진적으로 확대되는 ramp-up을 적용한다.
    if sku_type == "new" and week <= 12:
        base *= np.interp(week, [1, 12], [0.45, 1.20])
    # 연말 마지막 구간의 운영일수·출고 컷오프 영향을 단순화해 기본 채널 수요를 소폭 낮춘다.
    if week >= 50:
        base *= 0.95

    return base


def _promo_probability(sku_type: str, sales_channel: str) -> float:
    # [LOGIC TYPE] Business Logic Feature
    sku_effect = {
        "hero": 0.22,  # 핵심 SKU의 정기 캠페인 빈도
        "new": 0.16,  # 출시 지원 행사를 반영한 프로모션 빈도
        "slow": 0.08,  # 저회전 품목의 제한적 행사 빈도
        "global": 0.14,  # 해외 채널 캠페인을 반영한 행사 빈도
        "bundle": 0.30,  # 번들 구성 자체의 높은 행사 의존도
        "basic": 0.12,  # 기본품의 보조적 판촉 빈도
    }[sku_type]
    channel_effect = {
        "D2C": 0.04,  # 자사몰 자체 캠페인으로 추가되는 행사 가능성
        "Domestic_Retail": 0.12,  # 유통사 행사 캘린더의 높은 판촉 빈도
        "Amazon_US": 0.06,  # 글로벌 딜·쿠폰 행사 영향
        "Japan_Offline": 0.02,  # 수출 오프라인의 상대적으로 낮은 행사 유연성
    }[sales_channel]
    return min(sku_effect + channel_effect, 0.55)


def _make_channel_demand(
    rng: np.random.Generator,
    week: int,
    sku_type: str,
    sales_channel: str,
) -> tuple[int, int, float]:
    # ============================================================
    # [BLOCK] SKU·채널별 실제 수요 발생 구조
    # [LOGIC TYPE] Business Logic Feature + Business Grain Design
    # [현업 의미] SKU 생애주기, 채널 특성, 시즌성, 프로모션, 바이럴 수요를 반영해 예측 모델이 학습할 수요 패턴을 만든다.
    # [판단 기준] SKU 유형, 판매채널, 시즌성, 프로모션 여부, 신제품 ramp-up, 글로벌 채널 성장성
    # [산출물] true_demand, promo_flag, promo_uplift
    # [수정 포인트] 실제 POS/출고실적, 프로모션 캘린더, 신제품 출시 계획, 채널별 수요 이벤트로 교체한다.
    # [WHY] 모델이 추세·시즌·행사·바이럴·간헐수요가 섞인 환경을 학습하도록 공급 제약 전 잠재수요를 생성할 필요가 있다.
    # [ASSUMPTION] 시즌 주기, 추세율, 변동성, 행사 확률·uplift, 바이럴 확률·배율, 간헐수요 확률은 실제값이 아닌 synthetic assumption이다.
    # [DESIGN LOGIC] 기준 수요에 각 효과를 곱하되 true_demand와 프로모션 정보를 분리해 후속 결품 절단과 feature 생성을 추적할 수 있게 했다.
    # [DATA LINEAGE] 반환값은 data/sales_history.csv에 직접 반영되고 outputs/01_feature_table.csv, 02_forecast_result.csv, 04_allocation_plan.csv, 05_replenishment_decision.csv에 간접 반영된다.
    # [REAL DATA REPLACEMENT] POS·주문 데이터, 행사 캘린더, 가격·할인율, 출시계획, 검색·리뷰·트래픽, 품절 보정수요로 교체해야 한다.
    # [INTERVIEW CHECK] 각 효과의 곱셈 구조와 가정값이 실제 수요 인과관계를 입증하는 것은 아니며 실제 적용 시 효과 추정이 필요하다고 설명해야 한다.
    # ============================================================
    base = _sku_base_demand(sku_type) * _channel_multiplier(sku_type, sales_channel, week)  # SKU 판매력과 채널 도달력을 결합한 기준 수요
    seasonality = 1.0 + 0.10 * np.sin(2 * np.pi * week / 13) + 0.06 * np.cos(2 * np.pi * week / 26)  # 분기·반기 반복 수요를 반영한 시즌 배율
    trend = 1.0 + (0.003 * week if sku_type in {"hero", "basic"} else 0.0)  # 핵심·기본품의 완만한 상시 성장 추세

    noise_sigma = {
        "hero": 0.24,  # 캠페인과 바이럴이 있으나 기본 판매력이 큰 핵심품 변동성
        "new": 0.55 if week <= 12 else 0.30,  # 출시 초기 높은 불확실성이 안정화되는 신제품 변동성
        "slow": 0.65,  # 판매 발생 자체가 간헐적인 저회전품 변동성
        "global": 0.32,  # 국가·채널 성장 편차를 반영한 글로벌품 변동성
        "bundle": 0.42,  # 행사 일정에 민감한 번들 수요 변동성
        "basic": 0.18,  # 상시 판매 기본품의 낮은 변동성
    }[sku_type]
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma)  # 수요를 음수로 만들지 않으면서 SKU별 판매 변동을 반영

    promo_flag = int(rng.random() < _promo_probability(sku_type, sales_channel))  # 프로모션 여부
    promo_uplift = 0.0  # 프로모션이 없을 때 추가 수요가 없다는 기준값
    # 프로모션 발생 시 기본 판매력보다 높은 수요가 발생하므로 채널별 행사 민감도를 추가 반영한다.
    if promo_flag:
        uplift_base = {
            "hero": 0.45,  # 핵심 SKU 캠페인의 기본 판매 상승률
            "new": 0.28,  # 신제품 인지도 확대 행사 효과
            "slow": 0.18,  # 저회전품의 제한적인 행사 반응
            "global": 0.25,  # 글로벌 채널 판촉의 기본 상승률
            "bundle": 0.65,  # 가격·구성 혜택에 민감한 번들의 높은 행사 효과
            "basic": 0.20,  # 기본품의 완만한 행사 반응
        }[sku_type]
        channel_promo = 0.18 if sales_channel == "Domestic_Retail" else 0.08  # 대형 유통 행사 집중도를 추가 반영한 채널 효과
        promo_uplift = max(0.05, rng.normal(uplift_base + channel_promo, 0.18))  # 행사 시 최소 5% 이상의 수요 상승을 보장

    viral_probability = 0.0  # 일반 SKU·채널은 바이럴 급증이 없다는 기준값
    # 핵심품은 자사몰·글로벌 플랫폼에서 노출 확산 가능성이 있어 바이럴 급증 확률을 부여한다.
    if sku_type == "hero" and sales_channel in {"D2C", "Amazon_US"}:
        viral_probability = 0.035
    # 글로벌 품목은 Amazon 리뷰·랭킹 상승에 따른 돌발 수요 가능성을 반영한다.
    elif sku_type == "global" and sales_channel == "Amazon_US":
        viral_probability = 0.030
    # 신제품은 출시 초기 자사몰 콘텐츠 확산에 따른 돌발 수요 가능성을 반영한다.
    elif sku_type == "new" and sales_channel == "D2C" and week <= 16:
        viral_probability = 0.025

    viral_multiplier = rng.uniform(1.8, 3.6) if rng.random() < viral_probability else 1.0  # 바이럴 발생 시 평시 대비 급증하는 수요 배율

    intermittent_factor = 1.0  # 일반 품목은 매주 연속 수요가 발생한다는 기준값
    # 저회전품은 무판매 주차와 소량 주문 주차가 섞이는 간헐수요를 반영한다.
    if sku_type == "slow":
        intermittent_factor = 0.0 if rng.random() < 0.35 else rng.uniform(0.35, 1.35)

    true_demand = base * seasonality * trend * noise * (1.0 + promo_uplift) * viral_multiplier * intermittent_factor  # 공급 제약 전 고객이 실제로 원한 잠재수요
    true_demand = max(0, int(round(true_demand)))  # 수량 단위 수요로 변환하고 음수 수요를 방지
    return true_demand, promo_flag, round(promo_uplift, 3)


def _apply_stockout_censoring(
    rng: np.random.Generator,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 결품에 따른 관측 판매량 보정
    # [LOGIC TYPE] Business Logic Feature
    # [현업 의미] 실제 수요가 있어도 재고가 부족하면 판매실적에는 낮게 관측되는 결품 왜곡을 모델링한다.
    # [판단 기준] SKU 공통 가용재고, 총수요, 결품 이벤트 발생 여부, 채널별 배분 가중치
    # [산출물] sales_qty가 반영된 판매이력
    # [수정 포인트] 실제 결품 로그, 재고 제약, 주문 컷오프, 채널별 공급 제한 이력으로 교체한다.
    # [WHY] 관측 판매는 순수 수요가 아니라 재고 부족으로 절단될 수 있으므로 true_demand와 실제 판매실적의 차이를 모델링한다.
    # [ASSUMPTION] 실제 결품 로그가 없으므로 주차·SKU별 18% 확률로 부족 이벤트가 발생하고 잠재수요의 58~93%만 공급된다고 가정한다.
    # [DESIGN LOGIC] true_demand는 고객 잠재수요, sales_qty는 SKU 공통 가용량을 채널에 나눈 뒤 관측되는 판매량으로 분리한다.
    # [DATA LINEAGE] true_demand와 sales_qty는 data/sales_history.csv에 직접 저장되고 outputs/01_feature_table.csv부터 forecast·allocation·replenishment에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 결품 로그, 주문 미출고, lost sales, WMS ATP, 주문 컷오프, 채널별 출고 제한 및 재고 이력으로 교체해야 한다.
    # [INTERVIEW CHECK] sales_qty 감소가 수요 하락인지 공급 부족인지 구분하지 않으면 예측이 구조적으로 과소화될 수 있음을 설명해야 한다.
    # ============================================================
    censored_rows = []

    for _, group in demand_df.groupby(["week", "brand", "finished_good_sku"], sort=False):
        total_demand = group["true_demand"].sum()  # 동일 주차·SKU의 전 채널 잠재수요 합계
        shortage_event = rng.random() < 0.18  # 약 18% 주차에서 공급 부족이 발생하는 운영 상황을 가정
        # 결품 이벤트가 발생하면 실제 수요보다 낮은 가용재고만 채널에 배분되어 판매실적이 절단된다.
        if shortage_event and total_demand > 0:
            available_for_week = int(round(total_demand * rng.uniform(0.58, 0.93)))
        else:
            available_for_week = int(round(total_demand * rng.uniform(0.98, 1.18)))

        available_for_week = max(0, available_for_week)
        allocation_weight = group["true_demand"].to_numpy(dtype=float)  # 채널 잠재수요를 기본 배분 비중으로 사용
        allocation_weight = allocation_weight * rng.uniform(0.85, 1.15, size=len(group))
        allocation_weight_sum = allocation_weight.sum()

        if allocation_weight_sum <= 0:
            sales_qty = np.zeros(len(group), dtype=int)
        elif available_for_week >= total_demand:
            sales_qty = group["true_demand"].to_numpy(dtype=int)
        else:
            raw_alloc = available_for_week * allocation_weight / allocation_weight_sum  # 부족한 SKU 재고를 채널별 수요 비중으로 1차 배분
            sales_qty = np.minimum(group["true_demand"].to_numpy(dtype=int), np.floor(raw_alloc).astype(int))
            remaining = available_for_week - sales_qty.sum()  # 정수 절사 후 아직 배분 가능한 잔여 수량
            if remaining > 0:
                gaps = group["true_demand"].to_numpy(dtype=int) - sales_qty  # 잠재수요 대비 추가 공급 여지가 남은 채널별 부족분
                candidate_idx = np.where(gaps > 0)[0]
                if len(candidate_idx) > 0:
                    add_idx = rng.choice(candidate_idx, size=min(remaining, len(candidate_idx)), replace=False)
                    sales_qty[add_idx] += 1

        out = group.copy()
        out["sales_qty"] = sales_qty  # 재고 제약으로 절단되어 POS·출고실적에 실제 관측되는 판매수량
        censored_rows.append(out)

    return pd.concat(censored_rows, ignore_index=True)


def _build_sales_history(
    rng: np.random.Generator,
    sku_master: pd.DataFrame,
    channel_master: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 판매실적 생성
    # [LOGIC TYPE] Business Grain Design + Technical Transformation
    # [현업 의미] 예측 모델이 학습할 주차·SKU·채널 단위의 과거 출고/판매 실적을 구성한다.
    # [판단 기준] SKU 유형, 판매채널, 프로모션 여부, 결품으로 절단된 관측 판매량
    # [산출물] sales_history.csv의 sales_qty, promo_flag, promo_uplift
    # [수정 포인트] POS, 출고실적, 온라인 주문, B2B 납품실적, 프로모션 캘린더로 교체한다.
    # [WHY] SKU·채널·주차별 공통 grain의 판매이력이 있어야 Step 2에서 누수 없이 lag·rolling feature와 4주 actual을 생성할 수 있다.
    # [ASSUMPTION] 6개 SKU와 4개 채널이 60주 동안 모두 운영되며 누락 주차나 단종·입점 변경이 없다고 가정한다.
    # [DESIGN LOGIC] 잠재수요와 프로모션을 먼저 생성한 뒤 SKU 공통 재고 제약으로 sales_qty를 절단해 수요와 관측 판매를 구분한다.
    # [DATA LINEAGE] data/sales_history.csv에 직접 저장되며 outputs/01_feature_table.csv와 이후 예측·배분·발주 결과의 원천이 된다.
    # [REAL DATA REPLACEMENT] POS, ERP 출고, 주문·반품, 채널 입점기간, 품절 보정수요, 프로모션 캘린더를 같은 grain으로 정합화해야 한다.
    # [INTERVIEW CHECK] 실제 데이터에서는 미판매 주차와 데이터 누락을 구분하고 반품·취소·채널 개폐점을 정리해야 함을 설명해야 한다.
    # ============================================================
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
                        "week": week,  # 판매·프로모션·재고 제약이 발생한 운영 주차
                        "brand": sku.brand,  # 판매실적을 구분하는 브랜드
                        "finished_good_sku": sku.finished_good_sku,  # 재고를 공동 사용하는 완제품 SKU
                        "sales_channel": channel,  # 수요 발생과 재고 배분을 구분하는 판매채널
                        "true_demand": true_demand,  # 결품 제약 전 고객 잠재수요
                        "promo_flag": promo_flag,  # 해당 주차·SKU·채널 프로모션 실행 여부
                        "promo_uplift": promo_uplift,  # 프로모션이 기본 수요에 추가한 상승률
                    }
                )

    demand_df = pd.DataFrame(rows)
    sales_df = _apply_stockout_censoring(rng, demand_df)
    return sales_df[
        [
            "week",  # 판매실적 발생 주차
            "brand",  # 브랜드별 수요계획 단위
            "finished_good_sku",  # 완제품별 예측·재고 판단 단위
            "sales_channel",  # 채널별 수요 패턴 구분 단위
            "sales_qty",  # 결품 영향을 포함해 실제 관측된 판매수량
            "promo_flag",  # 프로모션 실행 여부
            "promo_uplift",  # 행사로 발생한 추가 수요 상승률
        ]
    ].sort_values(["week", "brand", "finished_good_sku", "sales_channel"])


def _build_inventory_snapshot(
    rng: np.random.Generator,
    sales_history: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 의사결정 주차 재고 스냅샷
    # [LOGIC TYPE] Business Logic Feature + Business Grain Design
    # [현업 의미] 예측수요를 실제로 공급 가능한지 판단하기 위해 현재고, 보류재고, 가용재고를 구성한다.
    # [판단 기준] 최근 판매속도, SKU 유형별 목표 커버 수준, 보류재고 비율
    # [산출물] inventory_snapshot.csv의 on_hand_inventory, blocked_inventory, available_inventory
    # [수정 포인트] WMS/ERP 재고 스냅샷, 품질보류, 예약재고, 출고불가재고 기준으로 교체한다.
    # [WHY] forecast를 실제 공급 가능량과 비교하려면 장부재고에서 품질보류·예약 등 출고 불가분을 제외한 가용재고가 필요하다.
    # [ASSUMPTION] 최근 8주 판매를 4주 수준으로 환산하고 SKU 유형별 임의 커버 배율과 1~8% 보류율을 적용한 synthetic snapshot이다.
    # [DESIGN LOGIC] 신제품·글로벌은 부족, 저회전·기본품은 과재고가 나타나도록 커버 배율을 달리해 다양한 후속 action을 생성한다.
    # [DATA LINEAGE] data/inventory_snapshot.csv에 직접 저장되고 outputs/03_sku_shortage_summary.csv, 04_allocation_plan.csv, 05_replenishment_decision.csv에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 의사결정 시점의 WMS 로케이션 재고, 품질보류, 예약·할당재고, 미출고 오더, 재고 상태코드로 교체해야 한다.
    # [INTERVIEW CHECK] on_hand와 available의 차이 및 snapshot 기준시각·재고 상태 정의가 회사별로 다르므로 실제 적용 시 확인 필요함을 설명해야 한다.
    # ============================================================
    recent_sales = (
        sales_history[sales_history["week"].between(53, 60)]
        .groupby(["brand", "finished_good_sku"], as_index=False)["sales_qty"]
        .sum()
    )
    recent_sales["avg_4w_sales"] = recent_sales["sales_qty"] / 2.0  # 최근 8주 합계를 4주 환산해 재고 커버 산정 기준으로 사용
    recent_sales = recent_sales.merge(
        sku_master[["brand", "finished_good_sku", "sku_type"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    rows = []
    for row in recent_sales.itertuples(index=False):
        coverage_factor = {
            "hero": rng.uniform(1.25, 1.70),  # 핵심 SKU는 결품 방지를 위해 4주 판매의 1.25~1.70배 재고를 보유
            "new": rng.uniform(0.25, 0.45),  # 신제품은 초기 공급 부족 가능성이 있는 낮은 재고 수준을 가정
            "slow": rng.uniform(2.20, 3.00),  # 저회전품은 판매 대비 장기재고가 누적된 상황을 가정
            "global": rng.uniform(0.55, 0.75),  # 해외 성장 대비 현재고가 부족한 공급 상황을 가정
            "bundle": rng.uniform(0.82, 0.95),  # 행사 수요를 완전히 커버하지 못하는 번들 재고 수준을 가정
            "basic": rng.uniform(2.00, 2.70),  # 안정 수요 대비 과도한 기본품 재고 보유 상황을 가정
        }[row.sku_type]
        on_hand = int(round(row.avg_4w_sales * coverage_factor))  # 최근 판매속도와 SKU 정책을 반영한 장부상 현재고
        blocked = int(round(on_hand * rng.uniform(0.01, 0.08)))  # 품질검사·예약·출고보류로 즉시 사용할 수 없는 재고
        available = max(0, on_hand - blocked)  # 실제 배분 판단에 사용할 가용재고
        rows.append(
            {
                "decision_week": 61,  # 과거 60주 이후 재고·발주 판단을 수행하는 기준 주차
                "brand": row.brand,  # 브랜드별 재고계획 구분 단위
                "finished_good_sku": row.finished_good_sku,  # 채널이 공동으로 사용하는 완제품 재고 단위
                "on_hand_inventory": on_hand,  # WMS·ERP 장부상 총 현재고
                "blocked_inventory": blocked,  # 품질·예약 사유로 배분할 수 없는 보류재고
                "available_inventory": available,  # 총 현재고에서 보류재고를 제외한 실제 배분 가능량
            }
        )

    return pd.DataFrame(rows)


def _build_inbound_plan(
    rng: np.random.Generator,
    sales_history: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 입고예정 계획 생성
    # [LOGIC TYPE] Business Logic Feature + Business Grain Design
    # [현업 의미] 신규 발주 필요량을 계산하기 전에 이미 발주되어 들어올 물량을 반영한다.
    # [판단 기준] 최근 판매속도, SKU 유형별 입고 예정 비율, 박스배수
    # [산출물] inbound_plan.csv의 61주차 현재 기준 향후 1주/2주/4주 누적 입고예정 수량
    # [수정 포인트] SAP MM 구매오더, 공급사 ASN, 생산계획, 선적/통관 일정으로 교체한다.
    # [WHY] 이미 발주·생산·선적된 물량을 무시하면 순소요와 추천 발주량이 중복 계산되므로 기간별 입고예정을 분리해야 한다.
    # [ASSUMPTION] 최근 12주 판매의 4주 환산값에 SKU 유형별 임의 입고계수를 적용하고 모든 입고가 박스배수를 따른다고 가정한다.
    # [DESIGN LOGIC] decision_week=61은 입고실적 주차가 아니라 계획 cut-off이며, inbound_qty_4w는 62~65주차 forecast horizon과 같은 향후 4주 누계다.
    # [DATA LINEAGE] data/inbound_plan.csv에 직접 저장되고 inbound_qty_4w가 outputs/05_replenishment_decision.csv의 net_requirement에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 오픈 PO, 생산오더, ASN, 확정납기, 선적·통관 상태, 공급사 납기준수율과 취소·지연 정보로 교체해야 한다.
    # [INTERVIEW CHECK] 예정수량을 확정 입고로 간주한 단순화가 있으며 실제 적용 시 ETA 신뢰도와 지연 확률 반영이 필요함을 설명해야 한다.
    # ============================================================
    recent_sales = (
        sales_history[sales_history["week"].between(49, 60)]
        .groupby(["brand", "finished_good_sku"], as_index=False)["sales_qty"]
        .sum()
    )
    recent_sales["avg_4w_sales"] = recent_sales["sales_qty"] / 3.0  # 최근 12주 판매를 4주 수요 수준으로 환산한 입고계획 기준
    plan = recent_sales.merge(
        sku_master[["brand", "finished_good_sku", "box_multiple"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    rows = []
    for row in plan.itertuples(index=False):
        base = row.avg_4w_sales  # SKU별 입고예정 규모를 산정하는 최근 4주 환산 판매량
        sku_type = sku_master.loc[
            sku_master["finished_good_sku"].eq(row.finished_good_sku),
            "sku_type",
        ].iloc[0]
        inbound_factor = {
            "hero": rng.uniform(0.20, 0.45),  # 핵심품은 일정 수준의 구매오더 잔량이 있는 상황을 가정
            "new": rng.uniform(0.00, 0.12),  # 신제품은 공급 안정화 전이라 확정 입고가 적은 상황을 가정
            "slow": rng.uniform(0.00, 0.05),  # 저회전품은 과재고 방지를 위해 추가 입고를 최소화
            "global": rng.uniform(0.55, 0.85),  # 해외 성장 대응을 위한 비교적 큰 입고예정을 가정
            "bundle": rng.uniform(0.08, 0.18),  # 번들은 행사 전 제한적인 보충 물량만 확정된 상황을 가정
            "basic": rng.uniform(0.00, 0.08),  # 기본품 과재고 상황을 고려해 추가 입고를 낮게 설정
        }[sku_type]
        inbound_1w = _round_to_multiple(base * inbound_factor * rng.uniform(0.05, 0.25), row.box_multiple)  # 1주 내 입고예정
        inbound_2w = _round_to_multiple(base * inbound_factor * rng.uniform(0.25, 0.55), row.box_multiple)  # 2주 내 입고예정
        inbound_4w = _round_to_multiple(base * inbound_factor, row.box_multiple)  # 발주 판단에 반영할 4주 내 입고예정
        rows.append(
            {
                "decision_week": 61,  # 입고실적 주차가 아니라 62~65주차 누적 입고예정을 조회하는 계획 기준시점
                "brand": row.brand,  # 브랜드별 공급계획 구분 단위
                "finished_good_sku": row.finished_good_sku,  # 입고예정과 발주 필요량을 연결하는 완제품 단위
                "inbound_qty_1w": inbound_1w,  # 61주차 현재 기준 향후 1주 이내 입고예정 누계
                "inbound_qty_2w": inbound_2w,  # 61주차 현재 기준 향후 2주 이내 입고예정 누계
                "inbound_qty_4w": inbound_4w,  # 61주차 현재 기준 62~65주차 내 입고예정 누계이며 Step05 순소요에서 차감
            }
        )

    return pd.DataFrame(rows)


def _round_to_multiple(value: float, multiple: int) -> int:
    # [LOGIC TYPE] Technical Transformation
    # 박스배수보다 작은 단위로 발주·입고가 불가한 공급 조건을 반영한다.
    if value <= 0:
        return 0
    return int(round(value / multiple) * multiple)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    # [LOGIC TYPE] Technical Transformation
    df.to_csv(path, index=False, encoding="utf-8-sig")


def generate_all_input_data() -> None:
    # ============================================================
    # [BLOCK] S&OP 입력 데이터 생성 및 저장
    # [LOGIC TYPE] Business Grain Design + Technical Transformation
    # [현업 의미] 예측, 배분, 발주 판단에 필요한 기준정보와 운영 데이터를 하나의 입력 세트로 준비한다.
    # [판단 기준] SKU 구매조건, 채널 우선순위, 판매실적, 가용재고, 입고예정
    # [산출물] data/ 하위 5개 CSV 입력 테이블
    # [수정 포인트] 실무에서는 각 테이블을 ERP/WMS/POS/프로모션 시스템에서 추출한 데이터로 대체한다.
    # [WHY] 예측·배분·발주 단계가 같은 키와 의사결정 시점을 공유하도록 synthetic 기준정보와 운영 데이터를 한 번에 준비한다.
    # [ASSUMPTION] RANDOM_SEED로 재현 가능한 단일 시나리오를 만들며 CSV 간 키 누락과 적재 지연이 없다고 가정한다.
    # [DESIGN LOGIC] master → sales history → inventory snapshot → inbound plan 순으로 생성해 뒤 데이터가 앞 기준정보를 참조하도록 했다.
    # [DATA LINEAGE] data/sku_master.csv, channel_master.csv, sales_history.csv, inventory_snapshot.csv, inbound_plan.csv를 직접 생성하고 전체 outputs/*.csv에 간접 반영한다.
    # [REAL DATA REPLACEMENT] 데이터마트의 품목·채널 master, POS/출고, WMS snapshot, SAP MM PO/ASN을 기준일과 키 체계에 맞춰 연결해야 한다.
    # [INTERVIEW CHECK] synthetic CSV끼리 정합성이 보장된 환경과 달리 실제 적용에서는 키 매핑·기준시각·결측·중복 품질검사가 필요함을 설명해야 한다.
    # ============================================================
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)  # 모든 Synthetic 운영 시나리오를 동일하게 재현하는 난수 생성기

    sku_master = _build_sku_master()  # SKU별 MOQ·박스배수·유통기한·비용 기준
    channel_master = _build_channel_master()  # 채널별 서비스·전략·마진·리드타임·최소 배분 기준
    sales_history = _build_sales_history(rng, sku_master, channel_master)  # 결품 제약이 반영된 주차별 관측 판매실적
    inventory_snapshot = _build_inventory_snapshot(rng, sales_history, sku_master)  # 의사결정 주차의 현재고·보류재고·가용재고
    inbound_plan = _build_inbound_plan(rng, sales_history, sku_master)  # 신규 발주 전에 고려할 1·2·4주 입고예정

    # [DATA LINEAGE] 주차·SKU·채널별 잠재수요·관측판매·프로모션 정보를 data/sales_history.csv에 직접 저장한다.
    _save_csv(sales_history, DATA_DIR / "sales_history.csv")
    _save_csv(sku_master, DATA_DIR / "sku_master.csv")
    _save_csv(channel_master, DATA_DIR / "channel_master.csv")
    # [DATA LINEAGE] 의사결정 주차의 장부·보류·가용재고를 data/inventory_snapshot.csv에 직접 저장한다.
    _save_csv(inventory_snapshot, DATA_DIR / "inventory_snapshot.csv")
    # [DATA LINEAGE] 1·2·4주 누적 입고예정을 data/inbound_plan.csv에 직접 저장한다.
    _save_csv(inbound_plan, DATA_DIR / "inbound_plan.csv")


if __name__ == "__main__":
    generate_all_input_data()
