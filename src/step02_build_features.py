"""
[FILE PURPOSE]
- 과거 판매실적과 SKU 기준정보를 결합해 4주 수요예측에 사용할 feature table을 생성한다.
- 판매 추세, 변동성, 프로모션, 시즌성을 수요계획자가 보는 판단 신호로 구조화한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차

[INPUT]
- data/sales_history.csv: 주차별 판매실적, 프로모션 여부, 프로모션 uplift
- data/sku_master.csv: SKU 유형과 품목 기준정보

[OUTPUT]
- outputs/01_feature_table.csv: 예측 모델 학습용 feature와 향후 4주 실제수요(actual_4w)

[현업 적용 시 교체 대상]
- POS/출고실적, 프로모션 캘린더, 상품 마스터, 채널 마스터, 시즌/캠페인 캘린더를 실제 운영 데이터로 교체한다.
"""

from pathlib import Path

import numpy as np
import pandas as pd


DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")


FEATURE_COLUMNS = [
    "lag_1",  # 직전 1주 판매속도를 반영해 가장 최근 수요 수준을 포착하는 신호
    "lag_4",  # 4주 전 판매량과 비교해 월간 반복 패턴과 단기 변화를 구분하는 신호
    "roll_4",  # 최근 4주 평균 판매로 단기 발주·배분 기준이 되는 수요 수준
    "roll_8",  # 최근 8주 평균 판매로 일시적 행사 영향을 완화한 중기 수요 기준선
    "sales_std_4",  # 최근 4주 변동성으로 안전재고와 예측 불확실성 해석에 쓰는 신호
    "recent_trend",  # 단기 평균과 중기 평균의 차이로 최근 수요 상승·하락 방향을 표현
    "growth_ratio",  # 중기 평균 대비 단기 판매 배율로 성장 또는 둔화 강도를 표현
    "lag1_vs_lag4",  # 직전 판매와 4주 전 판매의 비율로 최근 급증·급락 여부를 표현
    "promo_flag",  # 해당 주차의 프로모션 여부를 일반 판매와 행사 수요 구분에 반영
    "promo_uplift",  # 프로모션이 기본 수요보다 추가로 만든 판매 상승폭
    "week_sin",  # 연간 주차의 순환성을 연말·연초 단절 없이 표현하는 시즌 신호
    "week_cos",  # week_sin과 함께 연중 성수기·비수기 위치를 구분하는 시즌 신호
    "brand_code",  # 브랜드별 고객층과 판매력 차이를 모델이 구분하도록 만든 식별 신호
    "sku_type_code",  # 히어로·신제품·저회전 등 SKU 생애주기별 수요 특성 신호
    "channel_code",  # 자사몰·리테일·글로벌 등 채널별 판매 패턴을 구분하는 신호
]


KEY_COLUMNS = ["brand", "finished_good_sku", "sales_channel"]  # SKU·채널별 수요 패턴을 분리해 학습하기 위한 판단 단위


def _add_history_features(group: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 과거 판매 기반 수요 신호 생성
    # [현업 의미] 최근 판매속도, 중기 평균, 변동성, 향후 4주 실적을 만들어 예측 모델의 수요 판단 기준으로 사용한다.
    # [판단 기준] 1주 전 판매, 4주 전 판매, 최근 4주/8주 평균, 최근 판매 변동성, 4주 예측기간
    # [산출물] lag, rolling, sales_std_4, actual_4w 컬럼
    # [수정 포인트] 실무 적용 시 품절 보정 판매량, 반품 차감, 비정상 행사 물량 제외 기준을 반영한다.
    # [WHY] 미래 정보를 사용하지 않고도 최근 판매 수준·추세·변동성을 수치화해야 4주 수요예측과 재고 판단에 연결할 수 있다.
    # [ASSUMPTION] sales_qty가 학습 가능한 관측수요이며 1·4·8주 창이 모든 SKU·채널에 공통으로 적합하다고 가정한다.
    # [DESIGN LOGIC] 예측시점 이전 값만 shift해 lag·rolling을 만들고 이후 4주 합계를 actual_4w로 두어 정보 누수를 방지한다.
    # [DATA LINEAGE] 생성 feature는 outputs/01_feature_table.csv에 직접 저장되고 outputs/02_forecast_result.csv 이후 결과에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 품절 보정 판매, 반품·취소, 영업일, SKU별 적정 window와 실제 forecast horizon 정책을 반영해야 한다.
    # [INTERVIEW CHECK] 왜 4주 horizon과 4·8주 window를 사용했는지, 실제 적용 시 backtest로 재선정해야 한다는 점을 설명해야 한다.
    # ============================================================
    group = group.sort_values("week").copy()
    sales = group["sales_qty"]

    group["lag_1"] = sales.shift(1)  # 예측시점에 확인 가능한 직전 1주 판매실적
    group["lag_4"] = sales.shift(4)  # 예측시점 기준 4주 전의 비교 판매실적
    group["roll_4"] = sales.shift(1).rolling(window=4, min_periods=1).mean()  # 정보 누수를 제외한 최근 4주 평균 판매속도
    group["roll_8"] = sales.shift(1).rolling(window=8, min_periods=1).mean()  # 일시 변동을 완화한 최근 8주 수요 기준선
    group["sales_std_4"] = sales.shift(1).rolling(window=4, min_periods=2).std()  # 단기 수요 불확실성과 변동성 수준
    group["actual_4w"] = sales.shift(-1).rolling(window=4, min_periods=4).sum().shift(-3)  # 예측 성과를 평가할 향후 4주 수요

    return group


def _add_codes(feature_df: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 범주형 운영 기준의 모델 입력화
    # [현업 의미] 브랜드, SKU 유형, 판매채널별 수요 특성을 예측 모델이 구분할 수 있도록 기준정보를 숫자 신호로 변환한다.
    # [판단 기준] 브랜드, SKU 유형, 판매채널
    # [산출물] brand_code, sku_type_code, channel_code
    # [수정 포인트] 실무 적용 시 품목군, 가격대, 거래처군, 국가, 물류권역 등 추가 기준정보를 확장한다.
    # [WHY] 모델이 브랜드·SKU 유형·채널별로 다른 판매 패턴을 구분하려면 범주형 운영 기준을 입력 feature로 제공해야 한다.
    # [ASSUMPTION] 현재 데이터에 존재하는 category code가 학습·검증·테스트에서 일관되고 신규 범주가 없다고 가정한다.
    # [DESIGN LOGIC] 간단한 category code로 기준정보 차이를 표현했으며 숫자 크기 자체에 업무상 순위 의미는 부여하지 않는다.
    # [DATA LINEAGE] code 컬럼은 outputs/01_feature_table.csv에 직접 저장되고 선택 모델의 forecast에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 고정 mapping 또는 학습 pipeline encoder, 미등록 범주 처리, 품목군·국가·거래처군 master가 필요하다.
    # [INTERVIEW CHECK] 단순 code가 순서형 의미로 오해될 수 있으므로 모델 특성과 운영 배포 시 mapping 고정 필요성을 설명해야 한다.
    # ============================================================
    feature_df = feature_df.copy()
    feature_df["brand_code"] = pd.Categorical(feature_df["brand"]).codes  # 브랜드별 수요 특성을 구분하는 모델 입력값
    feature_df["sku_type_code"] = pd.Categorical(feature_df["sku_type"]).codes  # SKU 생애주기·회전 특성을 구분하는 모델 입력값
    feature_df["channel_code"] = pd.Categorical(feature_df["sales_channel"]).codes  # 판매채널별 주문 패턴을 구분하는 모델 입력값
    return feature_df


def build_feature_table() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 예측용 feature table 구축
    # [현업 의미] S&OP 수요회의에서 보는 과거 판매 흐름과 프로모션 정보를 모델 학습용 테이블로 정리한다.
    # [판단 기준] SKU·채널별 판매 추세, 성장률, 프로모션 여부, 시즌성, 향후 4주 예측기간
    # [산출물] outputs/01_feature_table.csv
    # [수정 포인트] 실무 데이터 적용 시 채널별 결품 보정, 행사 제외/포함 정책, 신제품 초기 수요 보정 로직을 조정한다.
    # [WHY] 원천 판매이력과 SKU master를 모델이 소비할 수 있는 단일 grain의 학습 테이블로 정합화해야 재현 가능한 forecast가 가능하다.
    # [ASSUMPTION] brand·finished_good_sku·sales_channel 키가 정확히 매칭되고 52주 연간 주기와 4주 S&OP horizon이 적절하다고 가정한다.
    # [DESIGN LOGIC] 이력 feature, SKU 유형, 프로모션, 연간 시즌성, actual_4w를 한 행에 결합해 시점별 모델 입력과 평가 대상을 함께 보존한다.
    # [DATA LINEAGE] data/sales_history.csv와 data/sku_master.csv를 읽어 outputs/01_feature_table.csv를 직접 생성하며 Step 3의 유일한 모델 입력이 된다.
    # [REAL DATA REPLACEMENT] POS·출고·프로모션·품목 master의 키 정합, 품절 보정, 캘린더, 신제품 cold-start feature가 필요하다.
    # [INTERVIEW CHECK] decision_week 기준으로 feature는 과거만, actual_4w는 미래 평가값만 사용해 누수를 차단했는지 설명해야 한다.
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    sales_history = pd.read_csv(DATA_DIR / "sales_history.csv")
    sku_master = pd.read_csv(DATA_DIR / "sku_master.csv")

    feature_parts = []
    for key_values, group in sales_history.groupby(KEY_COLUMNS, sort=False):
        enriched = _add_history_features(group).copy()
        for column, value in zip(KEY_COLUMNS, key_values):
            enriched[column] = value
        feature_parts.append(enriched)

    feature_df = pd.concat(feature_parts, ignore_index=True)
    feature_df = feature_df.merge(
        sku_master[["brand", "finished_good_sku", "sku_type"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    feature_df["decision_week"] = feature_df["week"]  # 예측과 발주 판단을 수행하는 기준 주차
    feature_df["week_sin"] = np.sin(2 * np.pi * feature_df["decision_week"] / 52)  # 연간 시즌 주기의 세로축 성분
    feature_df["week_cos"] = np.cos(2 * np.pi * feature_df["decision_week"] / 52)  # 연간 시즌 주기의 가로축 성분
    feature_df = _add_codes(feature_df)

    feature_df["sales_std_4"] = feature_df["sales_std_4"].fillna(0)
    feature_df[["lag_1", "lag_4", "roll_4", "roll_8"]] = feature_df[
        ["lag_1", "lag_4", "roll_4", "roll_8"]
    ].fillna(0)
    feature_df["recent_trend"] = feature_df["roll_4"] - feature_df["roll_8"]  # 최근 판매속도가 중기 평균 대비 증가/감소했는지 보는 수요 신호
    feature_df["growth_ratio"] = feature_df["roll_4"] / np.maximum(feature_df["roll_8"], 1)  # 최근 수요 성장률을 반영하는 예측 신호
    feature_df["lag1_vs_lag4"] = feature_df["lag_1"] / np.maximum(feature_df["lag_4"], 1)  # 최근 1주 판매가 4주 전 대비 얼마나 변했는지 나타내는 배율

    feature_df = feature_df[feature_df["decision_week"].between(1, 56)].copy()
    feature_df = feature_df[
        [
            "decision_week",  # 예측·배분·발주 판단의 기준 주차
            "brand",  # 수요와 공급 의사결정을 구분하는 브랜드 단위
            "finished_good_sku",  # 재고·발주 제약을 적용하는 완제품 SKU 단위
            "sales_channel",  # 수요 패턴과 배분 우선순위를 구분하는 판매채널
            *FEATURE_COLUMNS,  # 모델이 4주 수요를 판단할 판매·행사·시즌·기준정보 신호
            "actual_4w",  # 예측값과 비교해 정확도·편향을 평가할 향후 4주 실제 판매량
        ]
    ].sort_values(["decision_week", "brand", "finished_good_sku", "sales_channel"])

    feature_df.to_csv(OUTPUT_DIR / "01_feature_table.csv", index=False, encoding="utf-8-sig")
    return feature_df


if __name__ == "__main__":
    build_feature_table()
