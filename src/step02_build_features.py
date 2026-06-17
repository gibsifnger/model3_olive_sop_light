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
    "lag_1",
    "lag_4",
    "roll_4",
    "roll_8",
    "sales_std_4",
    "recent_trend",
    "growth_ratio",
    "lag1_vs_lag4",
    "promo_flag",
    "promo_uplift",
    "week_sin",
    "week_cos",
    "brand_code",
    "sku_type_code",
    "channel_code",
]


KEY_COLUMNS = ["brand", "finished_good_sku", "sales_channel"]  # SKU·채널별 수요 패턴을 분리해 학습하기 위한 판단 단위


def _add_history_features(group: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 과거 판매 기반 수요 신호 생성
    # [현업 의미] 최근 판매속도, 중기 평균, 변동성, 향후 4주 실적을 만들어 예측 모델의 수요 판단 기준으로 사용한다.
    # [판단 기준] 1주 전 판매, 4주 전 판매, 최근 4주/8주 평균, 최근 판매 변동성, 4주 예측기간
    # [산출물] lag, rolling, sales_std_4, actual_4w 컬럼
    # [수정 포인트] 실무 적용 시 품절 보정 판매량, 반품 차감, 비정상 행사 물량 제외 기준을 반영한다.
    # ============================================================
    group = group.sort_values("week").copy()
    sales = group["sales_qty"]

    group["lag_1"] = sales.shift(1)
    group["lag_4"] = sales.shift(4)
    group["roll_4"] = sales.shift(1).rolling(window=4, min_periods=1).mean()
    group["roll_8"] = sales.shift(1).rolling(window=8, min_periods=1).mean()
    group["sales_std_4"] = sales.shift(1).rolling(window=4, min_periods=2).std()
    group["actual_4w"] = sales.shift(-1).rolling(window=4, min_periods=4).sum().shift(-3)  # 예측 성과를 평가할 향후 4주 수요

    return group


def _add_codes(feature_df: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 범주형 운영 기준의 모델 입력화
    # [현업 의미] 브랜드, SKU 유형, 판매채널별 수요 특성을 예측 모델이 구분할 수 있도록 기준정보를 숫자 신호로 변환한다.
    # [판단 기준] 브랜드, SKU 유형, 판매채널
    # [산출물] brand_code, sku_type_code, channel_code
    # [수정 포인트] 실무 적용 시 품목군, 가격대, 거래처군, 국가, 물류권역 등 추가 기준정보를 확장한다.
    # ============================================================
    feature_df = feature_df.copy()
    feature_df["brand_code"] = pd.Categorical(feature_df["brand"]).codes
    feature_df["sku_type_code"] = pd.Categorical(feature_df["sku_type"]).codes
    feature_df["channel_code"] = pd.Categorical(feature_df["sales_channel"]).codes
    return feature_df


def build_feature_table() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 예측용 feature table 구축
    # [현업 의미] S&OP 수요회의에서 보는 과거 판매 흐름과 프로모션 정보를 모델 학습용 테이블로 정리한다.
    # [판단 기준] SKU·채널별 판매 추세, 성장률, 프로모션 여부, 시즌성, 향후 4주 예측기간
    # [산출물] outputs/01_feature_table.csv
    # [수정 포인트] 실무 데이터 적용 시 채널별 결품 보정, 행사 제외/포함 정책, 신제품 초기 수요 보정 로직을 조정한다.
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
    feature_df["week_sin"] = np.sin(2 * np.pi * feature_df["decision_week"] / 52)
    feature_df["week_cos"] = np.cos(2 * np.pi * feature_df["decision_week"] / 52)
    feature_df = _add_codes(feature_df)

    feature_df["sales_std_4"] = feature_df["sales_std_4"].fillna(0)
    feature_df[["lag_1", "lag_4", "roll_4", "roll_8"]] = feature_df[
        ["lag_1", "lag_4", "roll_4", "roll_8"]
    ].fillna(0)
    feature_df["recent_trend"] = feature_df["roll_4"] - feature_df["roll_8"]  # 최근 판매속도가 중기 평균 대비 증가/감소했는지 보는 수요 신호
    feature_df["growth_ratio"] = feature_df["roll_4"] / np.maximum(feature_df["roll_8"], 1)  # 최근 수요 성장률을 반영하는 예측 신호
    feature_df["lag1_vs_lag4"] = feature_df["lag_1"] / np.maximum(feature_df["lag_4"], 1)

    feature_df = feature_df[feature_df["decision_week"].between(1, 56)].copy()
    feature_df = feature_df[
        [
            "decision_week",
            "brand",
            "finished_good_sku",
            "sales_channel",
            *FEATURE_COLUMNS,
            "actual_4w",
        ]
    ].sort_values(["decision_week", "brand", "finished_good_sku", "sales_channel"])

    feature_df.to_csv(OUTPUT_DIR / "01_feature_table.csv", index=False, encoding="utf-8-sig")
    return feature_df


if __name__ == "__main__":
    build_feature_table()
