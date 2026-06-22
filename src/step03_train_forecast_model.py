"""
[FILE PURPOSE]
- feature table을 기반으로 4주 수요예측 모델을 학습하고, 검증 WAPE 기준으로 최종 모델을 선택한다.
- 예측 정확도, bias, hit rate를 산출해 이후 재고 배분과 발주 판단에 사용할 forecast_4w를 확정한다.
- Step02에서 구조화한 판매속도·추세·변동성·프로모션·시즌성·SKU/채널 신호를 actual_4w 학습 target과 연결한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차

[INPUT]
- outputs/01_feature_table.csv: SKU·채널·주차별 예측 feature와 actual_4w
- outputs/01_inference_feature_table.csv: 61주차 운영 forecast용 feature이며 actual_4w 없음

[OUTPUT]
- outputs/02_backtest_forecast_result.csv: 테스트 구간 forecast_4w, actual_4w, 오차 지표
- outputs/02_forecast_result.csv: 61주차 기준 62~65주차 운영 forecast이며 actual_4w 없음
- outputs/model_selection_summary.csv: 모델별 validation/test 성과와 최종 선택 모델
- forecast_4w는 Step04 채널 재고배분의 수요 기준이며, SKU 단위 집계를 거쳐 Step05 발주 액션에 간접 반영된다.

[현업 적용 시 교체 대상]
- 실제 POS/출고 기반 feature, 회사 표준 forecast accuracy 지표, 모델 후보군, validation/test 기간 정책으로 교체한다.
"""

from pathlib import Path
import os

import numpy as np
import pandas as pd

os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor


OUTPUT_DIR = Path("outputs")


# [LOGIC TYPE] Business Logic Feature
FEATURE_COLUMNS = [
    "lag_1",  # 직전 1주 판매속도
    "lag_4",  # 4주 전 판매와의 비교 기준
    "roll_4",  # 단기 수요 수준을 나타내는 최근 4주 평균
    "roll_8",  # 일시 변동을 완화한 최근 8주 평균
    "sales_std_4",  # 안전재고 판단과 연결되는 단기 수요 변동성
    "recent_trend",  # 단기 평균과 중기 평균 간 상승·하락 방향
    "growth_ratio",  # 중기 수요 대비 최근 수요 성장 배율
    "lag1_vs_lag4",  # 최근 1주 수요의 4주 전 대비 변화 배율
    "promo_flag",  # 프로모션 실행 여부
    "promo_uplift",  # 행사로 발생한 추가 수요 상승폭
    "week_sin",  # 연간 시즌성의 순환 위치 신호
    "week_cos",  # 연간 시즌성의 보완 순환 신호
    "brand_code",  # 브랜드별 수요 차이 식별값
    "sku_type_code",  # SKU 생애주기·회전 유형 식별값
    "channel_code",  # 판매채널별 수요 패턴 식별값
]


# [LOGIC TYPE] Business Logic Feature + ML Hygiene
TARGET_COLUMN = "actual_4w"  # Step02가 미래 4주 판매를 분리해 만든 모델 학습 target


def _split_feature_table(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """의사결정 주차 순서대로 학습·검증·테스트 기간을 분리한다.

    과거 수요 패턴으로 학습한 모델을 더 최근 기간에서 검증하고 테스트하여,
    실제 S&OP 운영처럼 미래 정보가 과거 학습에 섞이지 않도록 한다. 무작위
    분할을 사용하지 않으므로 프로모션·시즌·수요 변화의 시간 순서를 유지한다.

    Args:
        feature_df: Step02가 생성한 주차·SKU·채널 단위 feature 테이블.

    Returns:
        시간순으로 분리된 학습, 검증, 테스트 데이터프레임.
    """
    # ============================================================
    # [BLOCK] 학습·검증·테스트 기간 분리
    # [LOGIC TYPE] Business Grain Design + ML Hygiene
    # [현업 의미] 과거 실적으로 모델을 학습하고, 최근 기간으로 예측 안정성을 검증한 뒤 의사결정 적용 가능성을 확인한다.
    # [판단 기준] 의사결정 주차 기준 train/validation/test 기간
    # [산출물] train_df, validation_df, test_df
    # [수정 포인트] 실무 적용 시 시즌성, 출시주기, 회계월, S&OP 운영 캘린더에 맞춰 기간을 조정한다.
    # [WHY] 시간 순서를 유지한 별도 검증·테스트 구간이 있어야 모델 선택 성과와 최종 운영 성과를 분리해 평가할 수 있다.
    # [ASSUMPTION] 1~36주 학습, 37~48주 검증, 49~56주 테스트가 수요 패턴 변화와 시즌성을 대표한다고 가정한다.
    # [DESIGN LOGIC] 무작위 분할 대신 과거에서 미래로 이어지는 분할을 사용해 실제 S&OP forecast 시점과 유사한 평가 구조를 만든다.
    # [DATA LINEAGE] outputs/01_feature_table.csv 내부 행을 분할하며 결과는 model_selection_summary.csv와 02_forecast_result.csv에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 실제 S&OP 캘린더, 최소 학습기간, 시즌 이벤트, 출시·단종 구간을 반영한 rolling backtest가 필요하다.
    # [INTERVIEW CHECK] 단일 기간 분할의 우연성을 줄이려면 실제 적용 시 여러 forecast origin으로 검증해야 한다고 설명해야 한다.
    # ============================================================
    train_df = feature_df[feature_df["decision_week"] <= 36].copy()  # 과거 36주를 수요 패턴 학습 기간으로 사용
    validation_df = feature_df[feature_df["decision_week"].between(37, 48)].copy()  # 다음 12주 WAPE로 모델 선택 기준을 검증
    test_df = feature_df[feature_df["decision_week"].between(49, 56)].copy()  # 가장 최근 8주를 실제 운영 적용 전 성과 확인 구간으로 사용
    return train_df, validation_df, test_df


def _build_models() -> dict[str, object]:
    """동일한 수요 feature를 학습할 후보 예측모델을 정의한다.

    판매속도·프로모션·시즌성·SKU/채널 특성 사이의 비선형 관계를 서로 다른
    방식으로 학습하는 모델을 비교하여, 단일 알고리즘을 사전에 고정하지 않고
    validation WAPE가 더 안정적인 후보를 선택할 수 있게 한다.

    Returns:
        모델명과 초기화된 회귀모델을 연결한 후보 모델 딕셔너리.
    """
    # ============================================================
    # [BLOCK] 수요예측 후보 모델 정의
    # [LOGIC TYPE] Business Logic Feature + ML Hygiene
    # [현업 의미] 서로 다른 예측 성향의 모델을 비교해 SKU·채널 수요 패턴에 더 안정적인 모델을 선택한다.
    # [판단 기준] 비선형 판매 패턴, 프로모션 반응, 과적합 방지, 예측 안정성
    # [산출물] 모델 후보 딕셔너리
    # [수정 포인트] 회사 표준 모델, AutoML, 계층형 예측, 신제품 보정 모델을 후보군에 추가한다.
    # [WHY] 서로 다른 비선형 학습 방식의 후보를 같은 검증 WAPE로 비교해 특정 알고리즘을 임의 선택하는 위험을 줄인다.
    # [ASSUMPTION] 두 트리 기반 모델과 현재 hyperparameter가 이 synthetic 데이터의 비교 후보로 충분하다고 가정한다.
    # [DESIGN LOGIC] Random Forest의 평균화 안정성과 HistGradientBoosting의 점진적 오차 보정을 비교하되 과적합 제한값을 함께 둔다.
    # [DATA LINEAGE] 모델 정의는 직접 저장되지 않고 model_selection_summary.csv의 후보별 성과와 02_forecast_result.csv의 최종 forecast에 반영된다.
    # [REAL DATA REPLACEMENT] naive baseline, 회사 표준 모델, 계층형 forecast, hyperparameter backtest 및 모델 운영 제약을 추가해야 한다.
    # [INTERVIEW CHECK] 후보군과 hyperparameter가 최적임을 주장하지 않고 동일 기준 비교를 위한 설계임을 설명해야 한다.
    # ============================================================
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,  # 다양한 수요 패턴을 평균화해 단일 트리의 변동을 낮추는 트리 수
            min_samples_leaf=26,  # 소수 이상치·바이럴 주차에 과적합하지 않도록 확보하는 최소 표본 수
            max_depth=8,  # 프로모션·채널·추세 상호작용을 반영하되 과도한 세분화를 제한하는 깊이
            random_state=42,
            n_jobs=1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=120,  # 점진적으로 수요 오차를 보정하는 학습 반복 횟수
            learning_rate=0.03,  # 급격한 적합보다 안정적인 일반화를 위한 단계별 보정 폭
            l2_regularization=0.05,  # 변동성이 큰 SKU·채널의 과적합을 완화하는 규제 수준
            max_leaf_nodes=63,  # 수요 패턴 세분화 복잡도의 상한
            random_state=42,
        ),
    }


def _add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """SKU·채널별 forecast와 actual을 비교하는 행 단위 지표를 추가한다.

    ``forecast_4w``의 과대·과소 방향과 오차 규모를 함께 남겨 향후 재고 과잉과
    결품 위험을 구분한다. APE와 20% 적중 여부는 서로 규모가 다른 SKU·채널의
    상대적인 예측 품질을 확인하기 위한 보조 기준이다.

    Args:
        df: forecast_4w와 actual_4w를 포함한 예측 결과.

    Returns:
        error, abs_error, ape, bias, hit_20_flag가 추가된 결과.
    """
    # ============================================================
    # [BLOCK] SKU·채널 단위 예측 오차 산출
    # [LOGIC TYPE] Technical Transformation + ML Hygiene
    # [현업 의미] 예측이 재고·발주 판단에 미치는 위험을 행 단위로 확인할 수 있게 오차와 적중 여부를 계산한다.
    # [판단 기준] forecast_4w, actual_4w, 절대오차, APE, bias, 20% 이내 적중 기준
    # [산출물] error, abs_error, ape, bias, hit_20_flag
    # [수정 포인트] 실무 적용 시 회사 표준 정확도 지표, 결품 보정 수요, 채널별 가중 오차를 반영한다.
    # [WHY] 전체 평균 성과뿐 아니라 SKU·채널별 과대·과소예측과 허용 오차 충족 여부를 추적해야 재고 위험을 해석할 수 있다.
    # [ASSUMPTION] 음수 forecast는 0으로 제한하고 20% 이내 오차를 적중으로 보는 기준은 포트폴리오용 synthetic 정책이다.
    # [DESIGN LOGIC] error의 방향, abs_error의 규모, APE의 상대오차, hit flag를 함께 보존해 서로 다른 성과 관점을 제공한다.
    # [DATA LINEAGE] 행 단위 지표는 outputs/02_forecast_result.csv에 직접 저장되고 model_selection_summary.csv와 06_summary_table.csv에 집계 반영된다.
    # [REAL DATA REPLACEMENT] 회사 표준 forecast accuracy, 핵심 SKU·매출 가중치, zero-demand 처리, 허용 오차 기준으로 교체해야 한다.
    # [INTERVIEW CHECK] forecast accuracy 하나만으로 bias 방향과 저수요 SKU의 APE 왜곡을 설명할 수 없다는 점을 방어해야 한다.
    # ============================================================
    result = df.copy()
    result["forecast_4w"] = result["forecast_4w"].clip(lower=0).round().astype(int)  # 발주·배분에 사용할 음수 없는 4주 예측수량
    result["actual_4w"] = result["actual_4w"].round().astype(int)  # 예측 성과 비교 기준인 향후 4주 실제 판매수량
    result["error"] = result["forecast_4w"] - result["actual_4w"]  # 양수는 과대예측, 음수는 과소예측을 의미하는 수량 오차
    result["abs_error"] = result["error"].abs()  # 방향을 제외한 순수 예측 오차 규모
    result["ape"] = np.where(
        result["actual_4w"] > 0,
        result["abs_error"] / result["actual_4w"],
        np.where(result["abs_error"] == 0, 0.0, np.nan),
    )
    result["bias"] = result["error"]  # 재고 과잉 또는 결품으로 이어질 수 있는 예측 편향의 행 단위 값
    result["hit_20_flag"] = result["ape"].fillna(np.inf) <= 0.20  # 실제수요 대비 오차 20% 이내 여부
    return result


def _summarize_metrics(result_df: pd.DataFrame) -> dict[str, float]:
    """후보 모델 선택과 S&OP 위험 해석을 위한 성과지표를 집계한다.

    WAPE와 MAE로 전체 오차 규모를 확인하고, bias로 forecast가 구조적으로
    과대 또는 과소인지 판단한다. forecast accuracy와 20% hit rate는 모델
    성과를 현업이 해석하기 쉬운 정확도·적중률 관점으로 보완한다.

    Args:
        result_df: 행 단위 예측 오차가 계산된 SKU·채널별 결과.

    Returns:
        WAPE, forecast accuracy, bias, hit rate, MAE 집계값.
    """
    # ============================================================
    # [BLOCK] 모델 성과 요약
    # [LOGIC TYPE] Business Logic Feature + ML Hygiene
    # [현업 의미] S&OP 회의에서 모델을 선택할 수 있도록 전체 예측 정확도와 편향을 요약한다.
    # [판단 기준] WAPE, forecast accuracy, bias, hit rate, MAE
    # [산출물] 모델 선택용 성과 지표 딕셔너리
    # [수정 포인트] 실무 적용 시 매출가중 WAPE, 핵심 SKU 가중치, 결품 페널티 기반 지표를 추가한다.
    # [WHY] 후보 모델을 동일한 포트폴리오 수준에서 비교하고 과소예측에 따른 결품 위험까지 확인할 공통 지표가 필요하다.
    # [ASSUMPTION] 모든 SKU·채널 수량을 동일하게 합산한 WAPE·bias·MAE가 모델 선택 목적에 적합하다고 가정한다.
    # [DESIGN LOGIC] WAPE를 1차 선택 기준으로 두고 bias, 20% hit rate, MAE를 보조 진단지표로 함께 기록한다.
    # [DATA LINEAGE] 집계값은 outputs/model_selection_summary.csv에 직접 저장되고 outputs/06_summary_table.csv에 재집계된다.
    # [REAL DATA REPLACEMENT] 매출·마진·결품비용 가중치, 품목 중요도, 회사 KPI 정의 및 zero-demand 정책으로 교체해야 한다.
    # [INTERVIEW CHECK] WAPE 최소 모델이 모든 SKU에서 최선은 아니며 사업 중요도별 지표가 추가될 수 있음을 설명해야 한다.
    # ============================================================
    actual_sum = result_df["actual_4w"].sum()  # WAPE와 bias의 수요 규모 분모
    abs_error_sum = result_df["abs_error"].sum()  # 전체 수요 대비 예측오차 규모를 보는 WAPE 분자
    error_sum = result_df["error"].sum()  # 과대·과소예측 방향을 보는 bias 분자

    # 실제 수요가 없는 구간은 분모가 0이므로 예측 오차 해석이 왜곡되지 않도록 별도 처리한다.
    if actual_sum == 0:
        wape = 0.0 if abs_error_sum == 0 else np.nan
        bias_pct = 0.0 if error_sum == 0 else np.nan
    else:
        wape = abs_error_sum / actual_sum
        bias_pct = error_sum / actual_sum

    return {
        "wape": wape,  # 총 실제수요 대비 총 절대오차 비율
        "forecast_accuracy": 1 - wape if pd.notna(wape) else np.nan,  # WAPE를 정확도 관점으로 변환한 지표
        "bias_pct": bias_pct,  # 총수요 대비 구조적 과대·과소예측 비율
        "hit_rate_20": result_df["hit_20_flag"].mean(),  # SKU·채널 예측 중 오차 20% 이내 비중
        "mae": result_df["abs_error"].mean(),  # SKU·채널당 평균 수량 오차
    }


def _predict_raw(model: object, source_df: pd.DataFrame) -> np.ndarray:
    """Step02 feature를 사용해 보정 전 향후 4주 수요를 예측한다.

    FEATURE_COLUMNS만 모델 입력으로 사용하고 ``actual_4w``는 입력에서 제외해
    target 누수를 방지한다. 음수 예측은 실제 수요·배분·발주 수량으로 사용할 수
    없으므로 0을 하한으로 제한한다.

    Args:
        model: 학습이 완료된 수요예측 회귀모델.
        source_df: Step02 feature와 식별 컬럼을 포함한 예측 대상 데이터.

    Returns:
        음수가 제거된 보정 전 4주 예측수요 배열.
    """
    # [LOGIC TYPE] Business Logic Feature + Technical Transformation + ML Hygiene
    return np.clip(model.predict(source_df[FEATURE_COLUMNS]), a_min=0, a_max=None)


def _predict_with_metrics(
    model: object,
    source_df: pd.DataFrame,
    calibration_factor: float = 1.0,
) -> pd.DataFrame:
    """4주 forecast에 총량 보정을 적용하고 행 단위 평가 지표를 결합한다.

    모델의 원시 예측에 validation 기반 calibration factor를 적용해 구조적인
    총량 편향을 완화하고, actual_4w와 비교 가능한 평가 테이블로 변환한다.

    Args:
        model: 학습이 완료된 수요예측 회귀모델.
        source_df: feature와 actual_4w를 포함한 검증 또는 테스트 데이터.
        calibration_factor: 원시 forecast 총량을 조정하는 보정 배율.

    Returns:
        forecast_4w와 행 단위 오차 지표가 추가된 예측 결과.
    """
    # [LOGIC TYPE] Technical Transformation + ML Hygiene
    result_df = source_df.copy()
    result_df["forecast_4w"] = _predict_raw(model, result_df) * calibration_factor  # 검증기간 총량 편향을 보정한 4주 예측수량
    return _add_row_metrics(result_df)


def _calculate_calibration_factor(model: object, validation_df: pd.DataFrame) -> float:
    """검증기간의 실제수요와 예측 총량 차이로 보정계수를 계산한다.

    예측 총량이 지속적으로 낮거나 높으면 Step04 배분 부족량과 Step05 발주량도
    같은 방향으로 왜곡될 수 있다. 검증기간의 actual_4w 대비 원시 forecast 비율을
    제한된 범위에서 적용해 운영 수요 기준의 구조적 편향을 완화한다.

    Args:
        model: 최종 후보로 선택된 학습 모델.
        validation_df: 보정계수 산정에 사용할 시간순 검증 데이터.

    Returns:
        0.85~1.20 범위로 제한된 forecast 총량 보정계수.
    """
    # ============================================================
    # [BLOCK] 예측 총량 보정계수 산출
    # [LOGIC TYPE] Business Logic Feature + ML Hygiene
    # [현업 의미] 검증기간의 총수요 대비 예측 총량 편차를 보정해 발주 판단이 구조적으로 과소/과대가 되지 않게 한다.
    # [판단 기준] validation actual 합계, raw forecast 합계, 보정계수 상하한
    # [산출물] calibration_factor
    # [수정 포인트] 실무 적용 시 브랜드/채널/품목군별 보정계수 또는 회의 확정 forecast override를 반영한다.
    # [WHY] 모델의 총량 편향이 지속되면 배분과 발주가 일관되게 과소·과대 계산되므로 검증구간 기준 보정이 필요하다.
    # [ASSUMPTION] 검증구간의 actual/forecast 총량 비율이 테스트 기간에도 유효하며 0.85~1.20 범위면 충분하다고 가정한다.
    # [DESIGN LOGIC] 개별 행 순위는 유지하고 총 forecast 수준만 보정하며 극단적 배율은 상하한으로 제한한다.
    # [DATA LINEAGE] 계수는 outputs/model_selection_summary.csv에 직접 저장되고 outputs/02_forecast_result.csv의 forecast_4w에 간접 반영된다.
    # [REAL DATA REPLACEMENT] 브랜드·품목군별 bias 모니터링, rolling calibration, 영업 override와 승인 이력이 필요하다.
    # [INTERVIEW CHECK] calibration은 정확도 개선 보장이 아니라 총량 편향 보정이며 검증 외 기간 안정성을 별도로 확인해야 한다.
    # ============================================================
    raw_forecast_sum = _predict_raw(model, validation_df).sum()  # 보정 전 검증기간 총 예측수량
    actual_sum = validation_df[TARGET_COLUMN].sum()  # 보정 기준이 되는 검증기간 총 실제수요
    # 예측 총량이 0 이하이면 보정 배율을 계산할 수 없으므로 기본값으로 유지한다.
    if raw_forecast_sum <= 0:
        return 1.0
    return float(np.clip(actual_sum / raw_forecast_sum, 0.85, 1.20))


def train_and_select_forecast_model() -> pd.DataFrame:
    """후보 모델을 학습·검증하고 후속 SCM 판단용 forecast를 확정한다.

    ``outputs/01_feature_table.csv``를 시간순으로 분리한 뒤 Step02 feature를
    입력으로, ``actual_4w``를 target으로 학습한다. validation WAPE가 가장 낮은
    모델을 선택하고 총량 calibration을 적용한다. backtest 결과는 actual과 함께
    별도 저장하고, actual_4w가 없는 61주차 inference feature로 생성한 운영
    forecast만 Step04와 Step05의 의사결정 수요 기준으로 전달한다.

    Returns:
        테스트 구간의 식별정보, forecast_4w, actual_4w 및 오차 지표 테이블.
    """
    # ============================================================
    # [BLOCK] 최종 예측 모델 선택 및 forecast 산출
    # [LOGIC TYPE] Business Logic Feature + Business Grain Design + Technical Transformation + ML Hygiene
    # [현업 의미] 검증 성과가 가장 좋은 모델을 선택하고, 이후 배분·발주 판단에 투입할 4주 forecast를 확정한다.
    # [판단 기준] validation WAPE 최소, calibration_factor, test forecast accuracy, bias
    # [산출물] outputs/02_backtest_forecast_result.csv, outputs/02_forecast_result.csv, outputs/model_selection_summary.csv
    # [수정 포인트] 실무 적용 시 S&OP consensus forecast, 영업 override, 품절 보정 수요, 핵심 SKU 가중 모델 선택 기준을 반영한다.
    # [WHY] 예측 후보를 검증 WAPE로 선택하고 테스트 성과와 최종 4주 forecast를 고정해야 후속 배분·발주가 단일 수요계획을 참조할 수 있다.
    # [ASSUMPTION] validation WAPE 최소 모델과 calibration이 61주차 운영 forecast에도 유효하다고 가정한다.
    # [DESIGN LOGIC] backtest 성능평가와 actual이 없는 operation inference를 분리하고 운영 forecast만 Step04로 전달한다.
    # [DATA LINEAGE] 01_feature_table.csv는 backtest로, 01_inference_feature_table.csv는 61주차 operation forecast로 변환한다.
    # [REAL DATA REPLACEMENT] rolling retraining, 모델 registry, consensus override, 품절 보정 target, 운영 기준일의 미지 actual 분리 절차가 필요하다.
    # [INTERVIEW CHECK] 현재 test actual은 성과 검증용이며 실제 미래 운영에서는 존재하지 않는다는 점과 forecast 생성 절차를 구분해 설명해야 한다.
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(OUTPUT_DIR / "01_feature_table.csv")
    feature_df = feature_df.dropna(subset=[TARGET_COLUMN]).copy()
    inference_feature_df = pd.read_csv(OUTPUT_DIR / "01_inference_feature_table.csv")

    if TARGET_COLUMN in inference_feature_df.columns:
        raise ValueError("Operation inference feature must not contain actual_4w")
    missing_features = [column for column in FEATURE_COLUMNS if column not in inference_feature_df.columns]
    if missing_features:
        raise ValueError(f"Operation inference feature is missing model columns: {missing_features}")

    train_df, validation_df, test_df = _split_feature_table(feature_df)
    models = _build_models()
    summary_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        # Step02의 판매속도·추세·변동성·프로모션·시즌·SKU/채널 신호를 입력으로 사용하고 actual_4w를 학습 target으로 둔다.
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        fitted_models[model_name] = model

        validation_result = _predict_with_metrics(model, validation_df)
        metrics = _summarize_metrics(validation_result)
        summary_rows.append(
            {
                "model_name": model_name,  # 비교 대상 수요예측 모델명
                "split": "validation",  # 최종 모델 선택에 사용하는 검증 구간
                "wape": metrics["wape"],  # 검증 총수요 대비 절대오차 비율
                "forecast_accuracy": metrics["forecast_accuracy"],  # 검증구간 예측 정확도
                "bias_pct": metrics["bias_pct"],  # 검증구간 과대·과소예측 방향
                "hit_rate_20": metrics["hit_rate_20"],  # 오차 20% 이내 SKU·채널 비중
                "mae": metrics["mae"],  # SKU·채널당 평균 수량 오차
                "calibration_factor": 1.0,  # 모델 선택 전에는 총량 보정을 적용하지 않은 기준값
            }
        )

    summary_df = pd.DataFrame(summary_rows)
    best_model_name = summary_df.sort_values("wape", ascending=True).iloc[0]["model_name"]  # 최종 forecast 기준으로 채택할 최소 WAPE 모델
    final_model = fitted_models[best_model_name]
    calibration_factor = _calculate_calibration_factor(final_model, validation_df)

    test_result = _predict_with_metrics(final_model, test_df, calibration_factor=calibration_factor)
    test_metrics = _summarize_metrics(test_result)
    summary_df = pd.concat(
        [
            summary_df,
            pd.DataFrame(
                [
                    {
                        "model_name": best_model_name,
                        "split": "test",
                        "wape": test_metrics["wape"],
                        "forecast_accuracy": test_metrics["forecast_accuracy"],
                        "bias_pct": test_metrics["bias_pct"],
                        "hit_rate_20": test_metrics["hit_rate_20"],
                        "mae": test_metrics["mae"],
                        "calibration_factor": calibration_factor,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    summary_df["selected_final_model"] = summary_df["model_name"].eq(best_model_name)
    summary_df.loc[
        (summary_df["model_name"] == best_model_name) & (summary_df["split"] == "validation"),
        "calibration_factor",
    ] = calibration_factor

    result_columns = [
        "decision_week",  # 예측 및 후속 S&OP 판단 기준 주차
        "brand",  # 브랜드별 성과 확인 단위
        "finished_good_sku",  # 재고·발주 판단에 연결되는 완제품 단위
        "sales_channel",  # 채널별 수요와 배분 판단 단위
        "promo_flag",  # 해당 예측시점 프로모션 여부
        "forecast_4w",  # 배분·발주 판단에 투입할 향후 4주 예측수량
        "actual_4w",  # 모델 성과 검증용 향후 4주 실제수량
        "error",  # 예측수량에서 실제수량을 차감한 방향성 오차
        "abs_error",  # 예측오차의 절대 수량
        "ape",  # 실제수요 대비 행 단위 절대오차율
        "bias",  # 과대·과소예측 방향을 유지한 수량 오차
        "hit_20_flag",  # 예측오차 20% 이내 적중 여부
    ]
    backtest_forecast_result = test_result[result_columns].sort_values(
        ["decision_week", "brand", "finished_good_sku", "sales_channel"]
    )

    backtest_forecast_result.to_csv(
        OUTPUT_DIR / "02_backtest_forecast_result.csv",
        index=False,
        encoding="utf-8-sig",
    )

    operation_forecast = inference_feature_df.copy()
    operation_forecast["forecast_4w"] = (
        _predict_raw(final_model, operation_forecast) * calibration_factor
    ).round().astype(int)
    operation_columns = [
        "decision_week",
        "brand",
        "finished_good_sku",
        "sales_channel",
        "promo_flag",
        "forecast_4w",
    ]
    operation_forecast = operation_forecast[operation_columns].sort_values(
        ["decision_week", "brand", "finished_good_sku", "sales_channel"]
    )
    # actual_4w가 없는 61주차 운영 forecast만 Step04 allocation과 Step05 replenishment에 전달한다.
    operation_forecast.to_csv(OUTPUT_DIR / "02_forecast_result.csv", index=False, encoding="utf-8-sig")
    # 후보별 validation/test 성과와 calibration 정보를 별도 저장해 최종 모델 선택 근거를 추적한다.
    summary_df.to_csv(OUTPUT_DIR / "model_selection_summary.csv", index=False, encoding="utf-8-sig")
    return operation_forecast


if __name__ == "__main__":
    train_and_select_forecast_model()
