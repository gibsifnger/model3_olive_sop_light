"""
[FILE PURPOSE]
- feature table을 기반으로 4주 수요예측 모델을 학습하고, 검증 WAPE 기준으로 최종 모델을 선택한다.
- 예측 정확도, bias, hit rate를 산출해 이후 재고 배분과 발주 판단에 사용할 forecast_4w를 확정한다.

[BUSINESS UNIT]
- 브랜드 × 완제품 SKU × 판매채널 × 의사결정 주차

[INPUT]
- outputs/01_feature_table.csv: SKU·채널·주차별 예측 feature와 actual_4w

[OUTPUT]
- outputs/02_forecast_result.csv: 테스트 구간 forecast_4w, actual_4w, 오차 지표
- outputs/model_selection_summary.csv: 모델별 validation/test 성과와 최종 선택 모델

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


TARGET_COLUMN = "actual_4w"  # 발주·배분 판단에 연결되는 향후 4주 예측기간의 실제 수요


def _split_feature_table(feature_df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # ============================================================
    # [BLOCK] 학습·검증·테스트 기간 분리
    # [현업 의미] 과거 실적으로 모델을 학습하고, 최근 기간으로 예측 안정성을 검증한 뒤 의사결정 적용 가능성을 확인한다.
    # [판단 기준] 의사결정 주차 기준 train/validation/test 기간
    # [산출물] train_df, validation_df, test_df
    # [수정 포인트] 실무 적용 시 시즌성, 출시주기, 회계월, S&OP 운영 캘린더에 맞춰 기간을 조정한다.
    # ============================================================
    train_df = feature_df[feature_df["decision_week"] <= 36].copy()
    validation_df = feature_df[feature_df["decision_week"].between(37, 48)].copy()
    test_df = feature_df[feature_df["decision_week"].between(49, 56)].copy()
    return train_df, validation_df, test_df


def _build_models() -> dict[str, object]:
    # ============================================================
    # [BLOCK] 수요예측 후보 모델 정의
    # [현업 의미] 서로 다른 예측 성향의 모델을 비교해 SKU·채널 수요 패턴에 더 안정적인 모델을 선택한다.
    # [판단 기준] 비선형 판매 패턴, 프로모션 반응, 과적합 방지, 예측 안정성
    # [산출물] 모델 후보 딕셔너리
    # [수정 포인트] 회사 표준 모델, AutoML, 계층형 예측, 신제품 보정 모델을 후보군에 추가한다.
    # ============================================================
    return {
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=250,
            min_samples_leaf=26,
            max_depth=8,
            random_state=42,
            n_jobs=1,
        ),
        "HistGradientBoostingRegressor": HistGradientBoostingRegressor(
            max_iter=120,
            learning_rate=0.03,
            l2_regularization=0.05,
            max_leaf_nodes=63,
            random_state=42,
        ),
    }


def _add_row_metrics(df: pd.DataFrame) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] SKU·채널 단위 예측 오차 산출
    # [현업 의미] 예측이 재고·발주 판단에 미치는 위험을 행 단위로 확인할 수 있게 오차와 적중 여부를 계산한다.
    # [판단 기준] forecast_4w, actual_4w, 절대오차, APE, bias, 20% 이내 적중 기준
    # [산출물] error, abs_error, ape, bias, hit_20_flag
    # [수정 포인트] 실무 적용 시 회사 표준 정확도 지표, 결품 보정 수요, 채널별 가중 오차를 반영한다.
    # ============================================================
    result = df.copy()
    result["forecast_4w"] = result["forecast_4w"].clip(lower=0).round().astype(int)
    result["actual_4w"] = result["actual_4w"].round().astype(int)
    result["error"] = result["forecast_4w"] - result["actual_4w"]
    result["abs_error"] = result["error"].abs()
    result["ape"] = np.where(
        result["actual_4w"] > 0,
        result["abs_error"] / result["actual_4w"],
        np.where(result["abs_error"] == 0, 0.0, np.nan),
    )
    result["bias"] = result["error"]
    result["hit_20_flag"] = result["ape"].fillna(np.inf) <= 0.20
    return result


def _summarize_metrics(result_df: pd.DataFrame) -> dict[str, float]:
    # ============================================================
    # [BLOCK] 모델 성과 요약
    # [현업 의미] S&OP 회의에서 모델을 선택할 수 있도록 전체 예측 정확도와 편향을 요약한다.
    # [판단 기준] WAPE, forecast accuracy, bias, hit rate, MAE
    # [산출물] 모델 선택용 성과 지표 딕셔너리
    # [수정 포인트] 실무 적용 시 매출가중 WAPE, 핵심 SKU 가중치, 결품 페널티 기반 지표를 추가한다.
    # ============================================================
    actual_sum = result_df["actual_4w"].sum()
    abs_error_sum = result_df["abs_error"].sum()
    error_sum = result_df["error"].sum()

    # 실제 수요가 없는 구간은 분모가 0이므로 예측 오차 해석이 왜곡되지 않도록 별도 처리한다.
    if actual_sum == 0:
        wape = 0.0 if abs_error_sum == 0 else np.nan
        bias_pct = 0.0 if error_sum == 0 else np.nan
    else:
        wape = abs_error_sum / actual_sum
        bias_pct = error_sum / actual_sum

    return {
        "wape": wape,
        "forecast_accuracy": 1 - wape if pd.notna(wape) else np.nan,
        "bias_pct": bias_pct,
        "hit_rate_20": result_df["hit_20_flag"].mean(),
        "mae": result_df["abs_error"].mean(),
    }


def _predict_raw(model: object, source_df: pd.DataFrame) -> np.ndarray:
    return np.clip(model.predict(source_df[FEATURE_COLUMNS]), a_min=0, a_max=None)


def _predict_with_metrics(
    model: object,
    source_df: pd.DataFrame,
    calibration_factor: float = 1.0,
) -> pd.DataFrame:
    result_df = source_df.copy()
    result_df["forecast_4w"] = _predict_raw(model, result_df) * calibration_factor
    return _add_row_metrics(result_df)


def _calculate_calibration_factor(model: object, validation_df: pd.DataFrame) -> float:
    # ============================================================
    # [BLOCK] 예측 총량 보정계수 산출
    # [현업 의미] 검증기간의 총수요 대비 예측 총량 편차를 보정해 발주 판단이 구조적으로 과소/과대가 되지 않게 한다.
    # [판단 기준] validation actual 합계, raw forecast 합계, 보정계수 상하한
    # [산출물] calibration_factor
    # [수정 포인트] 실무 적용 시 브랜드/채널/품목군별 보정계수 또는 회의 확정 forecast override를 반영한다.
    # ============================================================
    raw_forecast_sum = _predict_raw(model, validation_df).sum()
    actual_sum = validation_df[TARGET_COLUMN].sum()
    # 예측 총량이 0 이하이면 보정 배율을 계산할 수 없으므로 기본값으로 유지한다.
    if raw_forecast_sum <= 0:
        return 1.0
    return float(np.clip(actual_sum / raw_forecast_sum, 0.85, 1.20))


def train_and_select_forecast_model() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 최종 예측 모델 선택 및 forecast 산출
    # [현업 의미] 검증 성과가 가장 좋은 모델을 선택하고, 이후 배분·발주 판단에 투입할 4주 forecast를 확정한다.
    # [판단 기준] validation WAPE 최소, calibration_factor, test forecast accuracy, bias
    # [산출물] outputs/02_forecast_result.csv, outputs/model_selection_summary.csv
    # [수정 포인트] 실무 적용 시 S&OP consensus forecast, 영업 override, 품절 보정 수요, 핵심 SKU 가중 모델 선택 기준을 반영한다.
    # ============================================================
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    feature_df = pd.read_csv(OUTPUT_DIR / "01_feature_table.csv")
    feature_df = feature_df.dropna(subset=[TARGET_COLUMN]).copy()

    train_df, validation_df, test_df = _split_feature_table(feature_df)
    models = _build_models()
    summary_rows = []
    fitted_models = {}

    for model_name, model in models.items():
        model.fit(train_df[FEATURE_COLUMNS], train_df[TARGET_COLUMN])
        fitted_models[model_name] = model

        validation_result = _predict_with_metrics(model, validation_df)
        metrics = _summarize_metrics(validation_result)
        summary_rows.append(
            {
                "model_name": model_name,
                "split": "validation",
                "wape": metrics["wape"],
                "forecast_accuracy": metrics["forecast_accuracy"],
                "bias_pct": metrics["bias_pct"],
                "hit_rate_20": metrics["hit_rate_20"],
                "mae": metrics["mae"],
                "calibration_factor": 1.0,
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
        "decision_week",
        "brand",
        "finished_good_sku",
        "sales_channel",
        "promo_flag",
        "forecast_4w",
        "actual_4w",
        "error",
        "abs_error",
        "ape",
        "bias",
        "hit_20_flag",
    ]
    forecast_result = test_result[result_columns].sort_values(
        ["decision_week", "brand", "finished_good_sku", "sales_channel"]
    )

    forecast_result.to_csv(OUTPUT_DIR / "02_forecast_result.csv", index=False, encoding="utf-8-sig")
    summary_df.to_csv(OUTPUT_DIR / "model_selection_summary.csv", index=False, encoding="utf-8-sig")
    return forecast_result


if __name__ == "__main__":
    train_and_select_forecast_model()
