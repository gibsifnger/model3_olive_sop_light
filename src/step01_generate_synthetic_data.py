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
RANDOM_SEED = 42


def _build_sku_master() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] SKU 구매·재고 기준 마스터
    # [현업 의미] SKU별 발주 제약과 재고 리스크 기준을 정의해 이후 발주/과재고 판단의 기준값으로 사용한다.
    # [판단 기준] MOQ, 박스배수, 유통기한, 단가, 결품비용, 보관비용, 폐기비용
    # [산출물] sku_master.csv의 SKU별 구매조건 및 비용 기준 컬럼
    # [수정 포인트] ERP 품목마스터, 공급사 계약조건, 원가정보, 폐기/보관 비용 기준으로 교체한다.
    # ============================================================
    rows = [
        ("GlowLab", "GL_HERO_SERUM", "hero", 500, 50, 104, 8.5, 18.0, 0.08, 3.2),
        ("GlowLab", "GL_NEW_AMPOULE", "new", 300, 30, 78, 7.2, 15.0, 0.07, 2.8),
        ("PureLeaf", "PL_SLOW_MASK", "slow", 200, 20, 52, 3.6, 7.5, 0.04, 1.4),
        ("PureLeaf", "PL_GLOBAL_CREAM", "global", 400, 40, 104, 6.8, 16.0, 0.06, 2.5),
        ("AquaMuse", "AM_BUNDLE_KIT", "bundle", 250, 25, 52, 11.5, 22.0, 0.10, 4.0),
        ("AquaMuse", "AM_BASIC_TONER", "basic", 350, 35, 104, 4.4, 9.0, 0.05, 1.6),
    ]
    columns = [
        "brand",
        "finished_good_sku",
        "sku_type",
        "moq",  # 공급사 최소 발주수량
        "box_multiple",  # 박스/카톤 단위 발주 배수
        "shelf_life_week",  # 유통기한 기반 폐기 리스크 판단 기준
        "unit_cost",
        "stockout_cost",  # 결품 발생 시 판매기회손실/서비스 리스크 가중치
        "holding_cost",  # 보관비·자금묶임 부담 가중치
        "disposal_cost",  # 유통기한 경과/폐기 리스크 가중치
    ]
    return pd.DataFrame(rows, columns=columns)


def _build_channel_master() -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 판매채널 우선순위 기준 마스터
    # [현업 의미] 제한된 재고를 어느 채널에 먼저 배분할지 판단하기 위한 채널별 서비스·전략 기준을 정의한다.
    # [판단 기준] 서비스 패널티, 전략 중요도, 마진 중요도, 채널 리드타임, 최소 배분율
    # [산출물] channel_master.csv의 채널별 배분 우선순위 기준 컬럼
    # [수정 포인트] 채널별 SLA, 영업 전략, 수익성, 수출/국내 리드타임, 계약상 최소 공급 조건으로 교체한다.
    # ============================================================
    rows = [
        ("D2C", "owned_online", 1.15, 1.25, 1.40, 1, 0.18),
        ("Domestic_Retail", "domestic_wholesale", 1.05, 1.05, 0.95, 2, 0.22),
        ("Amazon_US", "marketplace_global", 1.45, 1.35, 1.20, 4, 0.20),
        ("Japan_Offline", "export_offline", 1.30, 1.15, 0.90, 5, 0.25),
    ]
    columns = [
        "sales_channel",
        "channel_type",
        "service_penalty_weight",
        "strategic_weight",
        "margin_weight",
        "lead_time_week",  # 채널 공급까지 필요한 평균 리드타임
        "allocation_min_rate",  # 채널별 최소 공급률 기준
    ]
    return pd.DataFrame(rows, columns=columns)


def _sku_base_demand(sku_type: str) -> float:
    return {
        "hero": 520,
        "new": 230,
        "slow": 65,
        "global": 210,
        "bundle": 150,
        "basic": 260,
    }[sku_type]


def _channel_multiplier(sku_type: str, sales_channel: str, week: int) -> float:
    base = {
        "D2C": 1.05,
        "Domestic_Retail": 0.85,
        "Amazon_US": 0.75,
        "Japan_Offline": 0.55,
    }[sales_channel]

    if sku_type == "global" and sales_channel in {"Amazon_US", "Japan_Offline"}:
        base *= 1.0 + 0.012 * week
    if sku_type == "bundle" and sales_channel in {"D2C", "Domestic_Retail"}:
        base *= 1.15
    if sku_type == "slow" and sales_channel in {"Amazon_US", "Japan_Offline"}:
        base *= 0.55
    if sku_type == "new" and week <= 12:
        base *= np.interp(week, [1, 12], [0.45, 1.20])
    if week >= 50:
        base *= 0.95

    return base


def _promo_probability(sku_type: str, sales_channel: str) -> float:
    sku_effect = {
        "hero": 0.22,
        "new": 0.16,
        "slow": 0.08,
        "global": 0.14,
        "bundle": 0.30,
        "basic": 0.12,
    }[sku_type]
    channel_effect = {
        "D2C": 0.04,
        "Domestic_Retail": 0.12,
        "Amazon_US": 0.06,
        "Japan_Offline": 0.02,
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
    # [현업 의미] SKU 생애주기, 채널 특성, 시즌성, 프로모션, 바이럴 수요를 반영해 예측 모델이 학습할 수요 패턴을 만든다.
    # [판단 기준] SKU 유형, 판매채널, 시즌성, 프로모션 여부, 신제품 ramp-up, 글로벌 채널 성장성
    # [산출물] true_demand, promo_flag, promo_uplift
    # [수정 포인트] 실제 POS/출고실적, 프로모션 캘린더, 신제품 출시 계획, 채널별 수요 이벤트로 교체한다.
    # ============================================================
    base = _sku_base_demand(sku_type) * _channel_multiplier(sku_type, sales_channel, week)
    seasonality = 1.0 + 0.10 * np.sin(2 * np.pi * week / 13) + 0.06 * np.cos(2 * np.pi * week / 26)
    trend = 1.0 + (0.003 * week if sku_type in {"hero", "basic"} else 0.0)

    noise_sigma = {
        "hero": 0.24,
        "new": 0.55 if week <= 12 else 0.30,
        "slow": 0.65,
        "global": 0.32,
        "bundle": 0.42,
        "basic": 0.18,
    }[sku_type]
    noise = rng.lognormal(mean=0.0, sigma=noise_sigma)

    promo_flag = int(rng.random() < _promo_probability(sku_type, sales_channel))  # 프로모션 여부
    promo_uplift = 0.0
    # 프로모션 발생 시 기본 판매력보다 높은 수요가 발생하므로 채널별 행사 민감도를 추가 반영한다.
    if promo_flag:
        uplift_base = {
            "hero": 0.45,
            "new": 0.28,
            "slow": 0.18,
            "global": 0.25,
            "bundle": 0.65,
            "basic": 0.20,
        }[sku_type]
        channel_promo = 0.18 if sales_channel == "Domestic_Retail" else 0.08
        promo_uplift = max(0.05, rng.normal(uplift_base + channel_promo, 0.18))

    viral_probability = 0.0
    if sku_type == "hero" and sales_channel in {"D2C", "Amazon_US"}:
        viral_probability = 0.035
    elif sku_type == "global" and sales_channel == "Amazon_US":
        viral_probability = 0.030
    elif sku_type == "new" and sales_channel == "D2C" and week <= 16:
        viral_probability = 0.025

    viral_multiplier = rng.uniform(1.8, 3.6) if rng.random() < viral_probability else 1.0

    intermittent_factor = 1.0
    if sku_type == "slow":
        intermittent_factor = 0.0 if rng.random() < 0.35 else rng.uniform(0.35, 1.35)

    true_demand = base * seasonality * trend * noise * (1.0 + promo_uplift) * viral_multiplier * intermittent_factor
    true_demand = max(0, int(round(true_demand)))
    return true_demand, promo_flag, round(promo_uplift, 3)


def _apply_stockout_censoring(
    rng: np.random.Generator,
    demand_df: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 결품에 따른 관측 판매량 보정
    # [현업 의미] 실제 수요가 있어도 재고가 부족하면 판매실적에는 낮게 관측되는 결품 왜곡을 모델링한다.
    # [판단 기준] SKU 공통 가용재고, 총수요, 결품 이벤트 발생 여부, 채널별 배분 가중치
    # [산출물] sales_qty가 반영된 판매이력
    # [수정 포인트] 실제 결품 로그, 재고 제약, 주문 컷오프, 채널별 공급 제한 이력으로 교체한다.
    # ============================================================
    censored_rows = []

    for _, group in demand_df.groupby(["week", "brand", "finished_good_sku"], sort=False):
        total_demand = group["true_demand"].sum()
        shortage_event = rng.random() < 0.18
        # 결품 이벤트가 발생하면 실제 수요보다 낮은 가용재고만 채널에 배분되어 판매실적이 절단된다.
        if shortage_event and total_demand > 0:
            available_for_week = int(round(total_demand * rng.uniform(0.58, 0.93)))
        else:
            available_for_week = int(round(total_demand * rng.uniform(0.98, 1.18)))

        available_for_week = max(0, available_for_week)
        allocation_weight = group["true_demand"].to_numpy(dtype=float)
        allocation_weight = allocation_weight * rng.uniform(0.85, 1.15, size=len(group))
        allocation_weight_sum = allocation_weight.sum()

        if allocation_weight_sum <= 0:
            sales_qty = np.zeros(len(group), dtype=int)
        elif available_for_week >= total_demand:
            sales_qty = group["true_demand"].to_numpy(dtype=int)
        else:
            raw_alloc = available_for_week * allocation_weight / allocation_weight_sum
            sales_qty = np.minimum(group["true_demand"].to_numpy(dtype=int), np.floor(raw_alloc).astype(int))
            remaining = available_for_week - sales_qty.sum()
            if remaining > 0:
                gaps = group["true_demand"].to_numpy(dtype=int) - sales_qty
                candidate_idx = np.where(gaps > 0)[0]
                if len(candidate_idx) > 0:
                    add_idx = rng.choice(candidate_idx, size=min(remaining, len(candidate_idx)), replace=False)
                    sales_qty[add_idx] += 1

        out = group.copy()
        out["sales_qty"] = sales_qty
        censored_rows.append(out)

    return pd.concat(censored_rows, ignore_index=True)


def _build_sales_history(
    rng: np.random.Generator,
    sku_master: pd.DataFrame,
    channel_master: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 판매실적 생성
    # [현업 의미] 예측 모델이 학습할 주차·SKU·채널 단위의 과거 출고/판매 실적을 구성한다.
    # [판단 기준] SKU 유형, 판매채널, 프로모션 여부, 결품으로 절단된 관측 판매량
    # [산출물] sales_history.csv의 sales_qty, promo_flag, promo_uplift
    # [수정 포인트] POS, 출고실적, 온라인 주문, B2B 납품실적, 프로모션 캘린더로 교체한다.
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
                        "week": week,
                        "brand": sku.brand,
                        "finished_good_sku": sku.finished_good_sku,
                        "sales_channel": channel,
                        "true_demand": true_demand,
                        "promo_flag": promo_flag,
                        "promo_uplift": promo_uplift,
                    }
                )

    demand_df = pd.DataFrame(rows)
    sales_df = _apply_stockout_censoring(rng, demand_df)
    return sales_df[
        [
            "week",
            "brand",
            "finished_good_sku",
            "sales_channel",
            "sales_qty",
            "promo_flag",
            "promo_uplift",
        ]
    ].sort_values(["week", "brand", "finished_good_sku", "sales_channel"])


def _build_inventory_snapshot(
    rng: np.random.Generator,
    sales_history: pd.DataFrame,
    sku_master: pd.DataFrame,
) -> pd.DataFrame:
    # ============================================================
    # [BLOCK] 의사결정 주차 재고 스냅샷
    # [현업 의미] 예측수요를 실제로 공급 가능한지 판단하기 위해 현재고, 보류재고, 가용재고를 구성한다.
    # [판단 기준] 최근 판매속도, SKU 유형별 목표 커버 수준, 보류재고 비율
    # [산출물] inventory_snapshot.csv의 on_hand_inventory, blocked_inventory, available_inventory
    # [수정 포인트] WMS/ERP 재고 스냅샷, 품질보류, 예약재고, 출고불가재고 기준으로 교체한다.
    # ============================================================
    recent_sales = (
        sales_history[sales_history["week"].between(53, 60)]
        .groupby(["brand", "finished_good_sku"], as_index=False)["sales_qty"]
        .sum()
    )
    recent_sales["avg_4w_sales"] = recent_sales["sales_qty"] / 2.0
    recent_sales = recent_sales.merge(
        sku_master[["brand", "finished_good_sku", "sku_type"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    rows = []
    for row in recent_sales.itertuples(index=False):
        coverage_factor = {
            "hero": rng.uniform(1.25, 1.70),
            "new": rng.uniform(0.25, 0.45),
            "slow": rng.uniform(2.20, 3.00),
            "global": rng.uniform(0.55, 0.75),
            "bundle": rng.uniform(0.82, 0.95),
            "basic": rng.uniform(2.00, 2.70),
        }[row.sku_type]
        on_hand = int(round(row.avg_4w_sales * coverage_factor))
        blocked = int(round(on_hand * rng.uniform(0.01, 0.08)))
        available = max(0, on_hand - blocked)  # 실제 배분 판단에 사용할 가용재고
        rows.append(
            {
                "decision_week": 61,
                "brand": row.brand,
                "finished_good_sku": row.finished_good_sku,
                "on_hand_inventory": on_hand,
                "blocked_inventory": blocked,
                "available_inventory": available,
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
    # [현업 의미] 신규 발주 필요량을 계산하기 전에 이미 발주되어 들어올 물량을 반영한다.
    # [판단 기준] 최근 판매속도, SKU 유형별 입고 예정 비율, 박스배수
    # [산출물] inbound_plan.csv의 1주/2주/4주 입고예정 수량
    # [수정 포인트] SAP MM 구매오더, 공급사 ASN, 생산계획, 선적/통관 일정으로 교체한다.
    # ============================================================
    recent_sales = (
        sales_history[sales_history["week"].between(49, 60)]
        .groupby(["brand", "finished_good_sku"], as_index=False)["sales_qty"]
        .sum()
    )
    recent_sales["avg_4w_sales"] = recent_sales["sales_qty"] / 3.0
    plan = recent_sales.merge(
        sku_master[["brand", "finished_good_sku", "box_multiple"]],
        on=["brand", "finished_good_sku"],
        how="left",
    )

    rows = []
    for row in plan.itertuples(index=False):
        base = row.avg_4w_sales
        sku_type = sku_master.loc[
            sku_master["finished_good_sku"].eq(row.finished_good_sku),
            "sku_type",
        ].iloc[0]
        inbound_factor = {
            "hero": rng.uniform(0.20, 0.45),
            "new": rng.uniform(0.00, 0.12),
            "slow": rng.uniform(0.00, 0.05),
            "global": rng.uniform(0.55, 0.85),
            "bundle": rng.uniform(0.08, 0.18),
            "basic": rng.uniform(0.00, 0.08),
        }[sku_type]
        inbound_1w = _round_to_multiple(base * inbound_factor * rng.uniform(0.05, 0.25), row.box_multiple)  # 1주 내 입고예정
        inbound_2w = _round_to_multiple(base * inbound_factor * rng.uniform(0.25, 0.55), row.box_multiple)  # 2주 내 입고예정
        inbound_4w = _round_to_multiple(base * inbound_factor, row.box_multiple)  # 발주 판단에 반영할 4주 내 입고예정
        rows.append(
            {
                "decision_week": 61,
                "brand": row.brand,
                "finished_good_sku": row.finished_good_sku,
                "inbound_qty_1w": inbound_1w,
                "inbound_qty_2w": inbound_2w,
                "inbound_qty_4w": inbound_4w,
            }
        )

    return pd.DataFrame(rows)


def _round_to_multiple(value: float, multiple: int) -> int:
    # 박스배수보다 작은 단위로 발주·입고가 불가한 공급 조건을 반영한다.
    if value <= 0:
        return 0
    return int(round(value / multiple) * multiple)


def _save_csv(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")


def generate_all_input_data() -> None:
    # ============================================================
    # [BLOCK] S&OP 입력 데이터 생성 및 저장
    # [현업 의미] 예측, 배분, 발주 판단에 필요한 기준정보와 운영 데이터를 하나의 입력 세트로 준비한다.
    # [판단 기준] SKU 구매조건, 채널 우선순위, 판매실적, 가용재고, 입고예정
    # [산출물] data/ 하위 5개 CSV 입력 테이블
    # [수정 포인트] 실무에서는 각 테이블을 ERP/WMS/POS/프로모션 시스템에서 추출한 데이터로 대체한다.
    # ============================================================
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    sku_master = _build_sku_master()
    channel_master = _build_channel_master()
    sales_history = _build_sales_history(rng, sku_master, channel_master)
    inventory_snapshot = _build_inventory_snapshot(rng, sales_history, sku_master)
    inbound_plan = _build_inbound_plan(rng, sales_history, sku_master)

    _save_csv(sales_history, DATA_DIR / "sales_history.csv")
    _save_csv(sku_master, DATA_DIR / "sku_master.csv")
    _save_csv(channel_master, DATA_DIR / "channel_master.csv")
    _save_csv(inventory_snapshot, DATA_DIR / "inventory_snapshot.csv")
    _save_csv(inbound_plan, DATA_DIR / "inbound_plan.csv")


if __name__ == "__main__":
    generate_all_input_data()
