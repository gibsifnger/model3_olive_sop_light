# Model 3 — Olive S&OP Light MVP

## 1. 프로젝트 개요

본 프로젝트는 고성장 K-뷰티 브랜드사의 S&OP 운영 상황을 가정해 만든 **수요예측·채널별 재고배분·보충발주 의사결정 MVP**입니다.

단순히 SKU별 수요를 예측하는 데서 끝나지 않고, 예측값을 실제 SCM 의사결정 흐름에 연결하는 것을 목표로 했습니다.

전체 흐름은 다음과 같습니다.

```text
채널별 수요예측
→ SKU 공용재고 부족 여부 판단
→ shortage 발생 시 채널별 allocation priority 계산
→ 채널별 allocation_qty / unfulfilled_qty 산출
→ SKU 단위 보충발주 action 결정
→ Forecast Accuracy 및 decision trace 출력
```

본 MVP는 실제 기업 데이터를 사용하지 않고 synthetic data를 기반으로 구현했습니다. 따라서 예측 정확도 수치 자체보다, **Forecast Accuracy를 시간순 backtest로 정의하고, forecast_4w를 allocation 및 replenishment action으로 연결하는 의사결정 구조**를 검증하는 데 초점을 두었습니다.

---

## 2. 비즈니스 문제 정의

고성장 브랜드사는 브랜드, SKU, 판매채널이 동시에 늘어나면서 다음과 같은 문제가 반복됩니다.

* 히어로 SKU는 특정 채널에서 결품 발생
* 저회전 SKU는 과재고 및 유통기한 압박 발생
* 프로모션, 해외 채널 확장, 리테일 입점으로 수요 변동성 증가
* 판매채널별 수요는 증가하지만, 실제 재고는 SKU 단위 공용재고로 관리
* 제한된 재고를 어느 채널에 우선 배분해야 하는지 판단 필요
* 예측값이 발주, 배분, 보류, 감량검토 action으로 연결되지 않음

따라서 본 프로젝트는 수요예측 모델 자체보다, **수요예측값을 운영 가능한 S&OP 의사결정으로 변환하는 구조**에 집중했습니다.

---

## 3. Decision Row Unit

본 프로젝트의 기본 예측 단위는 다음과 같습니다.

```text
decision_week × brand × finished_good_sku × sales_channel
```

각 row는 다음 의미를 가집니다.

> 특정 주차에, 특정 브랜드의 특정 완제품 SKU가 특정 판매채널에서 향후 4주 동안 얼마나 판매될지 예측하는 단위

예시:

| decision_week | brand    | finished_good_sku | sales_channel   |
| ------------- | -------- | ----------------- | --------------- |
| 49            | GlowLab  | GL_HERO_SERUM     | Amazon_US       |
| 49            | GlowLab  | GL_HERO_SERUM     | D2C             |
| 49            | AquaMuse | AM_BUNDLE_KIT     | Domestic_Retail |
| 49            | PureLeaf | PL_GLOBAL_CREAM   | Japan_Offline   |

여기서 `sales_channel`은 판매채널입니다. 구매채널이 아닙니다.
또한 `finished_good_sku`는 완제품 SKU입니다. 원료, 부자재, 용기, 단상자 등의 구성품은 본 MVP 범위에 포함하지 않았습니다.

---

## 4. 데이터 스키마

본 프로젝트는 5개의 synthetic input CSV를 생성합니다.

### 4.1 sales_history.csv

판매 이력 데이터입니다.

| 컬럼                | 설명          |
| ----------------- | ----------- |
| week              | 주차          |
| brand             | 브랜드         |
| finished_good_sku | 완제품 SKU     |
| sales_channel     | 판매채널        |
| sales_qty         | 실제 판매수량     |
| promo_flag        | 프로모션 여부     |
| promo_uplift      | 프로모션 예상 상승률 |

---

### 4.2 sku_master.csv

SKU 운영 조건 데이터입니다.

| 컬럼                | 설명      |
| ----------------- | ------- |
| brand             | 브랜드     |
| finished_good_sku | 완제품 SKU |
| sku_type          | SKU 유형  |
| moq               | 최소 발주수량 |
| box_multiple      | 박스 단위   |
| shelf_life_week   | 유통기한    |
| unit_cost         | 단가      |
| stockout_cost     | 결품 비용   |
| holding_cost      | 보관 비용   |
| disposal_cost     | 폐기 비용   |

---

### 4.3 channel_master.csv

판매채널별 배분 우선순위 기준 데이터입니다.

| 컬럼                     | 설명         |
| ---------------------- | ---------- |
| sales_channel          | 판매채널       |
| channel_type           | 채널 유형      |
| service_penalty_weight | 결품 패널티 가중치 |
| strategic_weight       | 전략채널 가중치   |
| margin_weight          | 마진 가중치     |
| lead_time_week         | 채널 대응 리드타임 |
| allocation_min_rate    | 최소 배분 비율   |

---

### 4.4 inventory_snapshot.csv

SKU 단위 공용재고 데이터입니다.

| 컬럼                  | 설명      |
| ------------------- | ------- |
| decision_week       | 판단 주차   |
| brand               | 브랜드     |
| finished_good_sku   | 완제품 SKU |
| on_hand_inventory   | 현재고     |
| blocked_inventory   | 출고불가 재고 |
| available_inventory | 가용재고    |

본 MVP에서는 재고를 판매채널별로 분리하지 않고, `brand × finished_good_sku` 단위 공용재고로 관리합니다. 이 구조에서 채널별 수요 합계가 가용재고보다 클 경우 allocation 문제가 발생합니다.

---

### 4.5 inbound_plan.csv

입고예정 데이터입니다.

| 컬럼                | 설명        |
| ----------------- | --------- |
| decision_week     | 판단 주차     |
| brand             | 브랜드       |
| finished_good_sku | 완제품 SKU   |
| inbound_qty_1w    | 1주 내 입고예정 |
| inbound_qty_2w    | 2주 내 입고예정 |
| inbound_qty_4w    | 4주 내 입고예정 |

`decision_week = 61`은 61주차 입고실적을 뜻하지 않습니다. 61주차 현재를 계획 cut-off로 하며, `inbound_qty_1w`, `inbound_qty_2w`, `inbound_qty_4w`는 각각 향후 1주·2주·4주 이내 누적 입고예정입니다. 특히 `inbound_qty_4w`는 62~65주차 forecast horizon과 동일한 기간의 확정 입고예정 누계로 Step05 순소요 계산에 사용됩니다.

---

## 5. 파이프라인 구조

프로젝트 구조는 다음과 같습니다.

```text
model3_olive_sop_light/
│
├─ README.md
├─ requirements.txt
├─ run_pipeline.py
│
├─ src/
│  ├─ step01_generate_synthetic_data.py
│  ├─ step02_build_features.py
│  ├─ step03_train_forecast_model.py
│  ├─ step04_build_allocation_plan.py
│  ├─ step05_decide_replenishment_action.py
│  └─ utils.py
│
├─ data/
└─ outputs/
```

각 파일의 역할은 다음과 같습니다.

| 파일                                    | 역할                                          |
| ------------------------------------- | ------------------------------------------- |
| step01_generate_synthetic_data.py     | synthetic input data 생성                     |
| step02_build_features.py              | 수요예측 feature table 생성                       |
| step03_train_forecast_model.py        | 수요예측 모델 학습, 검증, 테스트                         |
| step04_build_allocation_plan.py       | 채널별 allocation priority 및 allocation_qty 산출 |
| step05_decide_replenishment_action.py | SKU 단위 replenishment action 결정              |
| utils.py                              | summary table 및 공통 utility 함수               |
| run_pipeline.py                       | 전체 파이프라인 실행                                 |

---

## 6. Synthetic Data 설계

Synthetic data는 모델이 너무 쉽게 맞히지 못하도록 현실적인 난이도로 구성했습니다.

반영한 요소는 다음과 같습니다.

* SKU별 수요 특성 차이

  * hero
  * new
  * slow
  * global
  * bundle
  * basic
* 채널별 수요 특성 차이

  * D2C
  * Domestic_Retail
  * Amazon_US
  * Japan_Offline
* 프로모션 효과의 불확실성
* 주차별 random noise
* 일부 viral spike
* 신제품 초기 변동성
* 저회전 SKU의 간헐적 판매
* 재고 부족 시 관측 판매량이 실제 수요보다 낮게 찍히는 stockout censoring

이 프로젝트는 synthetic data 기반이므로, 예측 정확도를 과도하게 높이는 것보다 실제 S&OP에서 발생할 수 있는 결품, 과재고, 채널별 배분 이슈가 드러나도록 구성했습니다.

---

## 7. Feature 설계

수요예측 feature는 decision_week 기준으로 해당 시점까지 확인 가능한 정보만 사용했습니다.

사용 feature는 다음과 같습니다.

| Feature       | 설명              |
| ------------- | --------------- |
| lag_1         | 직전 1주 판매량       |
| lag_4         | 4주 전 판매량        |
| roll_4        | 최근 4주 평균 판매량    |
| roll_8        | 최근 8주 평균 판매량    |
| sales_std_4   | 최근 4주 판매 변동성    |
| recent_trend  | roll_4 - roll_8 |
| growth_ratio  | roll_4 / roll_8 |
| lag1_vs_lag4  | lag_1 / lag_4   |
| promo_flag    | 프로모션 여부         |
| promo_uplift  | 프로모션 예상 상승률     |
| week_sin      | 주차 계절성          |
| week_cos      | 주차 계절성          |
| brand_code    | 브랜드 인코딩         |
| sku_type_code | SKU 유형 인코딩      |
| channel_code  | 판매채널 인코딩        |

중요한 점은, allocation 및 replenishment action 판단에는 `actual_4w`를 사용하지 않았다는 점입니다.
`actual_4w`는 학습 target 및 사후 Forecast Accuracy 평가에만 사용했습니다.

---

## 8. Forecast Accuracy 설계

본 프로젝트는 4주 누적 수요를 예측합니다.

각 decision_week `t` 기준으로:

```text
feature = t주까지 확인 가능한 정보
actual_4w = t+1 ~ t+4 실제 판매량 합계
forecast_4w = 모델이 예측한 향후 4주 판매량
```

즉, 실제 운영에서 4주 뒤 판매실적이 확정된 후 forecast와 actual을 비교하는 구조를 synthetic data에서 backtest 방식으로 구현했습니다.

---

### 8.1 Backtest Mode와 Operation Mode

모델 성능 검증과 실제 운영 의사결정의 기준시점을 분리합니다.

| 구분 | Backtest Mode | Operation Mode |
| --- | --- | --- |
| decision_week | 1~56주차 | 61주차 현재 기준 |
| 예측 대상 | 각 기준주차의 다음 4주 | 62~65주차 forecast_4w |
| actual_4w | 존재 | 없음 |
| 목적 | 모델 학습·성능 검증 | Step04 allocation 및 Step05 replenishment 입력 |
| feature 파일 | `outputs/01_feature_table.csv` | `outputs/01_inference_feature_table.csv` |
| forecast 파일 | `outputs/02_backtest_forecast_result.csv` | `outputs/02_forecast_result.csv` |

판매이력이 60주차까지 있으므로 향후 4주 actual을 모두 검증할 수 있는 마지막 decision_week는 56주차입니다. 운영 forecast는 60주차까지 확인된 정보로 61주차 feature를 만들고, 아직 발생하지 않은 62~65주차 actual을 사용하지 않습니다.

---

### 8.2 시간순 Split

Random split을 사용하지 않고 시간순으로 train, validation, test를 분리했습니다.

| 구간               | 역할                   |
| ---------------- | -------------------- |
| week <= 36       | train                |
| 37 <= week <= 48 | validation           |
| 49 <= week <= 56 | test                 |
| 57 <= week <= 60 | actual_4w 계산용 미래 실제값 |

---

### 8.3 모델 비교

수요예측 모델은 다음 두 개를 비교했습니다.

* RandomForestRegressor
* HistGradientBoostingRegressor

Validation 구간에서 WAPE가 낮은 모델을 final model로 선택하고, Test 구간에서 최종 Forecast Accuracy를 산출했습니다.

---

### 8.4 Accuracy 지표

사용한 지표는 다음과 같습니다.

| 지표                | 설명                |
| ----------------- | ----------------- |
| WAPE              | 전체 판매량 기준 가중 오차율  |
| Forecast Accuracy | 1 - WAPE          |
| Bias %            | 과대예측/과소예측 방향성     |
| Hit Rate ±20%     | 오차율 20% 이내 row 비율 |
| MAE               | 평균 절대오차           |

계산식은 다음과 같습니다.

```text
error = forecast_4w - actual_4w
abs_error = abs(error)
APE = abs_error / actual_4w
WAPE = sum(abs_error) / sum(actual_4w)
Forecast Accuracy = 1 - WAPE
Bias % = sum(error) / sum(actual_4w)
Hit Rate ±20% = mean(APE <= 0.20)
```

---

## 9. Allocation Logic

본 프로젝트의 핵심 기능은 shortage 발생 시 채널별 allocation priority를 계산하는 것입니다.

Allocation은 다음 순서로 작동합니다.

```text
1. SKU×채널별 forecast_4w 산출
2. brand × finished_good_sku 단위로 forecast_4w 합산
3. total_forecast_4w와 available_inventory 비교
4. shortage_qty 계산
5. shortage 발생 시 sales_channel별 priority_score 계산
6. priority_score가 높은 채널부터 available_inventory 배분
7. allocation_qty와 unfulfilled_qty 산출
```

Step04는 61주차 `available_inventory`만으로 채널별 즉시 공급 가능량을 배분합니다. 따라서 Step04의 `shortage_qty`는 현재 가용재고 기준의 즉시 공급 제약을 의미합니다. `inbound_plan.csv`는 Step04 allocation에 직접 합산하지 않고, Step05에서 신규 발주 필요량을 계산할 때 반영합니다.

priority_score에는 다음 요소를 반영했습니다.

| 요소                     | 설명           |
| ---------------------- | ------------ |
| demand_share           | 해당 채널의 수요 비중 |
| promo_flag             | 프로모션 여부      |
| service_penalty_weight | 결품 시 서비스 패널티 |
| strategic_weight       | 전략채널 가중치     |
| margin_weight          | 마진 가중치       |

Allocation 결과는 `outputs/04_allocation_plan.csv`에 저장됩니다.

주요 컬럼은 다음과 같습니다.

| 컬럼                   | 설명             |
| -------------------- | -------------- |
| forecast_4w          | 채널별 4주 예상수요    |
| total_forecast_4w    | SKU 전체 4주 예상수요 |
| available_inventory  | SKU 공용 가용재고    |
| shortage_qty         | SKU 단위 부족수량    |
| priority_score       | 채널별 배분 우선순위    |
| allocation_qty       | 실제 배분수량        |
| unfulfilled_qty      | 미충족 수요         |
| allocation_fill_rate | 수요 충족률         |
| allocation_reason    | 배분 사유          |

---

## 10. Replenishment Action Logic

Allocation 이후에는 SKU 단위로 보충발주 action을 결정합니다.

판단 단위는 다음과 같습니다.

```text
decision_week × brand × finished_good_sku
```

주요 계산식은 다음과 같습니다.

```text
net_requirement = total_forecast_4w - available_inventory - inbound_qty_4w
inventory_cover_week = available_inventory / max(total_forecast_4w / 4, 1)
```

여기서 `inbound_qty_4w`는 61주차 현재 이미 알고 있는 62~65주차 누적 입고예정입니다. Step04가 현재고만으로 즉시 shortage를 보여준다면, Step05는 현재고와 향후 4주 입고예정을 함께 고려해 실제 추가 발주 필요량과 추천 발주수량을 결정합니다. inbound가 shortage를 모두 커버하면 `hold_with_inbound_monitoring`, 반영 후에도 순소요가 남으면 발주 계열 action이 선택될 수 있습니다.

사용 action은 다음과 같습니다.

| Action                       | 의미                                     |
| ---------------------------- | -------------------------------------- |
| hold                         | 발주 보류                                  |
| hold_with_inbound_monitoring | 입고예정으로 커버 가능하나 모니터링 필요                 |
| order_moq                    | MOQ 기준 발주                              |
| order_2w_cover               | 2주 커버 목적 발주                            |
| order_with_timing_check      | 리드타임상 입고 전 결품 가능성 있어 발주 및 timing 확인 필요 |
| reduce_review                | 과재고 또는 유통기한 압박으로 감량 검토                 |

Replenishment 결과는 `outputs/05_replenishment_decision.csv`에 저장됩니다.

---

## 11. Output Files

파이프라인 실행 시 다음 output 파일이 생성됩니다.

| 파일                                    | 설명                           |
| ------------------------------------- | ---------------------------- |
| outputs/01_feature_table.csv          | 1~56주차 backtest feature와 actual_4w |
| outputs/01_inference_feature_table.csv | 61주차 운영 forecast feature, actual_4w 없음 |
| outputs/02_backtest_forecast_result.csv | test 구간 forecast와 actual 비교 결과 |
| outputs/02_forecast_result.csv        | 61주차 기준 62~65주차 운영 forecast, actual_4w 없음 |
| outputs/model_selection_summary.csv   | 모델 선택 결과                     |
| outputs/03_sku_shortage_summary.csv   | SKU 단위 shortage 요약           |
| outputs/04_allocation_plan.csv        | 채널별 allocation 결과            |
| outputs/05_replenishment_decision.csv | SKU 단위 보충발주 action           |
| outputs/06_summary_table.csv          | 전체 요약 지표                     |

---

## 12. 실행 방법

### 12.1 가상환경 생성

```bash
python -m venv .venv
```

### 12.2 가상환경 활성화

Windows PowerShell 기준:

```bash
.\.venv\Scripts\Activate.ps1
```

### 12.3 라이브러리 설치

```bash
python -m pip install -r requirements.txt
```

### 12.4 파이프라인 실행

```bash
python run_pipeline.py
```

실행 후 `data/`와 `outputs/` 폴더에 CSV 파일이 생성됩니다.

---

## 13. 주요 결과 해석

본 MVP 실행 결과 예시는 다음과 같습니다.

| 지표                           |                             값 |
| ---------------------------- | ----------------------------: |
| Selected Model               | HistGradientBoostingRegressor |
| Calibration Factor           |                      1.093568 |
| Validation WAPE              |                      0.199496 |
| Test WAPE                    |                      0.228493 |
| Test Forecast Accuracy       |                      0.771507 |
| Test Bias %                  |                     -0.076533 |
| Test Hit Rate ±20%           |                      0.552083 |
| Average Allocation Fill Rate |                      0.826558 |
| Total Unfulfilled Qty        |                         3,364 |
| Order Action Count           |                             1 |
| Timing Check Count           |                             1 |
| Reduce Review Count          |                             1 |

해석하면 다음과 같습니다.

* Test Forecast Accuracy는 약 77.2% 수준으로, synthetic data 기반 수요예측에서 과도하게 높지 않은 현실적인 수준입니다.
* Bias %는 약 -7.7%로 일부 과소예측 경향이 있으나 허용 가능한 범위로 판단했습니다.
* 61주차 운영 Allocation Fill Rate는 약 82.7%로, 현재 가용재고 기준 일부 채널 수요가 미충족되는 구조가 확인됩니다.
* Step05에서는 향후 4주 inbound를 반영해 `hold_with_inbound_monitoring` 2건이 발생했고, inbound 반영 후에도 순소요가 남은 SKU에는 발주·timing check 액션이 유지되었습니다.

---

## 14. 포트폴리오 해석 포인트

본 프로젝트의 핵심은 높은 예측 정확도 자체가 아닙니다.

핵심은 다음입니다.

> 예측값을 실제 S&OP 의사결정으로 연결하는 구조

즉, 본 MVP는 다음 역량을 보여주기 위한 프로젝트입니다.

1. SKU×채널 단위 수요예측 구조 설계
2. 시간순 backtest 기반 Forecast Accuracy 정의
3. WAPE, Bias, Hit Rate 기반 예측 성능 관리
4. SKU 공용재고 부족 여부 판단
5. shortage 발생 시 채널별 allocation priority 산출
6. allocation_qty와 unfulfilled_qty 산출
7. MOQ, 리드타임, 유통기한, 입고예정 기반 replenishment action 결정
8. forecast → allocation → replenishment → decision trace 연결

따라서 이 프로젝트는 단순한 ML 예측 모델이 아니라, **수요예측 결과를 SCM 실행 판단으로 전환하는 S&OP 의사결정 MVP**입니다.

---

## 15. 향후 확장 방향

향후 실제 기업 데이터 적용 시 다음 방향으로 확장할 수 있습니다.

* 실제 판매채널별 sell-out 데이터 연동
* 마케팅 캘린더 및 프로모션 계획 feature 추가
* 가격 할인율, 광고비, 리뷰 수, 랭킹 데이터 반영
* 채널별 마진 및 penalty cost 정교화
* lot별 유통기한 및 재고 aging 반영
* OEM/ODM 생산 리드타임 및 MOQ 반영
* BOM 전개를 통한 원부자재 구매계획 연계
* dashboard 기반 S&OP weekly review 체계 구축

---

## 16. 결론

본 프로젝트는 고성장 브랜드사의 S&OP 운영에서 발생하는 수요예측, 재고부족, 채널별 배분, 보충발주 판단 문제를 하나의 경량 MVP로 구조화한 프로젝트입니다.

특히 `forecast_4w`를 단순 수치로 끝내지 않고, `allocation_qty`, `unfulfilled_qty`, `selected_action`, `reason`, `warning`으로 연결함으로써 실제 SCM 담당자가 검토 가능한 decision trace를 남기는 데 초점을 두었습니다.
