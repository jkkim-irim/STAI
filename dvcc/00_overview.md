# STAI 프로젝트 개요

> 본 문서는 `STAI` 리포지토리의 중간 프로젝트 — **SVM 자체 구현 (EV 배터리 결함 분류)** — 에 대한 개요를 기술합니다.
> 세부 내용은 작업 진행에 따라 지속 업데이트.

---

## 프로젝트 위치

- 리포 루트: `/home/jkkim/STAI/`
- 데이터셋: `/home/jkkim/STAI/datasets/ev_battery_qc_train.csv`
- 문서: `/home/jkkim/STAI/docs/`, `/home/jkkim/STAI/dvcc/`

---

## 목적

EV 배터리 셀의 정량 측정값으로부터 결함 유형(`Defect_Type`)을 분류하는 SVM 모델을 **외부 SVM 라이브러리(sklearn, libsvm 등) 없이 직접 구현**한다.

산출물:

1. 선형 SVM (hard-margin) — 선형 분리 가능한 경우
2. 소프트 마진 SVM — 선형 분리 불가능한 경우 (slack 변수 도입)
3. 비선형(커널) SVM — RBF / 다항식 등 커널 트릭 적용
4. 학습 코드 — CSV 입력 → 학습 → 모델(가중치/서포트벡터/하이퍼파라미터) 저장
5. 예측 코드 — 모델 + 입력 CSV → 예측 결과 CSV 저장

---

## 데이터셋 스펙

| 항목 | 값 |
| --- | --- |
| 파일 | `datasets/ev_battery_qc_train.csv` |
| 샘플 수 | 13,565 |
| 입력 피처 (6개) | `Ambient_Temp_C`, `Anode_Overhang_mm`, `Electrolyte_Volume_ml`, `Internal_Resistance_mOhm`, `Capacity_mAh`, `Retention_50Cycle_Pct` |
| 타깃 | `Defect_Type` (다중 클래스) |
| 비입력 컬럼 | `Cell_ID`, `Batch_ID`, `Production_Line`, `Shift`, `Supplier` (식별자/메타데이터, 학습에는 미사용) |

> 테스트 데이터는 제공되지 않으며, 평가는 강사가 보유한 미공개 테스트셋으로 진행된다.

---

## 구현 제약

- **금지**: `sklearn.svm`, `libsvm`, `cvxopt.solvers`의 SVM 전용 wrapper 등 SVM 자체를 구현해 주는 라이브러리.
- **허용**: 일반 수치 최적화 라이브러리 (예: `scipy.optimize`, `cvxopt`의 QP solver, 직접 작성한 SMO/SGD 등).
- **권장**: numpy/pandas 기반의 기본 수치 연산은 자유롭게 사용.

---

## 평가 기준

1. 알고리즘 완성도 — 세 가지 SVM 변형이 수식·로직 측면에서 올바르게 구현되었는가
2. 학습 데이터 정확도 (제공된 train CSV 기준)
3. 테스트 데이터 정확도 (미제공 — 일반화 성능)

---

## 파라미터 사전

본 프로젝트의 파라미터는 **두 층 (layer)** 으로 구분된다. 둘은 누가 정하는가, 어떻게 정하는가가 다르다.

### 층 1 · 모델 파라미터 (model parameters) — 학습이 자동 결정

`src/svm.py` 의 `_BaseSVM.fit()` 안에서 cvxopt 듀얼 QP 를 풀어 자동으로 정해지는 값들. 사용자가 손댈 수 없음. 학습 후 `models/*.pkl` 에 저장됨.

| 이름 | 코드 attr | 의미 | 형태 |
| --- | --- | --- | --- |
| `α` (alpha) | `alpha_` | 각 학습 샘플의 듀얼 가중치. 듀얼 QP 의 해(解). `0 ≤ α_i ≤ C`. | 1D 배열 (#SV,) |
| `b` (bias) | `b_` | 결정 함수 `f(x)=…+b` 의 절편. margin 위 SV 의 평균으로 계산. | 스칼라 |
| `w` (weight vector) | `w_` | linear kernel 일 때만 명시 저장: `w = Σ α_i y_i x_i`. 6 차원 결정 평면의 법선. | 1D 배열 (6,) |
| support vectors | `support_vectors_` | `α > tol` 인 학습 점들의 입력 피처. 비선형 kernel 의 예측에 필요. | 2D 배열 (#SV, 6) |
| SV labels | `support_y_` | support vectors 의 ±1 라벨. | 1D 배열 (#SV,) |

**`α_i` 의 해석** (학습 후 SV 들의 역할):

```
α_i = 0           → 영향 없음 (margin 안쪽 정상 점)
0 < α_i < C       → margin 위 SV (결정 경계 형성)
α_i = C           → margin 위반자 (slack 활성, soft margin 에서만)
```

### 층 2 · 하이퍼파라미터 (hyperparameters) — 사람/sweep 이 결정

학습 시작 *전* 에 사용자가 정해줘야 하는 값. 데이터에 따라 베스트가 다르므로 sweep (`scripts/sweep_cv.py`) 으로 비교 탐색.

#### 알고리즘 선택

| 이름 | 가능한 값 | 의미 |
| --- | --- | --- |
| `variant` | `linear` / `soft` / `kernel` | 어느 SVM 변형을 쓸지. CLI `--variant` 로 지정. |
| `kernel` | `linear` / `rbf` / `poly` | `variant=kernel` 일 때 어떤 커널 함수를 쓸지. |

#### 정규화 (regularization)

| 이름 | 영향 | 작은 값 | 큰 값 | STAI 사용 |
| --- | --- | --- | --- | --- |
| `C` | 마진 위반에 대한 페널티 강도. | 부드러운 마진, 위반 봐줌 → 일반화↑, 학습 정확도↓ (under-fit 위험) | 엄격한 마진, 위반 거의 안 봐줌 → 학습 정확도↑, 오버피팅 위험 | hard: `1e6` (사실상 ∞), soft: `100`, kernel poly: `10` |

수식에서의 위치: 듀얼의 박스 제약 `0 ≤ α_i ≤ C`. C 가 클수록 α 의 상한 풀려서 SV 가 더 강하게 결정에 관여 가능.

#### 커널 모양 (kernel-specific)

| 이름 | 적용 대상 | 영향 | 작은 값 / 큰 값 |
| --- | --- | --- | --- |
| `gamma` (γ) | rbf, poly | 한 점이 다른 점에 미치는 영향력 범위 | 작음: 멀리까지 영향 → 부드러운 결정 경계. 큼: 가까운 점만 영향 → 뾰족한 경계 (과적합) |
| `degree` (d) | poly | 다항식의 차수. 결정 경계의 구불거림 정도. | 1=선형, 2=포물선/원, 3-4=S/W 자, 5+=매우 구불구불 (과적합 위험) |
| `coef0` (c) | poly | poly kernel 의 상수항. `K = (γ x·y + c)^d`. | 0=동차, >0=비동차 (편향 추가) |

`gamma` 의 특수 값:
- `'scale'` → `1 / (n_features × Var(X))` (numpy 자동)
- `'auto'`  → `1 / n_features`

#### 다중 클래스 / 불균형 처리

| 이름 | 가능한 값 | 의미 |
| --- | --- | --- |
| `class_weight` | `None` / `'balanced'` | 클래스 불균형 대처. `balanced` 면 binary OvR 안에서 positive 와 negative 클래스에 역빈도 가중 (`n / (2 × n_pos)`, `n / (2 × n_neg)`). 결과적으로 듀얼 박스 제약이 클래스마다 달라짐: `0 ≤ α_i ≤ C × w_i`. |

#### 수치 안정성

| 이름 | 의미 | STAI 기본값 |
| --- | --- | --- |
| `tol` | SV 판정 임계값. `α_i > tol` 만 SV 로 채택. 너무 작으면 수치 노이즈도 SV 로 들어옴. | `1e-5` |
| `ridge` | 듀얼 P 행렬 대각선에 더하는 작은 값. PSD 보장 + 수치 안정. | `1e-8` |

#### 데이터/학습 흐름 (CLI 메타)

| 이름 | 의미 |
| --- | --- |
| `seed` | 분할/셔플의 random seed. **재현성 핵심**. STAI 는 42 고정. |
| `val-ratio` | stratified split 의 val 비율. 0.2 (= 80/20). |
| `max-train-samples` | 학습 데이터 stratified subsample cap. 0 이면 full. linear hard 는 cvxopt 시간복잡도 회피용 8000. |

---

### 층 3 · sweep 자체의 메타-파라미터

스윕 스크립트 (`scripts/sweep_cv.py`) 의 검색 전략 자체를 조절하는 값. **sweep 결과를 신뢰할 수 있는지** 를 결정.

| 이름 | 의미 |
| --- | --- |
| `folds` | K-fold CV 의 K. 5 = 데이터를 5 분할해서 5 번 학습/평가 후 평균. 클수록 추정 안정성↑, 시간↑. |
| `train-cap` | 각 fold 의 학습 stratified subsample cap. 4000 이면 minority 클래스 cap 으로 실효 ~3000. cv 비교 시간 단축용. |
| `seed` | fold 분할 + subsample 셔플 시드. 동일 seed → 동일 비교. |

> sweep 의 메타-파라미터를 바꾸면 "베스트 config" 자체도 바뀔 수 있음. STAI 는 5-fold + cap 4000 기준.

---

## 파라미터 결정의 흐름

```
사용자 / sweep 결정 ─────────► 학습 시작 ────► 자동 결정 (학습 결과)
─────────────────────         ─────────       ──────────────────
variant, kernel,             cvxopt QP        α (#SV 개)
C, gamma, degree, coef0,     풀이             b
class_weight, tol, ridge,                     support_vectors
seed                                          w (linear 만)
                                              모델 = pickle 저장
```

이 흐름이 **한 config 학습 1회**. sweep 은 이 흐름을 **여러 config × 여러 fold** 만큼 반복하며 val_f1 으로 비교 → 사용자에게 추천 config 1 개를 도출.

---

## 작업 체크리스트 (모두 완료)

- [x] 데이터 로딩 / 전처리 (스케일링, 라벨 인코딩, stratified split) → `src/data.py`
- [x] 선형 SVM (hard-margin) 구현 → `LinearHardMarginSVM` (cvxopt 듀얼 QP, C=1e6)
- [x] 소프트 마진 SVM 구현 → `SoftMarginSVM`
- [x] 비선형 커널 SVM (RBF / poly) 구현 → `KernelSVM`
- [x] 다중 클래스 전략 — One-vs-Rest 와 One-vs-One 둘 다 구현
- [x] 학습 스크립트 → `train.py` (variant + multiclass flag, pickle 저장)
- [x] 예측 스크립트 → `predict.py` (`--in` CSV → `outputs/<basename>_pred.csv` 자동 저장)
- [x] 하이퍼파라미터 스윕 (1차 단일 val + 2차 5-fold CV) → `scripts/sweep*.py` + `03_sweep_2026-04-29.md`
- [x] 4 변형 × 2 multiclass = 8 모델 학습 (full data) → `models/*_ovo.pkl`
- [x] 변형별 비교 + OvR vs OvO 분석 → `04_results_2026-04-29.md`
- [x] 시각화 13 PNG 생성 → `figures/`
- [x] 200 행 stratified 샘플 + 4 모델 데모 예측 → `outputs_examples/`
- [x] 최종 보고서 작성 → `jkkim_final_report_2026-04-29.md` (리포 루트)

---

## 디렉터리 구조 (제출용 최종)

```
STAI/
├── jkkim_final_report_2026-04-29.md   ← 최종 보고서
├── README.md
├── train.py / predict.py / environment.yml
├── src/                                코드 (svm.py, data.py)
├── scripts/                            sweep + plot
├── datasets/ev_battery_qc_train.csv    제공 데이터
├── models/                             4 OvO 모델 (.pkl)
├── figures/                            13 PNG (4 카테고리)
├── run_logs/                           학습 로그 (8 .log)
├── outputs/                            predict.py 기본 저장 위치
├── outputs_examples/                   200-샘플 demo 예측
├── docs/                               과제 원문 + 강의자료
└── dvcc/                               단계별 작업 기록 (00~04)
```

---

## 변경 이력

| 날짜 | 변경 내용 |
| --- | --- |
| 2026-04-29 | STAI 프로젝트 개요 작성 — SVM 자체 구현 |
| 2026-04-29 | Phase 0\~3 완료 (env+git, data pipeline, SVM core, CLI, sweep) |
| 2026-04-29 | Phase 4\~5 완료 (full-data 4 변형 학습, OvR/OvO 비교) |
| 2026-04-29 | 시각화 + 데모 예측 생성, 최종 보고서 작성 후 제출 준비 완료 |
