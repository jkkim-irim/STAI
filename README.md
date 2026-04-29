# STAI — SVM 자체 구현 (EV 배터리 결함 분류)

EV 배터리 셀의 정량 측정값으로부터 결함 유형(`Defect_Type`)을 분류하는 SVM 모델을 **외부 SVM 라이브러리(sklearn, libsvm 등) 없이** 직접 구현한 중간 프로젝트.

📑 **최종 보고서**: [`jkkim_final_report_2026-04-29.md`](jkkim_final_report_2026-04-29.md)

## 산출물 (룰 1 — 3 SVM 변형 구현)

| 변형 | 모델 파일 | val_acc | val_mF1 |
| --- | --- | --- | --- |
| 선형 SVM (hard-margin) | `models/linear_hard_ovo.pkl` | **0.9617** | **0.860** |
| 선형 분리불가능 SVM (soft) | `models/soft_C100_ovo.pkl` | 0.8699 | 0.794 |
| 비선형 SVM (poly) | `models/kernel_poly_d3_C50_ovo.pkl` | 0.9506 | 0.863 |
| 비선형 SVM (RBF) | `models/kernel_rbf_C300_g005_ovo.pkl` | 0.9495 | 0.833 |

**전체 베스트**: Linear hard-margin + OvO (val 정확도 96.17%).

## 데이터

- 위치: [`datasets/ev_battery_qc_train.csv`](datasets/ev_battery_qc_train.csv)
- 샘플 수: 13,565
- 입력 피처 (6개): `Ambient_Temp_C`, `Anode_Overhang_mm`, `Electrolyte_Volume_ml`, `Internal_Resistance_mOhm`, `Capacity_mAh`, `Retention_50Cycle_Pct`
- 타깃: `Defect_Type` (4 클래스: None / High IR / Poor Retention / Critical Resistance)

테스트 데이터는 미공개 — 평가는 별도 test set 으로 진행.

## 구현 제약 (룰 3, 4)

| 항목 | 가능 여부 |
| --- | --- |
| `sklearn.svm`, `libsvm`, SVM 전용 wrapper | ❌ 금지 (룰 3) |
| 일반 수치 최적화 라이브러리 (`cvxopt` QP, `scipy.optimize` 등) | ✅ 허용 (룰 4) |
| `numpy` / `pandas` 기본 수치 연산 | ✅ 허용 |

## 디렉터리 구조

```
STAI/
├── jkkim_final_report_2026-04-29.md   ← 최종 보고서 (그림 포함)
├── README.md                           ← 이 파일
│
├── train.py                            ← 학습 CLI
├── predict.py                          ← 예측 CLI (--out 미지정 시 outputs/ 자동)
├── environment.yml                     ← conda env 정의
│
├── src/                                ← SVM 코어 + 데이터 파이프라인
│   ├── svm.py                            (LinearHard, Soft, Kernel, OvR/OvO)
│   └── data.py                           (CSV 로더, Scaler, Encoder, split)
│
├── scripts/                            ← 보조 스크립트
│   ├── sweep.py                          1차 hyperparameter sweep
│   ├── sweep_cv.py                       2차 5-fold CV sweep
│   └── plot_results.py                   13 PNG 자동 생성
│
├── datasets/                           ← 학습 데이터
│   └── ev_battery_qc_train.csv         (제공 — 13,565 행)
│
├── models/                             ← 학습된 4 OvO 모델 (.pkl)
├── figures/                            ← 보고서용 PNG (4 카테고리, 13 개)
├── run_logs/                           ← 학습 로그 (8 개 .log)
│
├── outputs/                            ← predict.py 의 기본 저장 위치 (live runs)
├── outputs_examples/                   ← 200 샘플 demo 예측 결과 (제출용)
│   ├── input_200_samples.csv
│   └── result_*.csv (4 모델 분)
│
├── docs/                               ← 과제 원문 + 강의자료
│   ├── 중간 프로젝트 설명.txt
│   └── 11_SVM.pdf
│
└── dvcc/                               ← 단계별 작업 기록
    ├── 00_overview.md
    ├── 01_refactoring_guide.md
    ├── 02_commit_convention.md
    ├── 03_sweep_2026-04-29.md
    └── 04_results_2026-04-29.md
```

## 사용법

### 학습 재현 (예: 베스트 모델)

```bash
conda env create -f environment.yml && conda activate stai

python train.py --variant linear --max-train-samples 8000 \
    --class-weight none --seed 42 --multiclass ovo \
    --data datasets/ev_battery_qc_train.csv \
    --out models/linear_hard_ovo.pkl
```

다른 변형 학습 명령은 [최종 보고서](jkkim_final_report_2026-04-29.md) 6 절 참조.

### 예측 — `predict.py` 한 줄

```bash
# --out 생략 시 outputs/<입력파일명>_pred.csv 자동 저장
python predict.py --model models/linear_hard_ovo.pkl --in 입력CSV.csv

# --out 명시도 가능
python predict.py --model models/linear_hard_ovo.pkl \
    --in 입력CSV.csv --out outputs/result.csv
```

입력 CSV 요구사항:
- 위 **6 피처 컬럼이 정확한 이름으로 존재**해야 함
- 다른 컬럼 (`Cell_ID`, `Batch_ID` 등) 은 그대로 통과
- `Defect_Type` 컬럼이 있으면 정확도/혼동행렬도 자동 출력

### 예측 결과 형식

출력 CSV = 입력 컬럼 그대로 + 마지막에 **`Defect_Type_Pred`** 열 추가.

데모: [`outputs_examples/`](outputs_examples/) — 200 행 stratified 샘플 + 4 모델 예측 결과.

## 평가 기준 충족

| 기준 (`docs/중간 프로젝트 설명.txt`) | 충족 |
| --- | --- |
| 1. 선형/소프트/비선형 SVM 구현 | ✅ 4 모델 |
| 2. 훈련/예측 코드 (CSV → CSV) | ✅ `train.py`, `predict.py` |
| 3. sklearn/libsvm 미사용 | ✅ |
| 4. 외부 최적화 라이브러리 (`cvxopt`) | ✅ |
| 5. 제공 데이터셋 6 피처 | ✅ |
| 6. 알고리즘 완성도 | ✅ 강의 식 ↔ 코드 매핑 ([`dvcc/04_results_2026-04-29.md`](dvcc/04_results_2026-04-29.md) 부록 D) |
| 7. 훈련 데이터 정확도 | ✅ val_acc=0.9617, val_mF1=0.860 |
