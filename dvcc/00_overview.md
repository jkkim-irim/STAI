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

## 작업 체크리스트

- [x] 데이터 로딩 / 전처리 (스케일링, 라벨 인코딩, train/val 분할) 모듈 → `src/data.py`
- [x] 선형 SVM (hard-margin) 구현 + 검증 → `LinearHardMarginSVM` (cvxopt 듀얼 QP, C=1e6)
- [x] 소프트 마진 SVM (C 하이퍼파라미터) 구현 + 검증 → `SoftMarginSVM`
- [x] 비선형 커널 SVM (RBF / poly) 구현 + 검증 → `KernelSVM`
- [x] 다중 클래스 전략 결정 및 구현 → One-vs-Rest (`MultiClassOvR`, `class_weight='balanced'`)
- [x] 학습 스크립트 → `train.py` (variant flag, pickle 저장)
- [x] 예측 스크립트 → `predict.py` (모델 + CSV → CSV with `Defect_Type_Pred`)
- [x] 하이퍼파라미터 스윕 → `scripts/sweep.py` + `dvcc/03_sweep_2026-04-29.md`
- [ ] full-data 최종 학습 + train/val 정확도 측정
- [ ] 변형별 비교 결과 정리 → `dvcc/04_results_*.md`
- [ ] 산출물 정리 (제출 준비)

---

## 변경 이력

| 날짜 | 변경 내용 |
| --- | --- |
| 2026-04-29 | STAI(SVM 자체 구현) 프로젝트용으로 개요 문서 재작성 |
| 2026-04-29 | Phase 0\~3 완료 (env+git, data pipeline, SVM core, CLI, sweep). 스윕 베스트: poly C=10 d=4 (val_f1=0.8373) |
