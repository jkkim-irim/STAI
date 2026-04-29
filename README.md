# STAI — SVM 자체 구현 (EV 배터리 결함 분류)

EV 배터리 셀의 정량 측정값으로부터 결함 유형(`Defect_Type`)을 분류하는 SVM 모델을 **외부 SVM 라이브러리(sklearn, libsvm 등) 없이** 직접 구현한다.

## 산출물

1. **선형 SVM** (hard-margin)
2. **소프트 마진 SVM** (선형 분리 불가능 — slack 변수)
3. **비선형(커널) SVM** (RBF / 다항식 등)
4. **학습 스크립트** — CSV → 모델 파일 저장
5. **예측 스크립트** — 모델 + 입력 CSV → 예측 결과 CSV 저장

## 데이터

- 위치: [datasets/ev_battery_qc_train.csv](datasets/ev_battery_qc_train.csv)
- 샘플 수: 13,565
- 입력 피처 (6개): `Ambient_Temp_C`, `Anode_Overhang_mm`, `Electrolyte_Volume_ml`, `Internal_Resistance_mOhm`, `Capacity_mAh`, `Retention_50Cycle_Pct`
- 타깃: `Defect_Type` (다중 클래스)
- 비입력 컬럼 (학습에 사용하지 않음): `Cell_ID`, `Batch_ID`, `Production_Line`, `Shift`, `Supplier`

테스트 데이터는 미공개 — 평가는 강사가 보유한 별도 테스트셋으로 진행된다.

## 구현 제약

| 항목 | 가능 여부 |
| --- | --- |
| `sklearn.svm`, `libsvm`, SVM 전용 wrapper | 금지 |
| 일반 수치 최적화 라이브러리 (`scipy.optimize`, `cvxopt` QP 등) | 허용 |
| `numpy` / `pandas` 기본 수치 연산 | 허용 |

## 디렉터리 구조

```
STAI/
├─ README.md            # 이 파일
├─ CLAUDE.md            # Claude Code 작업 가이드
├─ datasets/            # 학습 CSV
├─ docs/                # 과제 원문, 보조 문서
└─ dvcc/                # 프로젝트 운영 문서 (개요/리팩토링/커밋 컨벤션)
```

## 사용법 (예정)

```bash
# 학습: 데이터셋 → 모델 파일
python train.py --data datasets/ev_battery_qc_train.csv --out model.npz

# 예측: 모델 + 입력 CSV → 예측 CSV
python predict.py --model model.npz --in input.csv --out pred.csv
```

## 평가 기준

1. 알고리즘 완성도 — 세 가지 SVM 변형의 수식·로직 정확성
2. 학습 데이터 정확도 (제공)
3. 테스트 데이터 정확도 (미제공 — 일반화 성능)

## 문서

- [프로젝트 개요](dvcc/00_overview.md)
- [리팩토링 가이드](dvcc/01_refactoring_guide.md)
- [커밋 컨벤션](dvcc/02_commit_convention.md)
- [과제 원문](docs/중간%20프로젝트%20설명.txt)
