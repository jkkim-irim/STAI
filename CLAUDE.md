# CLAUDE.md

> Claude Code 가 STAI 리포에서 작업할 때 참고하는 프로젝트별 가이드.

## 프로젝트 한 줄 요약

EV 배터리 셀 측정값 → `Defect_Type` 다중 클래스 분류용 SVM 을 **외부 SVM 라이브러리 없이** 직접 구현하는 과제 (`docs/중간 프로젝트 설명.txt`).

## 절대 금지

- `sklearn.svm.*`, `libsvm`, 그 외 SVM 전용 wrapper 사용 금지. SVM 자체를 구현해 주는 라이브러리는 어떤 것도 import 하지 말 것.
- 일반 수치 최적화 (`scipy.optimize`, `cvxopt` QP solver, 직접 작성한 SMO/SGD) 와 `numpy`/`pandas` 는 자유롭게 사용 가능.

## 데이터셋 불변사항

- 파일: `datasets/ev_battery_qc_train.csv` (13,565 행)
- **학습 입력 피처 (6개, 이 순서)**: `Ambient_Temp_C`, `Anode_Overhang_mm`, `Electrolyte_Volume_ml`, `Internal_Resistance_mOhm`, `Capacity_mAh`, `Retention_50Cycle_Pct`
- **타깃**: `Defect_Type` (다중 클래스)
- **학습에 사용하지 않을 것**: `Cell_ID`, `Batch_ID`, `Production_Line`, `Shift`, `Supplier` — 식별자/메타데이터일 뿐 결함 유형과의 인과 관계가 없음.
- 피처 스케일이 서로 매우 다르므로 (`Capacity_mAh` 수천 단위 vs `Anode_Overhang_mm` 0.1 단위) **표준화/정규화는 필수**. 학습 시 사용한 스케일러 파라미터(평균/표준편차 등)는 모델과 함께 저장하고 예측 시 재사용해야 함.

## 코드 구조 원칙

- 단일 파일 800줄 이하 유지. 그 이상이면 모듈 분리 검토.
- 학습 / 예측 스크립트는 **CLI 인자로 입출력 경로를 받음** (과제 요구사항: "csv 파일을 지정해 주면 예측 결과를 csv 파일로 저장").
- 모델 파일에는 다음을 함께 저장: 가중치/서포트벡터/하이퍼파라미터, 라벨 인코더, 스케일러 파라미터. 예측 시 추가 정보 없이 모델 파일만으로 동일 파이프라인 재현 가능해야 함.
- random seed 는 명시적으로 고정. 같은 입력·같은 seed → 같은 모델.

## 운영 문서

세부 사항은 `dvcc/` 참조:

- [dvcc/00_overview.md](dvcc/00_overview.md) — 프로젝트 개요 / 체크리스트
- [dvcc/01_refactoring_guide.md](dvcc/01_refactoring_guide.md) — 코드 품질 원칙
- [dvcc/02_commit_convention.md](dvcc/02_commit_convention.md) — 커밋 메시지 규칙

## 문서화 규칙

- 중간 과정에서 기록이 필요한 결정/분석/기획은 `dvcc/` 폴더에 `.md`로 추가.
- **매 작업 시작 시 `dvcc/` 디렉토리를 먼저 확인할 것.** 기존 문서 맥락을 반영하고, 내용이 낡았으면 수정, 더 이상 유효하지 않거나 불필요하면 **파일 삭제도 허용**. `dvcc/`는 살아있는 노트 폴더로 관리.
- `README.md`는 사용자용 매뉴얼 — 직접 요청 없으면 수정 금지.

## Claude 작업 규칙

- 한국어로 응답.
- 새 파일을 만들기 전에 기존 파일 수정으로 해결 가능한지 먼저 검토.
- 커밋은 사용자가 명시적으로 요청할 때만 수행하고, 메시지는 `dvcc/02_commit_convention.md` 규칙을 따른다.
