# 커밋 컨벤션

Conventional Commits 기반. 메시지만 보고 변경 종류와 범위를 즉시 파악 가능하게.

## 구조

```
<type>[(scope)]: <description>

[body]
```

## 타입

| 타입 | 의미 | 예시 |
|------|------|------|
| `feat` | 기능 추가 | `feat(kernel): add RBF kernel implementation` |
| `fix` | 버그 수정 | `fix(smo): correct bias update for soft-margin case` |
| `refactor` | 리팩터링 | `refactor(train): extract feature scaling into module` |
| `docs` | 문서 | `docs: update SVM project overview` |
| `chore` | 빌드/패키지/스크립트 | `chore: add requirements.txt` |
| `test` | 테스트 | `test(linear): add toy 2D linear-separable case` |
| `perf` | 성능 개선 | `perf(kernel): vectorize gram matrix computation` |
| `style` | 포맷팅만 | `style: apply black formatter` |

## 규칙

- 제목 50자 이내, 첫 글자 소문자, 마침표 없음
- 명령문 사용 (Add, Fix — Added, Fixed 아님)
- 본문은 **무엇을/왜** 위주 (어떻게는 코드에)
- scope 예시: `data`, `linear`, `soft`, `kernel`, `smo`, `train`, `predict`, `multiclass`

## 커밋 절차

1. `git status` + `git diff --stat` 으로 변경 파악
2. 위 규칙에 맞게 메시지 작성
3. 관련 파일만 `git add` (데이터셋·모델 파일·임시 출력 CSV 는 `.gitignore` 로 관리)
4. `git commit` → `git log -1 --oneline` 으로 검증
5. push 요청 시: `git push` (추적 없으면 `git push -u origin <branch>`)

## 커밋 단위 가이드

- 데이터 로딩 / 전처리 / SVM 변형(linear, soft, kernel) / 학습 스크립트 / 예측 스크립트는 **각각 분리된 커밋**으로.
- 동일 변경에서 여러 SVM 변형을 동시에 건드리는 경우, scope 를 생략하고 본문에 영향 범위 명시.
