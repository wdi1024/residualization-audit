# HANDOFF — 모의 TMLR 리뷰 대응 (2026-08-01, EGX 세션에서 수행)

> **✅ 완료 (v48-히스토리 세션, 2026-08-01 03:10, v49)**: 아래 글쓰기 8건 전부 반영 + `grouped_full.py`로
> 본문 인용 전 수치의 grouped 세트 재산출 (cert/Δ_adv/CI/3-pop/y2/quartile/feature-split/floor/
> WikiQA scorer 복제/slice MRR). 중대 발견: ASNQ decorrelation 기준이 grouped에서 **실패**
> (0.047→0.139, 게이트 0.05) → 논문에서 철회 처리(‡), §3.4 실패조건의 실증 사례로 수록.
> query-level ranking은 전 세팅 하락 → attribution 프레임 강화 증거로 수록 (App F).
> 산출물: proxy_repair_TMLR_main_v49.pdf (34p, ?? 0, overfull 0).

## 상황
`notes/mock_tmlr_review_2026-08-01.md` (Leaning Reject 4/5) 도착 → 실험적 must-fix 2건을
이 폴더에서 완료. **남은 것은 전부 글쓰기 계열이며 v48 히스토리를 가진 세션이 진행.**

## 완료된 검증 (이 폴더의 스크립트·json)

### ① Query-grouping 재실행 (리뷰 #6 — 누수 실재했음)
QA 스크립트 전부가 행 단위 `KFold(5, shuffle=True)`였음 (MC run만 GroupKFold).
question-GroupKFold 재계산 결과 (`wikiqa_grouped_result.json`, `qa_grouped_results.json`,
`squad_duorc_grouped_results.json`):

| dataset | slice gain 행→Group | C2 R² 행→Group |
|---|---|---|
| WikiQA | +0.177 → **+0.181** | 0.435 → 0.314 |
| ASNQ | +0.247 → **+0.200** | 0.515 → 0.377 |
| TriviaQA | +0.152 → **+0.148** | 0.418 → 0.385 |
| SQuAD | +0.177 → **+0.176** | 0.493 → 0.491 |
| DuoRC | +0.061 → **+0.056** | 0.318 → 0.283 |

- headline 생존, verdict 뒤집힘 0 (C2 게이트 0.05는 전부 여유 통과)
- **논문 수치를 grouped 값으로 교체해야 함** (특히 ASNQ와 R² 열 전체)
- CV 서술을 "question-grouped 5-fold cross-fitting"으로 변경
- 행KFold 재현치가 논문 기록과 일치함을 확인한 뒤 delta를 신뢰했음

### ② Scale sensitivity (리뷰 #7)
5 데이터셋 × {logit(원본)/prob/rank-Gaussian/winsorized-z}, grouped CV
(`scale_sensitivity_results.json`):
- **20셀 전부 slice gain 양수·full gain 음수 — verdict 불변**
- 단 크기는 표현 의존: prob에서 체계적 축소 (DuoRC +0.056→+0.022)
- **rank-Gaussian이 5/5 최강** (ASNQ +0.245, WikiQA +0.216)
  → 권고: 논문에서 rank-Gaussianization을 operator 기본 전처리로 규정
  (표현 임의성 제거 + 성능 향상 = 리뷰 §7을 방법 개선으로 전환)

## 남은 작업 (글쓰기 — 리뷰 섹션 번호 기준)
1. **#2/#3 주장 축소 (최중요)**: label-defined slice 결과를 "construct-validity repair 직접
   증거"가 아니라 "label-conditioned sensitivity/error-attribution diagnostic"으로 재기술.
   코드 reward-model audit을 intervention-validated 주증거로 승격, 두 regime 분리 (#12).
2. **#5 C1 재정의**: repairability screen → slice-informativeness/readout 조건. 역할 단일화.
3. **#6 certificate 개명**: "empirical index-decorrelation diagnostic" 또는 CI-기반 pass rule.
4. **#4 SQuAD second reading**: "A2-independent" 아님 — "disjoint annotator-derived reading,
   shared conventions unresolved"로 한정 (v43 한정이 전 문장에 전파됐는지 확인).
5. **#11 query-level ranking**: slice별 MRR/nDCG 추가 (이 폴더 스크립트 확장으로 가능).
6. **#8 부정확 문장**: "y is used once", "one regression fit" 수정.
7. **#15-16 수사·용어**: honest/license 반복 제거, â→"surface predictor", 제목 후보 검토.
8. 사전 실패 조건 명시 (#13): C1∧C2 pass인데 independent gain≤0이면 실패 선언 등.

## 재실행 환경
`/Users/wondaein/ai-vault/experiments/expectation_grounded/api_run/.venv313/bin/python`
(py3.13 + sklearn/scipy/datasets). 데이터·스코어는 전부 `../cache/` (HF 재다운로드는
wiki_qa/squad/duorc 소량).
