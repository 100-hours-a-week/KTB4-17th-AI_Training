# [F2] 온보딩 태깅 SLM 증류 v0 결과 보고서

작성일: 2026-11-02
대상: `kakaocorp/kanana-2-3b-instruct` + LoRA `tagging-v0`
teacher: `Qwen/Qwen3.8-27B-FP8` (self-hosted vLLM 0.30.0)
평가 기준서: [`02_F2_온보딩_태깅_평가기준서.md`](02_F2_온보딩_태깅_평가기준서.md) 5절 배포 게이트

---

## 1. 요약

| 항목 | 결과 | 기준 | 판정 |
| --- | --- | --- | --- |
| JSON 파싱 실패·유령 차원·필드 위반 (strict 검증) | 0 / 47 | 0건 | 통과 |
| 타임아웃 등 `None` 실패율 | 0% | 5% 이하 | 통과 |
| regression 자동 일치율 (`primary` & `off_topic`) | **78.7% (37/47)** | **85% 이상** | **미달** |
| 경계 사례 사람 1/3/5 평가 | 하지 않음 | 평균 3점 이상 | 미측정 |

- 증류로 기본 Kanana 대비 크게 좋아졌다. 완전 일치는 44.7%에서 72.3%로, 형식 실패는 8.5%에서 0%로 바뀌었다.
- 비교 기준 `gemini-3.1-flash-lite` 와 견주면 완전 일치는 v0 이 높고(72.3% 대 66.0%), 게이트 지표는 조금 낮다(78.7% 대 83.0%). 게이트 지표 차이는 47건 중 4건이 갈린 결과(3 대 1)라 통계적으로 구분되지 않는다. 실제 운영 모델 `gemini-2.5-flash-lite` 와는 게이트 지표가 같다(78.7%).
- **비교한 세 모델 모두 F2 배포 게이트 85% 를 넘지 못했다.** 7건은 세 모델이 똑같이 틀렸고 모두 간접 증거 유형이다. 이 유형은 평가 라벨부터 사람 검수로 확인해야 한다.
- v0 만 틀린 3건 가운데 2건이 욕설·농담 비답변에 태그를 붙인 경우다. 운영에서 사용자에게 엉뚱한 성향이 붙는 경로라 v1 에서 먼저 고친다.
- **v0 은 운영 태깅에 넣지 않는다.** 서빙 경로 점검(스테이징)에는 쓸 수 있다.

## 2. 파이프라인과 설정

```
합성 세션 997개 ─ teacher 라벨링(5회 샘플·다수결) ─ 일치율 0.8 이상만 ─ SFT 7,743건 ─ LoRA 학습 ─ eval_tagging.py
 (sub-1000-rewritten)    Qwen3.8-27B-FP8                 from_teacher.py      build_sft.py   train_lora.py   regression 47건
```

### 2.1 teacher 검증 (골드셋, Claude 라벨 기준)

`evals/quality_datasets/claude_gold` 63건(태깅)·24건(추출)으로 쟀다. 골드셋은 사람 검수가 아니라 Claude 라벨이다.

| 지표 | Qwen3.8-27B-FP8 | 기준 |
| --- | --- | --- |
| 태깅 primary F1 | 0.851 (precision 0.74, recall 1.00) | 0.85 이상 |
| 태깅 완전 일치 | 76.2% | - |
| 형식 위반 | 2/315 샘플 (파싱 1, 미등록 차원 1) | 학습 데이터 변환 때 걸러진다 |
| 추출 범위 적중 / 환각 | 24/24 / 0 | 23/24 이상 / 0.6% 이하 |

teacher 단계에서 이미 precision 0.74 로 태그를 과하게 붙이는 경향이 보였다.

### 2.2 대량 라벨링

| 항목 | 값 |
| --- | --- |
| 입력 | `data/synthetic/sub-1000-rewritten` 997세션, 태깅 10,279건 + 추출 997건 |
| 호출 | 52,392회, 실패 0회, 소요 47.9분 |
| 프롬프트 버전 | 태깅 `67757f18f2f3`, 추출 `835686940519` |
| 품질 | 파싱·호출 실패율 0.02%, 5회 샘플 평균 일치율 0.84, 의도한 off_topic 회수율 0.968, primary 빈 비율 2.3% |
| GPU | RunPod RTX PRO 6000 1장, 골드셋 검증과 같은 설정 (`serving/teacher/config.a40-fp8.yaml` 의 FP8 가중치) |

### 2.3 학습 데이터

- `training/tagging/from_teacher.py --min-agreement 0.8` 로 5회 샘플 다수결 일치율 0.8 미만 라벨을 뺐다 (10,279 → 7,743건).
- 0.6 으로 두면 9,687건이지만, 과잉 태깅 노이즈를 줄이려고 0.8 을 골랐다.
- `build_sft.py --allow-unreviewed` 로 train 6,915건 / val 828건을 만들었다. 평가셋 누수·중복 검사는 이 단계에서 통과했다.
- **사람 검수 없이 teacher 라벨만으로 만든 데이터다** (`manifest.json` 의 `unreviewed_allowed: true`).
- train.jsonl sha256 `24a434be17315891e5fdb4df82a24e247031f5c5754196a4e8a95d6a6329d652`, prompt sha256 `67757f18f2f3…`

### 2.4 학습

| 항목 | 값 |
| --- | --- |
| 기본 모델 | `kakaocorp/kanana-2-3b-instruct` 커밋 `6a5d788` (운영 서빙 설정과 같음) |
| 방식 | LoRA r=16, alpha 32, dropout 0.05, all-linear, BF16. 손실은 completion(JSON)에만 |
| 하이퍼파라미터 | 3 epoch, lr 2e-4 cosine, batch 8 × grad accum 2, max length 2048, seed 42 |
| 환경 | torch 2.14.1(cu130), transformers 5.19.0, trl 1.14.0, peft 0.21.2 |
| 스텝·시간 | 1,299스텝, 50.5분, GPU 메모리 약 11GB |
| 어댑터 | 113MB, `adapter_model.safetensors` sha256 `5d447c8147cc5a2020bd622314af7d4c6ecad817560f296bb2fb1485b0e17018` |

| epoch | eval loss | 토큰 정확도 |
| --- | --- | --- |
| 1 | 0.0461 | 98.3% |
| 2 | 0.0306 | 98.8% |
| 3 | 0.0288 | 99.0% |

eval loss 가 마지막 epoch 까지 내려가 과적합 징후는 없다. 다만 이 수치는 teacher 라벨을 얼마나 그대로 따라 쓰는지만 보여 준다.

## 3. 평가 결과

`scripts/eval_tagging.py` 로 regression split(유효 47건)을 쟀다. 서빙 코드의 `TaggingAgent` 를 그대로 호출하고 strict 검증을 켠다.
두 모델 모두 같은 vLLM 서버(LoRA 활성, `max-model-len 4096`, `generation-config vllm`)에서 쟀다.

| 지표 | tagging-v0 | 기본 Kanana |
| --- | --- | --- |
| 완전 일치 | **72.3%** | 44.7% |
| 게이트 지표 (`primary` & `off_topic`) | **78.7%** | 46.8% |
| primary 일치 | 78.7% | 46.8% |
| secondary 일치 | 91.5% | 70.2% |
| off_topic 일치 | 97.9% | 91.5% |
| off_topic 양성 탐지 | 3/4 | 4/4 |
| 형식 실패 | 0% | 8.5% |
| primary precision / recall | 0.72 / 1.00 | 0.22 / 0.17 |
| 지연 p50 / p95 | 0.52s / 2.22s | 0.48s / 0.80s |

### 3.1 카테고리별 완전 일치 (tagging-v0)

| 카테고리 | 건수 | tagging-v0 | 기본 Kanana |
| --- | --- | --- | --- |
| direct | 11 | 81.8% | 18.2% |
| low_or_unknown | 11 | 81.8% | 18.2% |
| indirect_evidence | 11 | **36.4%** | 45.5% |
| insufficient_evidence | 5 | 100% | 100% |
| off_topic | 3 | 100% | 100% |
| short_uncertain | 2 | 100% | 100% |
| abusive_off_topic | 1 | 0% | 100% |
| joke_nonanswer | 1 | 0% | 100% |
| multi_axis_contact / independence | 1 / 1 | 100% / 100% | 0% / 0% |

### 3.2 Gemini 와 비교 (2026-11-03 재측정)

`training/README.md` 의 기준선 flash-lite 2.5(완전 일치 23.4%)는 `TAG_PROMPT` 에 차원 정의를 넣기 전 수치라 쓰지 않는다.
같은 47건, 같은 `TAG_PROMPT`(`67757f18f2f3`), 같은 strict 검증으로 OpenRouter Gemini 를 다시 쟀다.
비교 기준은 `gemini-3.1-flash-lite` 다. `gemini-2.5-flash-lite` 는 현재 운영 `LLM_MODEL` 이라 참고로 함께 적는다.

| 지표 | tagging-v0 | gemini-3.1-flash-lite (기준) | gemini-2.5-flash-lite (운영) |
| --- | --- | --- | --- |
| 게이트 지표 (`primary` & `off_topic`) | 78.7% | **83.0%** | 78.7% |
| 완전 일치 | **72.3%** | 66.0% | 63.8% |
| primary 일치 | 78.7% | 83.0% | 78.7% |
| secondary 일치 | **91.5%** | 76.6% | 80.9% |
| off_topic 일치 | 97.9% | 100% | 100% |
| off_topic 양성 탐지 | 3/4 | 4/4 | 4/4 |
| 형식 실패 | 0% | 0% | 0% |
| primary precision / recall | 0.72 / 1.00 | 0.78 / 1.00 | 0.74 / 0.93 |
| primary 오류 유형 (더 붙임 / 빠뜨림) | 10 / 0 | 8 / 0 | 8 / 2 |
| F2 게이트 85% | 미달 | 미달 | 미달 |

카테고리별 완전 일치:

| 카테고리 | 건수 | tagging-v0 | 3.1-flash-lite | 2.5-flash-lite |
| --- | --- | --- | --- | --- |
| direct | 11 | 81.8% | 81.8% | 100% |
| low_or_unknown | 11 | **81.8%** | 36.4% | 63.6% |
| indirect_evidence | 11 | 36.4% | 45.5% | 36.4% |
| insufficient_evidence | 5 | **100%** | 80.0% | 20.0% |
| abusive_off_topic · joke_nonanswer | 1 · 1 | 0% · 0% | 100% · 100% | 100% · 0% |

v0 이 완전 일치에서 앞서는 이유는 secondary 다. Gemini 는 근거가 약한 답변에 secondary 를 더 붙이고, v0 은 teacher 가 정한 유보 규칙을 더 일관되게 따른다 (`low_or_unknown`, `insufficient_evidence`).

#### 케이스별 대조 (게이트 지표, v0 대 3.1-flash-lite)

| | 3.1 맞음 | 3.1 틀림 |
| --- | --- | --- |
| **v0 맞음** | 36 | 1 (030) |
| **v0 틀림** | 3 (020·027·047) | 7 (018·032·040·048·052·068·072) |

- 둘이 갈린 건 4건뿐이다 (McNemar 정확검정 p≈0.63). 47건으로는 두 모델의 게이트 지표 차이를 말할 수 없다.
- **v0 만 틀림:** 027 욕설 한 단어에 `contact_rhythm`, 047 농담에 `ideal_warmth`, 020 간접 증거에 `contact_rhythm` 을 붙였다. 3.1 은 셋 다 비웠다.
- **3.1 만 틀림:** 030 정답 `problem_solving` 에 `compliance` 를 더 붙였다.
- **둘 다 틀림 (7건):** 018 은 두 모델이 똑같이 `disclosure` 를 더 붙였고, 나머지 6건은 간접 증거 답변의 근거 차원을 primary 로 올렸다. 2.5 도 이 유형에서 같은 실수를 한다. 모델 문제라기보다 평가 라벨과 `TAG_PROMPT` 의 간접 증거 규칙이 모델들에게 전달되지 않는다는 신호다.

#### 지연

v0 p50 0.52s / p95 2.22s, 3.1 0.87s / 1.24s, 2.5 0.80s / 0.99s.
v0 은 Mac 에서 SSH 터널로 잰 값이고 Gemini 는 OpenRouter 경유라 경로가 다르다. 운영 지연 비교는 스테이징 서빙(RunPod 프록시)에서 Langfuse 로 다시 잰다.

## 4. 오류 분석

### 4.1 과잉 태깅 — teacher 오류의 전이

primary 가 틀린 10건은 모두 예측이 정답의 상위집합이다 (빠뜨린 경우 0건).

| 유형 | 건수 | 내용 |
| --- | --- | --- |
| 간접 증거를 primary 로 올림 | 7 | 정답은 primary 없음 + secondary 에만 근거. 학생은 secondary 차원을 primary 로 올렸다 (020·032·040·048·052·068·072) |
| 차원 하나를 더 붙임 | 1 | 018: 정답 `positivity`, 예측 `disclosure, positivity` |
| 비답변에 태그 | 2 | 027(욕설 한 단어)에 `contact_rhythm` 을 붙이고 off_topic 을 놓쳤다. 047(농담)에는 `ideal_warmth` 를 붙였다 |

- 10건 가운데 6건(018·032·048·052·068·072)은 teacher 가 골드셋에서 5회 모두 같은 답으로 틀린 문항(`consistentWrong`)과 겹친다.
- 일치율 필터(0.8)는 teacher 가 흔들리는 라벨만 거른다. teacher 가 일관되게 틀리는 라벨은 걸러지지 않는다.
- 따라서 이 오류는 학습량이나 하이퍼파라미터보다 **라벨의 문제**다.
- 욕설 케이스(027)는 운영에서 사용자에게 엉뚱한 성향 태그가 붙는 경로라 가장 먼저 막아야 한다.

### 4.2 secondary

057·058 은 primary 는 맞고 secondary 에 `assurances` 를 더 붙였다. 게이트 지표에는 들어가지 않는다.

## 5. 한계

- 평가 표본이 47건이라 1건이 2.1%p 다. 카테고리별 수치는 1~11건이라 방향만 본다.
- 평가 라벨과 teacher 골드셋 모두 사람 이중 검수 전이다 (`human_anchor_pending`).
- 기준서는 회귀 유효 집합을 43건으로 적었지만, 현재 데이터셋의 유효 regression 은 47건이다. 기준서를 갱신해야 한다.
- 경계 사례 사람 평가(1/3/5)는 하지 않았다.
- 추출(F3) 학생 모델은 아직 없다 (`training/extraction/` 변환 코드 없음). 추출 라벨 997건은 확보했다.

## 6. 다음 단계

1. **평가 라벨 확인:** 세 모델이 모두 틀린 간접 증거 7건의 정답을 사람이 다시 본다. 라벨이 맞으면 `TAG_PROMPT` 의 간접 증거 규칙을 고치고, 틀리면 데이터셋을 고친다. 지금 기준으로는 운영 모델도 게이트를 넘지 못하니, 게이트 기준과 함께 정리한다.
2. **라벨 교정 (v1):** 간접 증거 답변은 primary 를 비우고 secondary 에만 두는 규칙을 teacher 라벨 후처리나 `TAG_PROMPT` 에 반영한다. 욕설·농담 비답변은 off_topic 예시를 보강한다 (v0 만 틀린 3건 중 2건).
3. **검수:** teacher 라벨 중 간접 증거 유형과 multi-primary(2개 이상, 약 3,800건) 표본을 사람이 확인하고 `reviewed_by` 를 채운다.
4. **게이트 재판정:** v1 을 같은 47건으로 3.1-flash-lite 와 다시 비교한다. 게이트를 넘으면 경계 사례 사람 평가와 `blind_holdout` 을 거쳐 운영 투입을 결정한다.

## 6.1 후속: `TAG_PROMPT` 정황 규칙 (2026-11-03)

4.1절의 공통 오류(정황만 있는 답을 primary 로 올림)를 프롬프트로 고쳤다. 브랜치 `feat/lena-tag-indirect-evidence`.

```
- 사용자가 자기 행동·선호를 직접 말하지 않고, 겪은 일이나 남은 흔적만 전해 성향을 짐작하게 하는 차원은
  primary 가 아니라 secondary 에 둡니다 (예: "운동 앱을 보니 아침마다 기록이 있더라고요" → secondary).
  한 답에 직접 말한 내용과 정황이 섞여 있으면 직접 말한 차원만 primary 이고, 정황으로만 짐작되는 차원은 secondary 입니다.
```

예시 문장은 평가셋·합성 데이터에 없는 표현으로 골랐다. 프롬프트 버전은 `67757f18f2f3` → `5159ed7bc812`.

| 모델 · split | 게이트 지표 (이전 → 규칙) | 완전 일치 (이전 → 규칙) |
| --- | --- | --- |
| 3.1-flash-lite · calibration 16 | 13 → 13 | 7 → 11 |
| 2.5-flash-lite · calibration 16 | 14 → 13 | 13 → 13 |
| 3.1-flash-lite · regression 47 | 39 → 42 (83.0% → 89.4%) | 31 → 37 |
| 2.5-flash-lite · regression 47 | 37 → 44 (78.7% → 93.6%) | 30 → 39 |

- 첫 문구("직접 말한 문장이 하나라도 있으면 그 차원은 primary")는 calibration 에서 섞인 답(042·044)의 정황 차원까지 primary 로 올려 효과가 없었다. 차원 단위로 고친 위 문구를 채택했다.
- regression 은 규칙을 만들 때 케이스를 직접 봤고, calibration 은 문구를 고칠 때 봤다. 두 split 모두 오염됐으므로 위 수치는 방향만 본다. **최종 판정은 `blind_holdout` 으로 한다.**
- calibration 게이트 지표는 그대로이고 완전 일치는 같거나 올랐다. regression 의 개선은 간접 증거 유형에 몰려 있다. 같은 프롬프트에서도 Gemini 출력이 실행마다 1~2건씩 흔들려, 16건 split 의 ±1건은 잡음 범위다.
- 남은 오류: 섞인 답(044)의 정황 차원을 primary 로 올림, 욕설(043)·농담(023·047)에 태그를 붙임, 018 의 `disclosure` (2절 라벨 확장 후보).
- 이 규칙은 운영 Gemini 태깅에도 그대로 적용된다. teacher 재라벨링(v1)도 이 버전으로 한다.

## 7. 재현

| 산출물 | 위치 (Git 제외) |
| --- | --- |
| teacher 라벨 | `data/teacher/qwen-fp8-sub1000/` |
| teacher 골드 채점 | `data/teacher/gold-claude-qwen-fp8-score.json` |
| SFT 데이터 | `training/tagging/out/` |
| 어댑터 | `training/tagging/adapters/tagging-v0/` (RunPod 볼륨 `ktb-slm` 의 `/workspace/backup/2026-11-02/` 에도 백업) |
| 평가 결과 | `evals/results/tagging-tagging-v0-2026-11-02.json`, `evals/results/tagging-kanana-2026-11-02.json`, `evals/results/tagging-gemini-3.1-flash-lite-2026-11-03.json`, `evals/results/tagging-gemini-2.5-flash-lite-2026-11-03.json` |

학습 때 기본 모델을 Pod 로컬 경로(`/root/models/kanana-2-3b-instruct`)로 넘겨서 `adapter_config.json` 의 `base_model_name_or_path` 에 그 경로가 남았다.
그대로면 `serving/vllm/start.sh` 의 기본 모델 일치 검사에서 서버가 뜨지 않는다. Mac 사본은 같은 커밋의 HF ID `kakaocorp/kanana-2-3b-instruct` 로 고쳤다.
볼륨 백업본은 고치기 전 상태다. 다음 학습부터는 `--base-model` 에 HF ID 를 넘기고 `HF_HOME` 캐시를 쓴다.

```bash
uv run python training/tagging/from_teacher.py data/teacher/qwen-fp8-sub1000/tagging.jsonl \
    --min-agreement 0.8 --out data/teacher/qwen-fp8-sub1000/tagging-sft-a0.8.jsonl
uv run python training/tagging/build_sft.py data/teacher/qwen-fp8-sub1000/tagging-sft-a0.8.jsonl --allow-unreviewed
python training/tagging/train_lora.py --base-model kakaocorp/kanana-2-3b-instruct \
    --data-dir training/tagging/out --output-dir training/tagging/adapters/tagging-v0
uv run python scripts/eval_tagging.py --base-url <vLLM>/v1 --model tagging-v0 \
    --out evals/results/tagging-tagging-v0-2026-11-02.json
```
