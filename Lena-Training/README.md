# Lena-Training

KTB4-17th-AI 레포의 모델 학습 관련 자료를 모은 폴더다. 원본 경로 구조를 그대로 유지했다.

| 경로 | 내용 |
| --- | --- |
| `training/` | 온보딩 태깅 LoRA 파인튜닝 파이프라인(`build_sft.py`, `train_lora.py` 등). 사용법은 `training/README.md` |
| `training/tagging/out/` | SFT train/val 분할(합성 데이터, teacher 라벨) |
| `training/tagging/adapters/tagging-v0/` | 학습된 LoRA 어댑터 v0 (`*.safetensors`·`*.bin` 은 Git LFS) |
| `training/extraction/` | 추출 SFT 빌더와 `out/`(train 851·val 84, teacher 라벨). 학습 명령은 `training/README.md` |
| `serving/` | vLLM 서빙·teacher(RunPod) 설정 |
| `scripts/` | 합성 온보딩 생성, 채팅 구독용 수동 시뮬레이터, teacher 라벨링, 라벨 검사, teacher 골드 채점, 태깅·추출 평가 |
| `tests/` | 위 스크립트·파이프라인 테스트 (AI 레포의 `app` 코드에 의존) |
| `data/synthetic/`, `data/teacher/` | 합성 세션과 teacher 라벨 (운영 데이터 없음). `synthetic/v1-add-400` 은 아직 라벨링 전 |
| `data/사람검수표_csv/` | 사람 검수표 |
| `docs/` | 태깅 SLM 증류 v0 결과보고서, 피드백 카드 추출학습 설계·스키마 |
| `training/notebooks/`, `training/RESULTS.md` | LoRA·QLoRA 비교와 teacher 라벨링 Colab 노트북, 학습 결과·게이트 표 |

스크립트와 테스트는 `app` 패키지를 import 하므로 KTB4-17th-AI 체크아웃 루트에서 실행한다.
운영 데이터(`data/gold_work/`)는 공개 레포라 넣지 않았다.
