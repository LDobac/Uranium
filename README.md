# Uranium

스타크래프트1 테란 밀리 AI 봇. 리플레이 모방 학습(BC) → 강화학습(RL). 학습은 OpenBW(1.16.1), 실전은 SC:R 로컬·LAN 커스텀 게임.
학습·추론은 Python + PyTorch(`python -m uranium`), 게임 연동·데이터 공급·저장소는 Rust(`uranium._core`).

## 현재 상태
- 시간순 기록: [docs/timeline.md](docs/timeline.md)
- 모델 2 계획·진행: [model2](docs/report/plan/2026_11_09_22_18_model2.md)
- 릴리스: [checkpoint1](docs/report/release/2026_11_08_12_21_checkpoint1.md), [checkpoint1_rl](docs/report/release/2026_11_08_13_35_checkpoint1_rl.md), [checkpoint2](docs/report/release/2026_11_17_09_05_checkpoint2.md), [checkpoint3_4](docs/report/release/2026_11_27_21_37_checkpoint3_4.md)

## 문서 위치
- 요구사항(단일 진실 공급원): [docs/spec/](docs/spec/README.md)
- 실행 안내: [docs/guide/](docs/guide/) — 명령 [commands.md](docs/guide/commands.md), SC:R 실전 [scr.md](docs/guide/scr.md), 경로 측정 [measurement_protocol.md](docs/guide/measurement_protocol.md)
- 리포트: `docs/report/<목적>/`(advice, bench, experiment, investigation, milestone, plan, release, review, survey)

## 빠른 시작
- 설치·`.env`·빌드: [commands.md 1절](docs/guide/commands.md#1-설치설정)
- BC 학습 큐·작업자: [commands.md 7절 큐 운영](docs/guide/commands.md#큐-운영-060-r8), [8절](docs/guide/commands.md#8-감시작업자)
- RL: [commands.md 9절](docs/guide/commands.md#9-rl)
- 테스트: `cargo test --workspace`, `<venv python> -m pytest tests/python`
