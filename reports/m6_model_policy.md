# M6 B1 — 에이전트 서버 모델 정책 세션

- 날짜: 2026-10-17
- 관련 spec: 001(AMB-001-3), 020-R11·R5, 031-R4(AMB-031-4·5), 032-R14·R15, 041-AC2(AMB-041-2), 042(AMB-042-4)
- 측정 장비: 학습 PC(Windows, RTX 4070 Ti SUPER 16GB, CUDA 13.4, 16 논리 코어). 측정하는 동안 같은 PC에서 리플레이 shard build(BWAPILauncher 12개)가 함께 돌았다.

## 요약
- 에이전트 서버가 BC 체크포인트로 게임을 플레이한다. 흐름은 RawFrame → feature(030) → 배치 `act`(032) → executor(031) → `Act`다.
- 041-AC2(BC 대 T0) 측정 plan `config/plans/bc-vs-t0.toml`도 준비했다. BC 체크포인트가 아직 없어 실행하지 않았다.
- CUDA 실게임 스모크 6판(무작위 초기화 체크포인트 대 T0 테란·저그): 6판 모두 완주, 세션 오류 0, 시야 위반 0, executor 거부 0.
- 결정 지연은 세션 기준으로 잰 큐 대기 + 배치 실행 + host 복사다. p50 42~47ms, p95 최대 55ms(배치 1, CUDA f32).

## 구현

| 영역 | 위치 | 내용 |
|---|---|---|
| crate 분리 (AMB-001-3) | `crates/uranium-runner` | 스케줄러(run·calibrate)를 옮겼다. 바이너리 이름 `uranium-env`과 CLI는 그대로다. `uranium-env`는 라이브러리만 남았다. 의존 방향은 runner → agent → model → env |
| 추론 서버 (032-R7, R14) | `uranium-agent::infer` | (체크포인트, 샘플링 모드)마다 GPU 스레드 하나를 둔다. 세션마다 Core 상태 슬롯을 두고 `select_rows`/`scatter_rows`로 배치 행을 꺼내고 되돌린다. 배치 실행 조건은 `max_batch` 도달, 활성 세션 전원 대기, `max_wait` 경과 중 하나다(기본 32 / 5ms, TBD(M6)). 엔티티 수는 64 배수로 패딩한다 |
| 모델 정책 | `uranium-agent::model_policy` | z 없이 feature를 추출한다. 실행 결과: 명령은 0개 또는 1개(031-R3), 진행 프레임은 delay bucket의 프레임 수(031-R9) |
| executor 거부 (031-R4, AMB-031-5) | 위와 같음 | 거부되면 명령 없이 샘플된 delay만 진행하고 이유별 건수를 기록한다. 에이전트 기억에는 BC(041-R11)와 같이 샘플된 액션과 선택을 기록한다 |
| 라우팅 (020-R11) | `uranium-agent::Routes`, `uranium-runner` | 스케줄러가 `StartGame` 전에 `(game_id, GameSpec.players 인덱스) → 정책`을 등록한다. 인덱스는 hostd가 붙이는 플레이어 이름 `uranium-p<i>`에서 읽는다(B9의 `spec_index`). BWAPI 플레이어 id는 쓰지 않는다(아래 발견 1) |
| 샘플링·behaviour (032-R15, AMB-042-4) | `uranium-model` | temp 0 = argmax, temp 1 = 확률적. `ActOutput.head_log_prob`([B,6])를 추가했다. M6 actor용 `SessionObserver` hook은 결정마다 관측, 액션, 헤드별·joint log-prob, value를 넘긴다. 기본값은 hook 없음 |
| 맵 사전계산 (AMB-031-4) | `uranium-env::maps`, `uranium-replay mapgen` | 최소 MPQ 리더로 .scx에서 CHK를 꺼내 `<맵 파일 SHA-256>.resources.toml`을 쓴다. 게임 시작 시 찾는 순서: ① 맵 파일 해시 사전계산 ② 같은 CHK의 리플레이 build 사전계산 ③ 자원 + `GameInfo.static_map`으로 계산 |
| 세션 기록 (020-R5) | `SessionRecord.policy`·`model` | 정책 버전(`model:<폴더>@<model.bpk SHA-256 12자>:<모드>`), 결정 수, 거부, 지연 p50/p95, 평균 배치, 사전계산 출처를 남긴다 |
| 보고서 | `uranium-runner::report` | Wilson 95% 구간(050-R2)과 모델 세션 표를 추가했다. 모델 판의 승리 귀속은 승자 세션의 정책 기록으로 정한다. 두 세션의 승자가 다르면 "winner disagree"로 센다 |
| CLI·설정 | | `uranium-env run --backend --max-batch --max-wait-ms`, `uranium-agent serve --policy model --checkpoint …`, `uranium-agent random-checkpoint`, `.env` `URANIUM_AGENT_BACKEND` |

## 테스트 (flex, `cargo test --workspace` 146개 통과)

| 테스트 | 확인 내용 |
|---|---|
| `slots_match_single_sessions_032_r14` (LSTM·GTrXL) | 슬롯 부분집합을 돌아가며 배치로 처리해도 세션을 하나씩 처리한 결과와 같다(1e-4) |
| `head_log_probs_and_argmax_032_r15` | 헤드별 log-prob의 합이 joint log-prob과 같다. argmax는 결정론적이다 |
| `plays_valid_acts_031_r4` | 명령 ≤ 1개, 진행 프레임 ∈ bucket, 마스크 위반으로 거부된 결정 0 |
| `argmax_is_deterministic_031_r7` | 같은 프레임 열에서 같은 `Act` 열이 나온다 |
| `batched_sessions_match_single_032_r14` | 세션 3개를 배치로 처리한 결과가 하나씩 처리한 결과와 같다(배치가 실제로 형성됐는지도 확인) |
| `routes_pick_policy_per_slot_020_r11` | 가짜 브리지 2개: 라우트가 있는 slot은 모델, 없는 slot은 T0 |
| `observer_sees_behaviour_log_probs_042` | hook이 결정마다 호출되고 헤드 합 = joint다 |
| `model_side_plays_terran_over_the_bridge_020_r11`, `wilson_interval_050_r2`, MPQ 해시·복호 | plan 확장, Wilson 구간, MPQ 테이블 키 |

## CUDA 실게임 스모크 (`config/plans/smoke-model.toml`)
무작위 초기화 LSTM 체크포인트(기본 크기)를 썼다. 동시 게임 1, 게임당 최대 12,000 프레임이다. 리플레이 build가 CPU를 함께 쓰는 상태에서 측정했다.

| 매치업 | 판 | 결과 | 세션 오류 | 시야 위반 | 거부 | 결정 수 | 결정 ms p50 / p95 | 실행 ms p50 | host ms p50 |
|---|---|---|---|---|---|---|---|---|---|
| 모델(argmax) 대 T0 테란 | 4 | 4 완주 (T0 3승, 1판 승자 불일치) | 0 | 0 | 0 | 7,549 | 46.4 / 55.1 | 46.4 | 0.62 |
| 모델(stochastic) 대 T0 저그 | 2 | 2 완주 (프레임 상한 무승부) | 0 | 0 | 0 | 2,037 | 42.3 / 46.0 | 42.2 | 0.63 |

- 배치는 항상 1이었다(동시 게임 1). 결정 지연은 거의 모두 GPU forward다. host 쪽 feature 추출과 executor는 0.6ms다.
- 맵 사전계산은 투혼 3판이 `replay_build(match=true)`, 서킷브레이커 3판이 `resources`였다.
  - 투혼 1.3 맵 파일의 CHK가 리플레이 build에서 이미 사전계산한 CHK와 같았다. 맵 파일 자원으로 다시 계산한 결과도 그 사전계산과 같았다. 라이브 입력과 BC 입력이 같다는 뜻이다.
  - 서킷브레이커는 리플레이가 없어 자원 목록으로 계산했다.
- 6판 모두 모델 세션의 BWAPI 플레이어 id가 1이었다. 플레이어 이름 기반 라우팅이 모델을 테란 브리지에 정확히 붙였다.

## 발견
1. **BWAPI 플레이어 id가 런처 순서와 다르다(Windows TCP LAN).**
   - T0 테란(p0) 대 T0 저그(p1) 4판 중 2판에서 p1이 id 0을 받았다.
   - 첫 스모크는 `Hello.slot`으로 라우팅했다. 그래서 모델 대 저그 2판에서 모델이 저그 브리지에 붙어 세션 오류가 났다.
   - B9 세션이 같은 문제를 hostd 플레이어 이름 `uranium-p<i>`로 해결해 master에 올려 두었다. 라우팅을 그 방식으로 바꿨고 020-R11 본문에 반영했다.
   - 원격 Linux hostd도 이 이름을 붙이려면 master 빌드가 필요하다. 이름이 없으면 `Hello.slot`으로 대체한다.
2. **같은 게임의 두 세션이 승자를 다르게 보고한다.**
   - 첫 스모크에서 4판 중 2판, 두 번째 스모크에서 1판이었다. 예: 한쪽은 `winner=None`, 다른 쪽은 `winner=0`. 또는 양쪽이 각자 자기가 이겼다고 보고했다.
   - 원인은 조사하지 않았다(브리지 shim의 `onEnd(isWinner)` 경로). 보고서는 첫 세션 값을 쓰고 불일치 판 수를 따로 표시한다.
   - 041-AC2·050 승률 집계 전에 원인을 확인해야 한다.
3. **`URANIUM_MAPS_DIR` 이름을 두 용도가 함께 쓴다.** hostd의 .scx 폴더(020-R9)와 사전계산 폴더(031)다.
   - 이 PC에서는 같은 폴더에 함께 놓여 동작한다.
   - 다만 리플레이 build는 PC마다 로컬 `URANIUM_MAPS_DIR`에 사전계산을 쓴다. 학습 PC에 없는 맵 해시는 BC 학습에서 사전계산 없이(`None`) 들어간다. 전체 build가 끝난 뒤 추출 PC들의 사전계산 파일을 학습 PC로 모아야 한다.

## 남은 일
- BC 체크포인트가 나오면 `config/plans/bc-vs-t0.toml`의 `checkpoint`를 고쳐 041-AC2를 측정한다(조합당 50판, 두 모드).
- 배치 파라미터(`max_batch`, `max_wait_ms`)와 추론 지연 허용치를 측정한다(TBD(M6)). 측정은 동시 게임 여러 개로 한다.
- M6 actor: `SessionObserver` 구현, T=64 경계에서 core_state0 읽기(추론 서버 슬롯), 보상(033)과 privileged 연결.
- 발견 2(승자 불일치)의 원인 확인, 발견 3(사전계산 파일 수집) 처리.
