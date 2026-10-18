# M6 3단계: RL 루프 구현과 CUDA 스모크 (2026-10-18)

3단계 계획서(2026-10-17 사용자 승인, Q1~Q6 추천안)의 구현 결과다. 042-AC1 본 실행은 하지 않았다(사용자 지시). 마지막 절에 준비 상태와 예상 시간을 적었다.

## 구현 (master)

| 커밋 | 내용 |
|---|---|
| `f6919d9` | spec 개정: 042 구현 구조·042-R3 Aborted 회수(AMB-042-6)·AMB-042-5·7·8·9, AMB-050-2, 020-R12, 032-R16, 033-R7·R8 |
| `6177d47` | runner 라이브러리화와 Job 공급자(020-R12), `StopGame` 일괄 중지, 승자 불일치 판정(020-R13, AMB-020-3), 추론 서버 가중치 교체·슬롯 상태 읽기·정책 버전(032-R16), ModelPolicy RL hook |
| `e25dce8` | `uranium-train` RL: return 계산(V-trace·UPGO·GAE), trajectory(T=64 1스텝 겹침 세그먼트, core_state0, zstd 압축 관측, Aborted 회수, 정책 지연 버림), learner(V-trace+UPGO·PPO, 보상 항목별 value, BC 재forward 정확 KL, 헤드별 엔트로피, micro_lanes), 리그(PFSP·승률 EMA·커리큘럼·스냅샷·리셋·시분할·종족·Random·보상 가중 스케줄·042-AC1 고정 설정), 오케스트레이터 `uranium-train rl`(061 storage·ckpt API, 1분 heartbeat, Ctrl+C 정상 종료, CSV 로그), `uranium-train z-table` |
| 이 커밋 | RL 망 dropout 0(032 §2, 아래 발견 1), `checkpoint::load_with`, 이 리포트 |

- 게임 결과는 020-R13 판정(`EndBoard` + `game_result`, 더 늦게 끝난 세션의 승자)을 리그 승률 표, r_win(종료 세그먼트를 판정 뒤에 큐에 넣음), games.csv에 모두 쓴다.
- 정상 종료(042-R10)는 ① 새 게임 중지 ② 진행 중 게임 `StopGame` ③ 체크포인트 push ④ push가 `Pushed`일 때만 잠금 해제(061-R11-5·6, 3B 주의사항) 순서다. heartbeat가 `Lost`면 학습을 멈추고 push 없이 잠금을 그대로 둔다.
- `cargo test --workspace`: 190개 통과, 실패 0(dropout 수정 포함).

## CUDA 스모크 (학습 PC, RTX 4070 Ti SUPER 16GB)

설정:
- 출발 정책: 짧은 표본 BC. `stardata/p00` shard(깨진 p16과 무관)로 120 update, 약 1시간. 수렴하지 않은 약한 정책이다.
- 042-AC1 고정 설정(`--ac1-race terran`): 투혼, T0 표준 테란.
- 이 PC hostd만 사용, 동시 3게임. z 테이블 없음(전 게임 z 없음, 033-R8).
- `rl.toml` 사본에서 `lanes`만 8로 줄였다(update 주기를 짧게). 나머지는 초기값 그대로.
- 길이는 20분이다(계획서의 1시간은 사용자 지적으로 줄임). 체크포인트 간격 10분.

| 항목 | 결과 |
|---|---|
| 게임 | 48판. 45판 Finished(모두 패배), 3판은 종료 시 `StopGame`으로 중지 → 학습 제외(042-R3) |
| update | 66회(20분), update당 471~504 결정 |
| 처리량 | **약 32 결정/초**(3게임). 게임당 결정 간격은 추론 지연이 대부분 |
| 추론 지연 (세션별 중앙값) | 결정 p50 54ms, p95 350ms. 배치 실행 p50 51ms, p95 334ms(learner update와 GPU 경쟁). 평균 배치 2.9 |
| 정책 지연 | 평균 2.1 update, 버린 세그먼트 0, 큐 넘침 0 |
| ρ 평균 / KL(π‖π_BC) | 0.94 / 0.031(첫 update는 1.0 / 0) |
| executor 거부, 시야 위반 | 0 / 0 |
| 승자 불일치 (020-R13) | 45판 중 4판. 모두 더 늦게 끝난 세션의 보고를 채택했다(버그 아님, 기록만) |
| GPU 메모리 / 프로세스 RAM | 약 5.3GB / 0.5~0.8GB |
| 체크포인트 | `ckpt-14406`(10분), `ckpt-32570`(종료 시) 모두 `Pushed`. 내용: 에이전트 3개 가중치, optimizer 3개, `league.bin`, `train_state.json`, 설정 사본, manifest. 종료 후 잠금 해제 확인 |
| 세션 오류 | 4건, 모두 종료 시 중지된 게임의 연결 끊김(10054) |

### 발견 1: learner가 dropout 0.1로 학습하고 있었다 (수정함)
- 첫 스모크에서 π = π_BC인 첫 update의 KL이 0.18, ρ 평균이 0.87로 나왔다.
- 원인: BC 체크포인트 설정(entity dropout 0.1)을 RL에서도 그대로 써서, learner forward에만 dropout이 걸렸다. 032 §2는 "dropout은 BC 0.1, RL 0"으로 정한다.
- 수정: RL 망은 `checkpoint::load_with`로 dropout 0 설정으로 만든다. 수정 후 첫 update의 KL 0, ρ 1.0을 확인했다.

### 발견 2: 승자 불일치는 이번에도 나왔다 (규칙으로 처리)
- 예: 게임 `…170004`. 우리 세션은 frame 8156에 "승자 = 자기", T0 세션은 frame 8187에 "승자 = 자기"를 보고했다.
- 020-R13대로 늦게 끝난 T0 쪽을 채택해 패배(r_win −1)로 처리했다. 규칙이 없었다면 승리로 잘못 학습됐을 판이다.
- 리플레이는 저장하지 않아(`save_replay=false`) 리플레이 대조는 하지 않았다. 원인(OpenBW 퇴장 타이밍) 수정은 범위 밖이다.

## 042-AC1 준비 상태
- 실행 명령(본 BC 체크포인트가 나온 뒤):
  `uranium-train rl --run ac1-<algo>-s<seed> --bc <BC ckpt> --algo <vtrace|ppo> --seed <s> --backend cuda --ac1-race <BC 승률 20~80% 종족> --max-updates <1,000만 step에 해당하는 수>`
- 남은 선행 조건:
  - 본 BC 체크포인트(BC 데이터 세션 진행 중).
  - 041-AC2로 BC 승률이 20~80%인 T0 종족을 고른다(TBD(M6)).
  - 1,000만 step에서 멈추는 옵션: 지금은 `--max-updates`·`--max-minutes`뿐이라 step 기준 종료를 추가해야 한다(작은 작업).
  - 학습 전후 200판 평가(050-R1 두 모드)는 4단계 평가 파이프라인 또는 `uranium-env run` plan으로 한다.
- **예상 시간**: 이번 처리량(3게임, 32 결정/초)이면 run 하나(1,000만 step)에 약 87시간, 6 run이면 약 22일이다.
  - 동시 게임을 늘려야 한다(bc250 두 대 추가, AMB-042-8). 지금 추론 배치가 2.9로 작아, 게임이 늘면 배치가 커져 GPU 효율이 오른다.
  - 같은 비율로 늘어난다고 가정하면 20게임 기준 약 200 결정/초, run당 약 14시간, 6 run 약 3.5일이다. 측정값이 아닌 추정이며, 동시 게임 수(020-R10 calibrate)와 추론 배치 파라미터(TBD(M6))를 측정한 뒤 다시 보고한다.

## 바꾼 spec 문구 (동작 불변, 사용자 결정 2026-10-18에 따라 직접 수정)
- 033-R8: z 테이블 명령 `uranium-replay z-table` → `uranium-train z-table`(`replay_z`가 uranium-train에 있음).
