# 042 — RL Training & League

- **Status**: Approved (2026-11-03 개정 3(사용자 결정, AMB-042-15, `reports/m6_rush_ladder.md`): 러시 커리큘럼 run(`rush-s1`), C1 러시 사다리 상대(`[curriculum]` `c1_ladder`·`ladder_pick`, `[[ladder]]`, 사다리 안 PFSP f_var), RL 학습 게임 붕괴 항복(042-R14, `[concede]`, `games.csv` `end_reason`, 042-AC5), 2차 `t1self-s2` 정지 기록. 2026-11-03 개정 2(사용자 결정, AMB-042-14, `reports/m6_t1self_stop.md`): C1 셀프 몫을 현재 Main·과거 Main 스냅샷(PFSP)·고정 상대 anchor로 나눔(`[curriculum]` `c1_snapshot`·`c1_anchor`, `[league] anchor`, `[snapshot] main_candidates`), 모델 상대 게임의 r_value 끔(`r_value_vs_model`, 033), 스냅샷은 그 에이전트 모델 파일만 공개, "T1·셀프 run" 1차 결과와 2차(`t1self-s2`) 구성. 2026-11-03 개정(사용자 결정, AMB-042-13): 042-AC1 결과 기록(6 run 모두 미달, 규칙대로 V-trace 채택), Main 출발 가중치 `--init`, `[curriculum]` `start_stage`·`auto_advance`(042-R5)·`r_value_stages`(033), 재개 시 커리큘럼 비율·`[[t1]]` 목록은 지금 설정 파일, "T1·셀프 run" 구성. 2026-10-31 개정: 큐 역압·판 벽시계 설정·learner 레인 그룹, AMB-042-11·12 사용자 결정. 2026-10-15, 2026-10-23 개정: PyTorch 전환(사용자 결정, AMB-000-4) — 학습 호스트 Linux ROCm·CUDA, 프로세스 구성(Python learner·추론 스레드 + Rust 에이전트·러너·리그 스레드), R1 CUDA↔ROCm 재개. 2026-10-16 개정: 042-R10 Ctrl+C 정상 종료, 2026-10-17 개정: 초기 설정값 확정(AMB-042-3, `config/rl.toml`), 첫 학습 호스트, AMB-042-2 M6 시작 재확인, behaviour log-prob 기록(AMB-042-4), 구현 구조·Aborted 회수(042-R3)·검증 정책·AC1 규모·게임 호스트·KL 계산(AMB-042-5~9, 사용자 결정), 2026-10-18 개정: 학습 중 리플레이 저장(042-R11)·agent step 상한 종료(042-R12), 사용자 결정. 2026-10-19 개정: 옵티마이저 clip = 041-R3 전역 norm(G-D4), R11 리플레이 보존과 060-R7 정합, 장비 표기 중립화(001-R10). 2026-10-20 개정: 주 장비 규칙 참조(060-R8, AMB-032-8). 2026-10-21 개정: 042-R13 win_open RL 학습(사용자 결정, AMB-031-6). 2026-10-31 개정: AC1 상대 종족 저그(사용자 결정). 2026-10-31 개정안(사용자 결정 ②·④, `reports/m6_rl_sample.md`): trajectory 큐 가득 참 → 역압(`queue_full`, `queue_wait_max_s`, AMB-042-11), R3 기다림 취소·close, R2·R12 학습한 step 누계·대기 시간 기록, RL 판 벽시계 제한 설정(`game_wall_timeout_s`, AMB-042-12), learner 레인 그룹 호스트 덮어쓰기(`URANIUM_RL_MICRO_LANES`)·같은 수식 최적화·`bench rl`(032-R19, 060-R3), `--logs` 상대 경로 버그 수정 기록)
- **Related**: [020](020-env-orchestration.md), [021](021-opponent-pool.md), [032](032-model-architecture.md), [033](033-reward.md), [050](050-evaluation.md), [060](060-infra.md)

## Purpose
BC 정책을 출발점으로 셀프플레이와 리그를 거쳐 승률을 극대화한다.

## 구조 (actor-learner)
```
[uranium-env actors] ×M 게임 ──obs──► [inference server (학습 호스트 GPU, batched)] ──action──► actors
         └── trajectory(T=64: obs, privileged, action, behaviour log-prob(헤드별·joint), rewards, core_state0) ──► replay queue ──► [learner (학습 호스트 GPU)]
learner ──► 새 가중치 (매 K 스텝) ──► inference server
```
- **behaviour 기록 (확정 2026-10-17, AMB-042-4)**: actor는 전체 logits(target_loc 하나만 16,384개)가 아니라 **선택한 액션의 헤드별·joint behaviour log-prob**(032-R15)을 기록한다. V-trace·UPGO의 중요도 비율에는 이것으로 충분하다. KL(π‖π_BC)은 learner가 같은 관측으로 BC 모델을 다시 forward해 헤드별 분포로 계산한다.

학습 호스트는 Linux ROCm 호스트 또는 CUDA 호스트다([060](060-infra.md), 2026-10-23 변경: 이전 Linux Vulkan 호스트). **첫 RL 실험(042-AC1)은 Windows CUDA 호스트에서 한다**(2026-10-17 사용자 결정). 그 뒤 RL의 추론·learner 장비는 주 장비 규칙(060-R8: CUDA 호스트가 유휴면 주, 바쁘면 Linux ROCm 호스트가 주, 2026-10-20 사용자 결정)을 따르고, 전환 구현은 M6 RL에서 한다. 게임은 그 호스트의 로컬 hostd로 돌리고, 필요하면 원격 Linux 게임 호스트를 hosts 파일에 더한다(020). 한 호스트의 GPU 하나로 추론과 학습을 모두 하므로 GPU 시간을 분할한다. 학습 중에는 추론 지연이 늘어나지만 lockstep이라 정확성에는 영향이 없다. 게임 호스트는 LAN으로 학습 호스트의 에이전트 서버에 접속한다(001-R6). 체크포인트·리그 스냅샷·승격 체크포인트는 공용 드라이브에 저장하며, 학습 시작·재개 절차와 run 잠금은 [061](061-shared-storage.md)을 따른다.

## 구현 구조 (2026-10-17, 3단계 계획 승인)
- **프로세스**: 학습 호스트에서 `python -m uranium rl` 하나가 learner, 추론(Main 1개 + 상대 스냅샷), 에이전트 서버(001-R6), 게임 배정, 리그 관리를 모두 맡는다(2026-10-23 개정: 이전 `uranium-train rl`). learner와 추론 스레드는 Python(PyTorch)이고, 에이전트 서버·게임 배정(`uranium-runner` 라이브러리, 020 동적 Job 공급자)·trajectory 수집·리그 관리는 `uranium._core`의 Rust 스레드가 GIL 없이 맡는다. 추론 스레드와 learner는 같은 GPU에 각자 모델 사본을 둔다. learner는 K update마다 가중치를 추론 스레드의 모델에 복사하고(같은 프로세스), 추론 응답에는 정책 버전이 붙는다(042-R4). learner의 trajectory 배치(펼치기·[B,T] 정렬)는 Rust가 만들어 numpy로 넘긴다. 추론은 별도 프로세스로 나누지 않는다(2026-10-24 사용자 결정, AMB-042-10: 두 학습 호스트 실측에서 GIL 대기는 사실상 0이고 지연은 GPU 공유에서 왔다. Linux ROCm 호스트는 분리하면 추론 B1이 12 → 36ms(GPU 시간 분할)이고 프로세스 하나에 약 0.43 GiB가 더 든다. 추론을 별도 GPU 스트림에 두는 구성은 그 호스트에서 학습·추론이 수 분씩 멈춰 쓰지 않는다. CUDA 호스트는 차이가 몇 ms. `reports/pytorch_opt.md` 2.25).
- **trajectory**: 학습 중인 에이전트의 세션에만 기록한다. 한 결정의 보상은 직전 결정 뒤 받은 프레임들의 보상 합이다(033). 세그먼트는 T=64 결정이고, 시작 결정 직전의 Core 상태(core_state0)를 추론 서버 슬롯에서 읽어 함께 둔다. 게임이 끝나면 마지막 세그먼트를 마스크로 채우고 bootstrap은 0이다(γ = 1, 종료). 마지막 결정 뒤 종료까지의 dense 보상(`GameEnd.final_frame`, 010-R12)과 r_win은 마지막 transition에 더한다(033-R7, E-8, 사용자 결정 2026-10-19).
- **손실**: 보상 항목별 value head마다 V-trace target과 advantage를 따로 구하고, 정책 gradient advantage는 033 가중치(감쇠 포함)로 더한다(033-R3). KL(π‖π_BC)은 고정된 BC 모델을 같은 입력으로 다시 forward해 헤드별 전체 분포로 계산하고 사용된 헤드만 더한다(032-R10). 정확한 분포 KL을 쓴다. GPU 메모리가 부족하면 근사로 바꾸지 않고 사용자에게 보고한다(2026-10-17 사용자 결정, AMB-042-9).
- **출발 정책 `--init` (2026-11-03 사용자 결정, AMB-042-13)**: `--bc` 체크포인트는 KL 참조 모델(AMB-042-9)과 Exploiter 리셋 대상이고, `--init`이 없으면 세 에이전트의 출발 정책도 된다(지금 동작). `--init <ckpt>`를 주면 Main의 출발 가중치를 그 체크포인트의 `agents/main`에서 읽는다. Main Exploiter·League Exploiter는 그대로 BC에서 시작한다.
  - run에 LATEST가 없을 때(새 run)만 적용한다. 재시작·재개는 `--init`과 상관없이 항상 그 run의 LATEST에서 이어 간다(042-R1, 061).
  - 모델 구조 설정(dropout 같은 가중치 없는 설정은 RL 값 0으로 맞춘 뒤)이 `--bc` 모델과 다르면 시작 오류다. KL은 같은 헤드 분포끼리 계산하기 때문이다(032-R10). feature schema 버전은 따로 같을 필요가 없고 빌드의 로드 호환(030-R5)을 따른다(모델 입력이 같다).
  - 가중치만 가져온다. 옵티마이저는 새로 만들고, step·update는 새 run id에서 0부터 센다(KL 계수도 0.1부터, 메인 임시 결정 2026-11-03).
  - 출발 출처(`--init` 체크포인트, 없으면 `--bc`)를 체크포인트의 에이전트별 `meta.json` `train.start`에 기록한다(준 경로와 절대 경로, `--init`이면 그 체크포인트의 update·step). 재개해도 유지한다.
- **스냅샷 상대 ID**: `uranium-<agent>@<run>/<update>` (agent = `main` | `main-exploiter` | `league-exploiter`, 021-R3 형식).
- **로그 (042-R2)**: CSV 두 개. `train.csv`(update별 손실 항목, KL, 헤드별 엔트로피, 정책 지연, 버린·회수한 trajectory 수, 학습한 agent step 누계 `trained_steps`, 직전 행 이후 게임들이 큐 자리를 기다린 시간 합 `queue_wait_s`(2026-10-31), 처리량, 추론 p50·p95)와 `games.csv`(게임별 결과, 보상 항목 합, 상대 ID, 종족, 맵, 저장한 리플레이 경로·사유(042-R11), 끝난 사유 `end_reason`(042-R14, 2026-11-03)).
  - **`--logs` 경로(2026-10-31 버그 수정)**: `--logs`(기본 `runs/rl`)가 상대 경로면 시작할 때 현재 작업 폴더 기준 절대 경로로 바꾼 뒤 로그와 추론 서버 모델 폴더(`<logs>/infer/<이름>`)를 만든다. 추론 서버(Rust Broker)는 상대 체크포인트 경로를 `URANIUM_SHARED_DIR` 기준으로 찾으므로(020-R11), 상대 경로를 그대로 넘기면 모델 파일을 찾지 못해 시작에 실패했다(`reports/m6_rl_sample.md` 발견 1). 고치기 전에는 절대 경로를 준다.
- **learner 레인 그룹 (2026-10-31 사용자 결정 ④, 032-R19)**: RL learner도 BC(041-R3, AMB-041-7)처럼 배치의 레인을 그룹으로 나눠 계산하고, 모든 그룹의 gradient를 합친 뒤 전역 clip과 옵티마이저를 한 번 적용한다. 그룹 크기는 호스트 `.env` 키 `URANIUM_RL_MICRO_LANES`(001-R9, 레인 수, 0 = 전체, 없거나 비면 `config/rl.toml` `micro_lanes`)로 정한다.
  - Windows CUDA 호스트: 2 또는 4 중 메모리 측정으로 고른다(TBD(M6)). 정확 KL의 BC 재forward(AMB-042-9)를 포함한 최대 GPU 할당이 GPU 메모리 안에 드는 큰 쪽이다.
  - Linux ROCm 호스트: 1 그대로다. 060-R3의 채택 기준(최악 모양, 레인 그룹 1)도 RL에서 그대로다.
  - 수식·배치 내용·손실·하이퍼파라미터는 같고, gradient 합산 순서와 dropout 난수열 배치만 달라진다(032-R19 ② 허용 오차). 결과를 바꾸지 않는 실행 설정이므로 재개할 때 호스트마다 달라도 된다.
  - 동등성 검사: 같은 체크포인트와 고정한 세그먼트 묶음(기록 또는 합성)으로 그룹 1과 덮어쓴 그룹 크기를 각각 30 update 돌린다. update별 손실 항목·KL·전역 gradient norm이 허용 오차 안임을 리포트에 남긴다. 작은 모양의 CPU 단위 테스트도 둔다.
- **learner 최적화 (2026-10-31 사용자 결정 ④, 032-R19)**: 같은 수식을 더 빠르게 계산하는 변경만 쓴다. 구조·손실·하이퍼파라미터·정확 KL(AMB-042-9)은 바꾸지 않는다. 허용하는 예:
  - 배치를 고정(pinned) 메모리에 준비해 GPU로 옮기기, CUDA에서 bf16으로 준비하기(032-R8 정밀도 범위 안)
  - 통계의 호스트 동기화를 update당 한 번으로 모으기
  - 바깥 LSTM 위상별 구간 인덱스(032 §2.7)를 호스트에서 미리 계산하기
  - 리턴 타깃(V-trace·TD(λ)·UPGO) 계산의 벡터화
  - 다음 배치 준비를 현재 update와 겹치기
  - 측정은 새 벤치 `python -m uranium bench rl`로 한다(체크포인트와 고정 세그먼트 묶음으로 update 시간·단계별 시간·GPU 메모리). 채택 전에 032-R19 ⑤처럼 동등성(위 30 update 비교)과 속도·메모리를 리포트에 남긴다.
- **BC 본학습 전 검증 (2026-10-17 사용자 결정, AMB-042-5)**: 이미 build된 shard 표본으로 짧은 BC(몇 시간)를 돌린 체크포인트로 042-AC1 형식을 미리 연습한다. 본 BC 체크포인트가 나오면 그것으로 바꾼다.
- **게임 호스트 (2026-10-17 사용자 결정, AMB-042-8)**: 첫 RL(스모크·042-AC1)은 학습 호스트의 로컬 hostd만으로 시작한다. 리플레이 build가 끝나면 원격 Linux 게임 호스트 두 대를 hosts 파일에 더한다(hostd를 master로 다시 빌드, 020-R11 플레이어 이름).

## 알고리즘
- **V-trace**(IMPALA) off-policy 보정: actor 정책 지연을 허용한다.
- **UPGO** policy gradient를 보조로 쓴다.
- **KL(π ‖ π_BC)** 정규화: 계수 초기값 0.1, 점진 감쇠, 하한 0.01.
- 엔트로피 보너스를 헤드별로 둔다.
- 대안: PPO(clip 0.2)를 같은 인터페이스로 구현해 비교한다. 초기 소규모 실험에서 더 안정적인 쪽을 채택한다.

## 리그
| 에이전트 | 상대 샘플링 | 목적 |
|---|---|---|
| Main | PFSP(전체 스냅샷 + T0/T1) 50%, 셀프 35%, T1 15% | 최종 정책 |
| Main Exploiter | 현재 Main만 | Main 약점 발견 |
| League Exploiter | PFSP 전체 | 전략 다양성 |

- 스냅샷 주기: 매 N 업데이트 또는 승률 수렴 시.
- 상대 종족 분포: T/P/Z 균등. 승률이 가장 낮은 종족을 가중 샘플링한다. 여러 종족을 지원하는 T0·T1 봇은 일부 게임에서 Random 종족을 고른다(AMB-000-3, 021-R7). 비율은 설정 파일로 관리한다.
- **종족 구성의 한계**: 리그의 학습 에이전트(Main, Exploiter, 스냅샷)는 모두 테란이므로 셀프플레이로는 TvT만 연습된다. TvP·TvZ 상대는 T0·T1 봇뿐이다(아래 AMB-042-2).
- 맵: 투혼과 서킷브레이커 균등, 시작 위치는 랜덤.

## 초기 설정값 (2026-10-17, AMB-042-3)
사용자 위임으로 Claude가 정했다. 근거는 IMPALA(Espeholt et al. 2018)와 AlphaStar(Vinyals et al. 2019)의 설정을 이 프로젝트 규모(학습 호스트 1대, GPU 하나를 추론과 공유)에 맞춰 줄인 것이다. **모든 값은 042-R9 튜닝 대상**이며 `config/rl.toml`로 관리하고 체크포인트에 기록한다. 042-AC1 소규모 실험 결과로 조정을 제안할 수 있고, 조정은 사용자 확인 후 반영한다.

용어: **agent step** = `act()` 1회(한 게임의 한 결정). **update** = learner 파라미터 갱신 1회.

### 학습 배치·옵티마이저
| 항목 | 값 | 근거 |
|---|---|---|
| 배치 | 레인 32 × T64 = 2,048 agent step/update. `micro_lanes`로 나눠 gradient를 합친다(041과 같은 방식). 호스트별 그룹 크기는 `URANIUM_RL_MICRO_LANES`로 덮어쓴다(위 "learner 레인 그룹", 2026-10-31) | T=64는 042-R8. 레인 수는 BC(`config/bc.toml`)와 같게 해 메모리 특성을 재사용 |
| 옵티마이저 | Adam, lr 3e-5, β1 0, β2 0.99, ε 1e-5, weight decay 0, gradient-norm clip 1.0(전역 L2 norm, 041-R3과 같은 구현, 2026-10-19 사용자 결정 G-D4, AMB-041-5) | AlphaStar RL 단계 설정(사전학습 정책 미세조정). BC lr(3e-4)의 1/10 |
| 할인 γ | 1.0 (에피소드는 020 60분 상한으로 유한) | AlphaStar는 할인 없이 λ로 분산 조절 |
| value 목표 | TD(λ), λ = 0.8 | AlphaStar |
| V-trace 절단 | ρ̄ = 1.0, c̄ = 1.0 | IMPALA 기본값 |
| 손실 가중치 | V-trace PG 1.0, UPGO 1.0, value 0.5(헤드별 동일), KL·엔트로피는 아래 | AlphaStar(PG·UPGO 동일 가중), value 0.5는 BC `value.win`과 같음 |
| advantage 결합 | 보상 항목별 value head(033-R3)로 advantage를 따로 구하고 [033](033-reward.md) 가중치로 더한다 | 033-R3 |

### 정책 지연·가중치 배포·큐
| 항목 | 값 | 근거 |
|---|---|---|
| 가중치 배포 주기 K | 4 update마다 추론 서버에 반영 | 같은 GPU라 복사 비용이 작다. IMPALA는 매 update 배포 |
| 정책 지연 임계값 (042-R4) | behaviour 버전이 learner보다 **32 update 넘게** 뒤처진 trajectory는 버린다. 버린 비율을 로그에 남긴다 | V-trace는 지연을 보정하지만 지연이 크면 ρ 절단으로 학습 신호가 줄어든다. K의 8배 |
| trajectory 큐 | 에이전트마다 최대 `queue_batches` 배치(기본 4 = 128 trajectory). `queue_batches`는 1 이상이어야 하고 0이면 설정 로드 오류다. **가득 차면 역압(2026-10-31 사용자 결정 ②)**: 세그먼트를 넣으려던 게임이 자리가 날 때까지 기다린다. 중간 세그먼트면 그 게임의 결정 응답이 멈추고(lockstep이라 게임도 멈춘다), 마지막 세그먼트면 그 판의 자리에 새 판을 배정하지 않는다. 기다림은 그 게임에만 걸리고 다른 게임의 진행·결과 처리는 막지 않는다. 기다리는 게임은 들어온 순서(FIFO)로 깨운다. 한 번 기다리는 시간의 상한은 `queue_wait_max_s`, 120초다(2026-10-31 사용자 결정, AMB-042-11). 상한은 OpenBW 드롭 시간(011-R18, 기본 600초)보다 충분히 작아야 한다(기다리는 동안 그 게임의 OpenBW 클라이언트도 멈춰 있다). 상한을 넘으면 큐에서 가장 오래된 세그먼트를 버리고(`dropped_full`로 셈) 기다리던 세그먼트를 넣는다. 설정은 `[actors] queue_full = "wait"`(기본) \| `"drop_oldest"`(이전 동작: 기다리지 않고 가장 오래된 것부터 버림). 다른 값은 설정 로드 오류다 | 큐 대기로 생기는 지연을 임계값 안으로 묶는다. 역압 근거: 학습 중 추론 지연과 마찬가지로 lockstep이라 기다려도 정확성에는 영향이 없다(위 학습 호스트 문단). 버리면 게임 끝 세그먼트의 승패 보상(r_win)을 잃는다. 2026-10-31 표본에서 learner가 병목인 두 구성은 만든 세그먼트의 45~75%를 버렸다(`reports/m6_rl_sample.md`) |
| RL 판 벽시계 제한 | `[actors] game_wall_timeout_s`, 7,200초(2026-10-31 사용자 결정, AMB-042-12). 넘으면 hostd가 판을 `Timeout`으로 끝내고 Aborted처럼 학습에서 뺀다(042-R3, 020-R3). 1 이상이어야 한다 | 지금까지는 코드에 3,600초로 고정돼 있었다. 역압으로 기다린 시간도 벽시계에 들어간다. BC 평가 plan은 느린 추론에서 60분 무승부 판이 잘리지 않게 7,200초를 쓴다(최종 비교 300판 최대 2,467초). 2026-10-31 Linux ROCm 표본(결정 지연 p50 71 ms)에서는 멈출 때 진행 중이던 판이 2,845초였다(`reports/m6_pre_rl_review.md` 5-1절 C1, `reports/m6_rl_sample.md`). 긴 판(무승부)이 잘리면 장기전을 덜 배우는 편향이 생긴다 |

### 엔트로피·KL
| 항목 | 값 | 근거 |
|---|---|---|
| 헤드별 엔트로피 계수 | action_type 3e-3, delay 1e-3, queued 1e-3, selected 1e-3, target_unit 1e-3, target_loc 1e-3. 감쇠 없음 | KL(π‖π_BC)이 붕괴를 막으므로 IMPALA(1e-2)보다 작게 둔다. 행동 종류 다양성이 가장 중요해 action_type만 크게. 032-R10대로 사용된 헤드만 합산 |
| KL 계수 스케줄 | `max(0.01, 0.1 × 0.5^(update / 20,000))` (반감기 20,000 update ≈ 4,100만 agent step) | 초기 0.1·하한 0.01은 기존 결정. 반감기는 C0~C1 동안 BC 근처를 유지하도록 정함 |
| Exploiter의 KL | Main과 같은 스케줄. Exploiter를 BC로 리셋하면 스케줄도 처음부터 | AlphaStar는 Exploiter에도 KL을 적용 |

### 리그·상대 선택
| 항목 | 값 | 근거 |
|---|---|---|
| learner 시분할 (042-R7) | Main 50%, Main Exploiter 25%, League Exploiter 25%. **C2부터 적용**하고, C0·C1에서는 Main만 학습한다(100%) | 042-R7 하한. AlphaStar(Main:ME:LE = 1:1:2)보다 Main 비중을 크게 |
| Main 상대 (C2) | PFSP 50%(전체 스냅샷 + T0/T1), 셀프 35%, T1 15% | 기존 리그 표 |
| PFSP 가중 | Main·League Exploiter: `f_hard(x) = (1 − x)²`, x = 상대에 대한 자기 승률 | AlphaStar |
| Main Exploiter | 현재 Main과 대전. 최근 승률이 20% 미만이면 게임의 50%를 Main 스냅샷 상대로 `f_var(x) = x(1 − x)` 가중 | AlphaStar(너무 어려운 상대에서 학습 신호 확보) |
| 승률 표 | 상대별 지수 이동 평균(게임마다 감쇠 0.99), 처음 보는 상대는 0.5 | 오래된 결과의 영향을 줄인다 |
| 종족 샘플링 | `p_r = 0.5 × 1/3 + 0.5 × (1 − w_r) / Σ(1 − w)`, w_r = 종족 r 상대 최근 승률. 모든 종족 최소 1/6 | 균등 + 약한 종족 가중(기존 결정)을 절반씩 |
| Random 종족 비율 (021-R7) | 상대가 T0·T1 봇이고 세 종족을 모두 지원할 때 20%. 두 종족만 지원하면(`[[t1]] races`가 둘) Random을 주지 않고, 종족 샘플링으로 정한 종족(목록 안)으로 둔다(2026-11-03, 메인 임시 결정, AMB-042-14) | 종족을 하나만 하는 T1 봇은 적용하지 않는다. 두 종족 봇에 Random을 주면 목록에 없는 종족(예: 2차 run의 Steamhammer 프로토스)이 나올 수 있다 |
| C1 모델 상대 (2026-11-03 사용자 결정, AMB-042-14) | C1 셀프 몫을 셋으로 나눈다: 현재 Main(`c1_self`), 과거 Main 스냅샷(`c1_snapshot`), 고정 상대 anchor(`c1_anchor`). 키는 아래 "단계 설정" | 셀프 상대가 현재 Main뿐이면 양쪽이 함께 수동적으로 바뀌어도 벌받지 않는다. 1차 T1·셀프 run은 자기끼리 9/10 무승부, 출발 체크포인트에 0-10으로 무너졌다(`reports/m6_t1self_stop.md`) |
| C1 스냅샷 상대 선택 | 후보 = 공개에 성공한 Main 스냅샷 중 최근 `[snapshot] main_candidates`개(0 = 전체). 후보 안에서 PFSP `f_hard(x) = (1 − x)²`, x = 그 스냅샷에 대한 Main 승률(위 승률 표, 처음 보는 스냅샷은 0.5). 후보에서 빠진 스냅샷은 공용 드라이브에 그대로 둔다(지우지 않음) | AlphaStar Main의 PFSP를 C1 셀프 몫에 일부 도입. 후보 수를 묶어 추론 서버 수(GPU 메모리)를 제한 |
| anchor | `[league] anchor`: 고정 상대 체크포인트(체크포인트 폴더 `ckpt-<step>` 또는 그 `agents/main`, 상대 경로는 `URANIUM_SHARED_DIR` 기준, 020-R11과 같음). 승률 표 키 `anchor@<체크포인트 이름>`(예: `anchor@ckpt-3133569`). `c1_anchor > 0`인데 `anchor`가 비었거나 시작할 때 그 경로에 모델 파일이 없으면 설정·시작 오류다(메인 임시 결정) | 출발점에게 지는 정책을 바로 벌한다 |
| 스냅샷이 아직 없을 때 | `c1_snapshot` 몫은 anchor로 간다. anchor도 없으면(`c1_anchor = 0`이고 `anchor` 없음) 현재 Main으로 간다 | 사용자 결정(anchor로). anchor가 없는 경우는 메인 임시 결정 |
| 모델 상대 실패 (메인 임시 결정) | 판을 배정할 때 뽑은 스냅샷·anchor 상대의 추론 서버를 쓸 수 없으면(띄우기 실패, 모델 파일 읽기 실패 등) 그 판은 현재 Main 셀프로 배정한다. 게임 자리를 멈추지 않고, 바꾼 사실을 로그에 남긴다. 판 도중 실패는 지금처럼 Aborted로 학습에서 뺀다(042-R3) | 상대 하나의 실패로 학습이 멈추지 않게 |
| 모델 상대 게임과 커리큘럼 창 | 현재 Main·스냅샷·anchor 게임은 커리큘럼 전환 창(종족별 T0·T1 최근 300판)에 넣지 않는다. 승률 표에는 상대별로 기록한다(메인 임시 결정) | 전환 조건은 봇 상대 승률 기준 |
| C1 러시 사다리 (2026-11-03 사용자 결정, AMB-042-15) | `[curriculum] c1_ladder`: C1에서 러시 사다리 상대와 두는 몫. 사다리는 RL 설정의 `[[ladder]]` 목록이다. 항목은 `kind = "t0"`와 `build`(T0 빌드 이름, 예: `"zerg_rush"`) 또는 `kind = "t1"`와 `name`(T1 봇 폴더 이름, 021 설정 변형 포함), 그리고 `race`(그 상대가 두는 종족, `"Terran"`·`"Protoss"`·`"Zerg"`)다. 종족이 항목마다 고정이라 종족 샘플링과 Random(021-R7)을 쓰지 않는다. 승률 표 키와 상대 클래스는 기존 규칙 그대로다: T0는 `t0-<race>-<빌드 태그>`(예: `t0-zerg-rush`, 021-R3)·T0, T1은 `<name>-<race>`(예: `steamhammeroverpool-zerg`)·T1. 그래서 r_value(봇 판, 033)와 커리큘럼 창도 그 클래스 규칙을 따른다. 사다리는 C1에서만 쓰고 C2 PFSP 풀에는 넣지 않는다 | 사다리는 종족·빌드가 고정인 상대 묶음이고 뽑는 방식(f_var)도 다르다. `[[t1]]`(종족 샘플링 → 그 종족 봇 균등, Random)에 거르기 설정을 붙이면 같은 목록이 run마다 다른 뜻이 되므로 따로 둔다. 기존 run 설정과 `[[t1]]` 선택 코드는 그대로다(메인 임시 결정) |
| 사다리 안 상대 선택 | `[curriculum] ladder_pick = "f_var"`(기본이자 지금 유일한 값, 다른 값은 설정 로드 오류): PFSP `f_var(x) = x(1 − x)`, x = 그 항목에 대한 Main 승률 표 값(승 1·무 0.5·패 0의 지수 이동 평균, 위 "승률 표", 처음 보는 상대는 0.5). 승률이 50%에 가까운 상대가 가장 자주 뽑힌다. 가중치 합이 0이면 균등 | 사용자 결정(2026-11-03, AMB-042-15). 승률 20~70% 대역 상대의 학습 신호가 가장 강하다는 자문 권고(`reports/m6_rush_ladder.md` 1절) |
| 사다리 설정 검사 | `c1_ladder > 0`인데 `[[ladder]]`가 비었을 때, `kind`가 `t0`·`t1`이 아닐 때, `t0` 항목의 빌드가 없는 이름이거나 종족과 맞지 않을 때(예: `zerg_rush`인데 `race`가 저그가 아님), `race`가 Random일 때, 같은 승률 표 키가 두 번 나올 때는 설정 로드 오류다. 봇은 `[[t1]]`처럼 모든 게임 호스트에 설치돼 있어야 한다(리그는 확인하지 않는다) | 메인 임시 결정 |
| 맵·시작 위치 | 투혼·서킷브레이커 균등, 시작 위치 랜덤 | 기존 결정 |

### 스냅샷·리셋 (스냅샷 주기 N)
| 에이전트 | 스냅샷 조건 | 리셋 |
|---|---|---|
| Main | 2,000 update마다(약 410만 agent step) | 없음 |
| Main Exploiter | 현재 Main 상대 승률 ≥ 70%(최근 200판) 또는 4,000 update | 스냅샷 후 BC 정책으로 리셋 |
| League Exploiter | PFSP 상대 전체 승률 ≥ 70%(최근 200판) 또는 4,000 update | 스냅샷 후 25% 확률로 BC 리셋 |

근거: AlphaStar의 조건(승률 70%, 고정 주기, LE 리셋 확률 25%)을 이 규모 update 수로 바꿨다.

- Main 스냅샷 주기는 `[snapshot] main_every_updates`(기본 2,000)다. 2차 T1·셀프 run은 250이다(2026-11-03 사용자 결정, AMB-042-14).
- **스냅샷 공개 (2026-11-03, 메인 임시 결정, AMB-042-14)**: 스냅샷 때 체크포인트를 저장·push한 뒤, 그 에이전트의 모델 파일(`agents/<agent>`)만 `checkpoints/<run>/league/<스냅샷 이름>/agents/<agent>`로 복사한다(061 구조). 이전에는 체크포인트 전체(옵티마이저·다른 에이전트 포함)를 복사했다. 복사에 실패한 스냅샷은 공개되지 않은 것으로 보고 상대 후보에 넣지 않는다.
- **스냅샷 C1 후보 수**: `[snapshot] main_candidates`(기본 4, 0 = 전체). C1 스냅샷 몫의 후보를 공개된 Main 스냅샷 중 최근 이 개수로 묶는다(위 "C1 스냅샷 상대 선택"). C2 PFSP는 지금처럼 전체 스냅샷이다.
- **스냅샷 추론 서버**: 어느 상대 선택에도 쓰이지 않는 스냅샷(예: C1에서 후보에서 빠진 것)의 추론 서버는 그 서버를 쓰는 진행 중 판이 끝나면 내리고 GPU 메모리를 돌려준다. 스냅샷 파일은 지우지 않는다(메인 임시 결정).

리셋하면 그 에이전트 큐에 남은 세그먼트와 자리를 기다리던 세그먼트는 리셋 전 정책의 것이므로 버리고(기다리던 게임은 넣지 않고 풀어 준다), 그 뒤 도착하는 리셋 전 정책의 세그먼트(behaviour 버전이 learner 버전보다 큰 것)도 버린다. 모두 `dropped_lag`로 센다(042-R4, 2026-10-31).

### 커리큘럼 (042-R5)
| 단계 | 상대 구성 | 다음 단계 조건 |
|---|---|---|
| C0 | T0 봇 80%(종족·빌드 균등), 셀프 20%. r_value 사용(033, `r_value_stages` 기본값) | 종족마다 T0 상대 최근 300판 승률 ≥ 70% |
| C1 | T0 30%, T1 40%, 셀프 30%. 셀프는 현재 Main·과거 Main 스냅샷·anchor로 나눌 수 있다(아래 단계 설정, 기본은 현재 Main만). 러시 사다리 몫(`c1_ladder`, 기본 0)을 더할 수 있다(2026-11-03, AMB-042-15) | 종족마다 T1 상대 최근 300판 승률 ≥ 50% |
| C2 | 풀 리그(위 표) | 없음(중단은 사용자 결정, AMB-042-1) |
- 시간 기준 강제 전환은 두지 않는다. 한 단계에서 오래 정체하면 M6 중간 점검에서 사용자에게 보고한다.
- pseudo-reward(r_build·r_units, 033)는 C0·C1에서 고정하고, C2 진입 후 Main 20,000 update에 걸쳐 0으로 선형 감쇠한다(033의 "리그 진행에 따라 감쇠"의 초기값, 033-R4).
- **단계 설정 (2026-11-03 사용자 결정, AMB-042-13)**: RL 설정 파일 `[curriculum]`. 기본값은 지금 동작과 같다.

  | 키 | 기본 | 의미 |
  |---|---|---|
  | `start_stage` | `"C0"` | 새 리그(run에 LATEST 없음)의 시작 단계. `"C0"` \| `"C1"`, 다른 값은 설정 로드 오류다. 재개하면 리그 상태의 단계를 그대로 쓴다 |
  | `auto_advance` | `true` | `false`면 위 승률 조건을 채워도 단계를 바꾸지 않는다(단계 변경은 사용자 결정). 전환 창(종족별 최근 300판)은 그대로 기록한다. 042-R5(SHOULD)에서 벗어나는 사유: 상대 비중을 추이를 보고 사용자가 정하기로 함 |
  | `r_value_stages` | `["C0"]` | r_value(033)를 쓰는 단계. 다른 단계에서는 가중치 0 |
  | `c1_self` | `0.3` | C1에서 현재 Main과 두는 몫(지금 뜻 그대로) |
  | `c1_snapshot` | `0` | C1에서 과거 Main 스냅샷과 두는 몫(위 "C1 스냅샷 상대 선택", 2026-11-03, AMB-042-14) |
  | `c1_anchor` | `0` | C1에서 anchor(`[league] anchor`)와 두는 몫(2026-11-03, AMB-042-14) |
  | `c1_ladder` | `0` | C1에서 러시 사다리(`[[ladder]]`)와 두는 몫(위 "C1 러시 사다리", 2026-11-03, AMB-042-15) |
  | `ladder_pick` | `"f_var"` | 사다리 안 상대 선택 방식. 지금은 `"f_var"`만 있다(위 "사다리 안 상대 선택") |
  | `r_value_vs_model` | `true` | `false`면 모델 상대(현재 Main·스냅샷·anchor) 게임에서 r_value의 정책 가중치를 0으로 하고, T0·T1 봇 게임만 `r_value_stages`를 따른다(033, 2026-11-03 사용자 결정, AMB-042-14). r_value value head는 모든 게임에서 자기 목표를 그대로 학습한다 |

  C1 상대는 `c1_t0`·`c1_t1`·`c1_ladder`·`c1_self`·`c1_snapshot`·`c1_anchor`의 합에 대한 비율로 뽑는다(지금처럼 합으로 나눈다). 새 키(`c1_snapshot`·`c1_anchor`, 2026-11-03 개정 3의 `c1_ladder`)가 0이면 지금 동작과 같다. 사용자 결정의 "셀프 쪽 현재 Main 15%"는 기존 `c1_self`(현재 Main 몫)로 쓴다(메인 임시 결정: 따로 `c1_self_current` 키를 두지 않음).
- **r_value의 게임별 적용 (2026-11-03 사용자 결정, AMB-042-14)**: r_value 정책 가중치는 판마다 정한다. 그 판의 단계가 `r_value_stages`에 있고, 상대가 봇이거나 `r_value_vs_model = true`면 033 가중치(0.05), 아니면 0이다. 한 update 배치에 두 종류 판의 세그먼트가 섞이므로 가중치는 세그먼트(판) 단위로 적용한다. V-trace PG와 UPGO 모두 같은 가중치를 쓴다. value 손실(033-R3 value↔r_value)은 가중치와 상관없이 모든 판에서 학습한다.
- **재개 시 설정 (2026-11-03)**: 단계·승률 표·전환 창·스냅샷은 리그 상태(체크포인트)에서 이어 가고, 상대 비율(`c0_*`·`c1_*`, C2 `main_*`)·`auto_advance`·`r_value_stages`·`r_value_vs_model`·`[league] anchor`·`[snapshot] main_candidates`와 T1 목록(`[[t1]]`)은 지금 설정 파일을 따른다(새 키는 2026-11-03 개정 2, 메인 임시 결정). `c1_ladder`·`ladder_pick`·러시 사다리 목록(`[[ladder]]`)·`[concede]`도 지금 설정 파일을 따른다(2026-11-03 개정 3, 메인 임시 결정). 사다리 항목의 승률은 승률 표 키로 이어 간다. 이전에는 T1 목록이 새 리그를 만들 때 리그 상태에 고정됐다. 상대 비중은 정상 종료(042-R10) → 설정 수정 → 재시작으로 바꾼다.

### 붕괴 항복 (042-R14, 2026-11-03 사용자 결정, AMB-042-15)
RL 학습 게임에서 우리 쪽이 무너지면 모델 정책이 스스로 항복해 그 판을 패배로 끝낸다. 무너진 뒤 상대가 마무리하지 못해(T0 기억·탐색 결함, UAlbertaBot 러시 뒤 정지) 60분 무승부(r_win 0)로 끌리는 판을 패배(−1)로 바로잡기 위해서다. 러시 상대 기준 측정에서 무승부 40판 중 34판이 그런 판이었다(`reports/m6_rush_ladder.md` 6절).

| 항목 | 값 |
|---|---|
| 설정 | RL 설정 `[concede]`(메인 임시 결정): `enabled`(기본 `false` = 지금 동작), `max_supply`(기본 4), `hold_s`(기본 120), `after_s`(기본 180). `max_supply`는 0~200, `hold_s`는 1 이상, `after_s`는 0 이상이어야 하고 벗어나면 설정 로드 오류다 |
| 무너짐 | 결정마다 관측의 자기 `PlayerState.supply_used`(010, BWAPI `supplyUsed()`의 테란 값, 생산 중인 유닛 포함)를 본다. 이 값은 게임 화면 표시의 2배 단위(SCV 1기 = 2)이므로 `supply_used ≤ 2 × max_supply`면 무너짐이다. 기본값은 `supply_used ≤ 8`, 곧 화면 표시 4 이하(SCV 4기 상당. 마린도 1기 = 화면 1)다 |
| 연속 조건 | 게임 프레임 ≥ `after_s` × 24(기본 4,320 = 3분, 020·031-R12와 같은 24fps 환산)인 결정부터 센다. 구간은 그 뒤 무너짐인 첫 결정의 프레임에서 시작하고, 무너짐이 아닌 결정이 하나라도 나오면 끊긴다. 이번 결정 프레임 − 구간 첫 프레임 ≥ `hold_s` × 24(기본 2,880 = 120초)인 첫 결정에서 항복한다. 기본값에서 가장 이른 항복은 5:00이다(3분 전에 무너졌어도 구간은 3분부터 센다, 메인 임시 결정) |
| 입력 | 자기 `PlayerState`와 프레임만 쓴다. 적 정보·privileged 입력을 쓰지 않으므로 맵핵 문제가 없다(010-R5, 033-R5) |
| 실행 | 에이전트 서버가 관측을 받으면 추론 전에 판정한다. 성립하면 그 결정은 추론하지 않고 `Act { cmds: [], surrender: Some("gg") }`(010-R13, 채팅은 020-R14 기본과 같음)로 답하며, 그 세션은 더 결정하지 않는다. 브리지는 011-R21대로 채팅 뒤 게임을 떠난다. 떠난 쪽 `GameEnd`는 winner = 상대, reason = `Surrender`이고, 남은 쪽(봇·모델)은 011-R17대로 `Surrender`와 자기 승리를 보고하므로 020-R13에 따라 일반 패배로 판정한다(Aborted·상호 퇴장 아님, 020-R18). 프로토콜·브리지·hostd는 바꾸지 않는다. 판정 코드는 031-R12처럼 `uranium-env`에 둔다(메인 임시 결정) |
| 적용 대상 | RL 학습 게임(042 동적 공급자, 020-R12)의 모든 모델 정책 세션: 학습 중인 Main, 그리고 모델 상대(현재 Main·과거 스냅샷·anchor) 쪽 세션(모델 상대 쪽 적용은 메인 임시 결정). 봇(T0·T1) 세션에는 적용하지 않는다(T0는 바꾸지 않는다). 한 판에서 한 번만: 먼저 성립한 세션이 항복하면 같은 판의 다른 모델 세션은 더 판정하지 않는다(메인 임시 결정: 두 세션이 같이 떠나면 서로 상대 승리를 보고해 020-R13 판정이 먼저 항복한 쪽의 승리로 뒤집힐 수 있다) |
| 평가 게임 | 적용하지 않는다. plan으로 도는 게임(020-R14·R15, 050, 아래 run 점검 포함)은 지금 규칙(60분 무승부, 031-R12 항복 모드)대로다(메인 임시 결정: 출발 체크포인트 기준 측정 `reports/m6_rush_ladder.md`와 같은 조건으로 비교하기 위해. 무너짐은 점검의 막음 비율로 따로 본다) |
| 학습 | 항복한 쪽은 r_win −1, 상대 모델이 항복해 이긴 학습 중인 쪽은 +1(033). 항복 결정은 agent step이 아니다(추론·trajectory·step 수에 들지 않음). r_win과 마지막 결정 뒤 퇴장까지의 프레임 보상(`GameEnd.final_frame`, 011-R21의 퇴장 직전 프레임)은 그 세션의 마지막 실제 결정에 더한다(033-R7). 승률 표·커리큘럼 창에는 일반 승패로 들어간다(메인 임시 결정) |
| 기록 | `games.csv` 끝에 열 `end_reason`을 더한다(메인 임시 결정): 판정한 결과의 `GameEnd.reason`(`Elimination`·`Surrender`·`TimeLimitDraw` 등, 020-R13으로 쓴 세션의 것), 이 규칙으로 끝난 판은 `Concede`, 결과가 없는 판(Aborted·Timeout)은 빈 값. 항복한 쪽은 `outcome`으로 안다(패 = 학습 중인 쪽이 항복, 승 = 상대 모델이 항복). 학습 로그에는 판 id·항복한 세션(학습 쪽·상대 쪽)·프레임을 한 줄 남긴다 |
| 처리량 | 무너진 판이 60분까지 가지 않고 끝나 게임 자리가 빨리 빈다(기본값에서 무너진 뒤 늦어도 120초, 3분 전에 무너졌으면 5:00) |

### T1·셀프 run (2026-11-03 사용자 결정, AMB-042-13·14)
042-AC1 뒤 다음 RL run이다. AC1은 6 run 모두 기준 미달이었고(아래 AC1 결과), AC1 무승부 조사에서 T0 봇은 무너진 상대를 마무리하지 못하는 등 상대로 부족했다(`reports/m6_ac1_draws.md`). T1 저그 3종 각 10판에서 BC는 1/30승, AC1 V-trace s1은 3/30승이었다(`<shared>/eval/t1zerg-20261103/`).

**1차 `t1self-s1` (실패, 2026-11-03 사용자 결정으로 정지)**

| 항목 | 값 |
|---|---|
| 설정 파일 | `config/rl_t1self.toml` |
| 알고리즘 | V-trace(042-AC1 채택) |
| 출발 정책 | `--init ac1-vtrace-s1/ckpt-3133569`(`agents/main`). KL 참조·Exploiter 리셋 대상은 `--bc bc-s2e2-pt/ckpt-11387` 그대로 |
| step·update | 새 run id로 0부터. KL 계수 0.1부터, 옵티마이저 새로(메인 임시 결정) |
| 단계 | `start_stage = "C1"`, `auto_advance = false`. C1이라 Main만 학습한다(042-R7) |
| 상대 비율 | `c1_t0 = 0`, `c1_t1 = 0.5`, `c1_self = 0.5`. T0 봇은 고치지 않고 이 run의 상대에서 뺀다(사용자 결정) |
| T1 풀 | McRave 제외(메인 임시 결정: lockstep에서 다른 T1 봇보다 10~40배 느려 처리량이 떨어짐). 종족마다 3개로 021-R4 충족: T = UAlbertaBot·Steamhammer·Iron, P = UAlbertaBot·Steamhammer·Stardust, Z = UAlbertaBot·Steamhammer·ZZZKBot |
| 종족·Random | T/P/Z 모두, 위 종족 샘플링 그대로. 여러 종족 봇 Random 20%(021-R7) |
| 맵 | 투혼·서킷브레이커 균등(`--ac1-race` 없음) |
| 보상 | `r_value_stages = ["C0", "C1"]`: AC1처럼 r_value(가중치 0.05) 켬(033) |
| z | 없음(2026-11-03 사용자 결정): AC1·출발 체크포인트와 같은 조건, z 테이블 없이 모든 게임 z 없음(033-R8 pseudo-reward 꺼짐) |
| 상한 | agent step 상한 없음(`--max-agent-steps` 안 줌). 중단은 사용자 결정 |
| 점검 | 100만 agent step마다 T1 저그 3종(Steamhammer·ZZZKBot·UAlbertaBot) 각 10판, stochastic, 투혼, seed_base 43000(11-03 측정과 같음). 첫 중간 보고는 500만 step |
| 단계 전환 | 자동 전환 없음. T1 비중을 올릴지는 점검 추이를 보고 사용자가 정한다 |

1차 결과(`reports/m6_t1self_stop.md`): update 2,198(agent step 441만, LATEST `ckpt-4413190`)에서 정지했다. 셀프 상대가 현재 Main뿐이라 양쪽이 함께 "일꾼·터렛·확장, 병력 거의 없음"의 수동적 정책으로 무너졌다. 마지막 체크포인트는 자기끼리 10판 중 9판 시간 제한 무승부, 출발 체크포인트에게 0-10이다. 셀프 판이 학습 step의 67~76%였고, 무승부는 r_win 0이라 그 판들의 정책 신호는 r_value뿐이었다. T1 초반 러시 약점은 그대로였고 c3M에는 셀프 습관(배럭 없이 터렛·CC)이 T1 판까지 번졌다. Windows 빌드의 Steamhammer 프로토스는 8~10분에 같은 위치에서 결정적으로 크래시했다(프로토스 판 27판 중 10판). `t1self-s1` 체크포인트·로그는 실패 기록으로 보존한다.

**2차 `t1self-s2` (2026-11-03 사용자 결정, AMB-042-14)**: 1차와 같은 출발에서 새 run으로 다시 시작한다. 아래만 바꾸고 나머지(V-trace, `--init`·`--bc`, step·update 0부터, C1 고정·자동 전환 없음, 두 맵, T/P/Z와 종족 샘플링, z 없음, step 상한 없음·중단은 사용자 결정, 100만 step마다 T1 저그 3종 점검을 학습과 동시에, 첫 중간 보고 500만)는 1차와 같다.

| 항목 | 값 |
|---|---|
| run id | `t1self-s2`. 시작 `--init ac1-vtrace-s1/ckpt-3133569 --bc bc-s2e2-pt/ckpt-11387` |
| 설정 파일 | `config/rl_t1self2.toml`(메인 임시 결정: 1차 설정 `config/rl_t1self.toml`은 기록으로 그대로 둔다) |
| 상대 비율 | `c1_t0 = 0`, `c1_t1 = 0.5`, `c1_self = 0.15`(현재 Main), `c1_snapshot = 0.25`(과거 Main 스냅샷, PFSP f_hard), `c1_anchor = 0.10`. 스냅샷이 아직 없으면 스냅샷 몫은 anchor로 간다 |
| anchor | 출발 체크포인트 AC1 V-trace s1 `ckpt-3133569`의 `agents/main`(`[league] anchor = "checkpoints/ac1-vtrace-s1/ckpt-3133569"`) |
| 스냅샷 | `[snapshot] main_every_updates = 250`, `main_candidates = 4`(최근 공개 4개가 후보, 이전 것은 공용 드라이브에 남김) |
| 보상 | `r_value_stages = ["C0", "C1"]` 그대로, `r_value_vs_model = false`: T1 판만 r_value(0.05) 켬, 모델 상대 판은 r_value 정책 가중치 0(value head는 모든 판에서 학습) |
| T1 풀 | McRave 제외 그대로. Steamhammer는 테란·저그만(`races = ["Terran", "Zerg"]`, Windows 빌드 프로토스 크래시). 종족별 T = UAlbertaBot·Steamhammer·Iron, P = UAlbertaBot·Stardust, Z = UAlbertaBot·Steamhammer·ZZZKBot(021-R4 종족마다 2개 이상 충족) |
| Random | 021-R7: 세 종족 봇(UAlbertaBot)만 Random 20%. Steamhammer는 Random 없이 테란·저그 판에만 나온다 |

2차 결과(2026-11-03 사용자 결정으로 정지, `reports/timeline.md` 11-03 17:03): update 1,751(LATEST `ckpt-3525959`)에서 정상 종료했다. 최근 120판에서 과거 스냅샷 상대 3-17-10, anchor 상대 1-4-4로 퇴보 신호가 있었고, 점검의 UAlbertaBot 저그는 2-7-1 → 1-5-4 → 0-5-5였다. 셀프 붕괴는 1차보다 느렸지만 같은 방향이었다. 같은 날 자문 진단: 4~5풀 방어에 성공한 궤적이 학습 데이터에 없고, 학습 step의 72%가 TvT 모델 상대, 32%가 보상 0인 무승부 판이었다. `t1self-s2` 체크포인트·로그는 보존한다.

### 러시 커리큘럼 run (`rush-s1`) (2026-11-03 사용자 결정, AMB-042-15)
T1·셀프 run 다음 RL run이다. 러시 상대 기준 측정(`reports/m6_rush_ladder.md`, 출발 체크포인트, 상대당 30판)에서 승률은 UAlbertaBot 9풀 변형 87%, T0 `zerg_rush` 47%, UAlbertaBot 기본 저그 27%, Steamhammer 오버풀 변형 3%, ZZZKBot 0/10(참고)이었다. 무승부 대부분은 우리가 무너졌는데 상대가 마무리하지 못한 판이라 위 붕괴 항복(042-R14)을 함께 쓴다.

| 항목 | 값 |
|---|---|
| run id | `rush-s1`. 시작 `--init ac1-vtrace-s1/ckpt-3133569 --bc bc-s2e2-pt/ckpt-11387` |
| 설정 파일 | `config/rl_rush1.toml`(메인 임시 결정. 1·2차 설정 파일은 기록으로 그대로 둔다) |
| 알고리즘 | V-trace |
| step·update | 새 run id로 0부터. KL 계수 0.1부터, 옵티마이저 새로(1·2차와 같음) |
| 단계 | `start_stage = "C1"`, `auto_advance = false`. 사다리는 C1 상대 몫으로 두고 단계 변경은 사용자 결정이다(메인 임시 결정). C1이라 Main만 학습한다(042-R7) |
| 상대 비율 | `c1_ladder = 0.70`, `c1_anchor = 0.10`, `c1_snapshot = 0.15`, `c1_self = 0.05`(현재 Main), `c1_t0 = 0`, `c1_t1 = 0`. 스냅샷이 아직 없으면 스냅샷 몫은 anchor로 간다(2차와 같음) |
| 러시 사다리 | 저그 러시 상대만(`race = "Zerg"`): T0 `zerg_rush` 빌드(T0는 고치지 않는다, 021)와 T1 러시 봇·그 설정 변형(021 "T1 설정 변형"). 후보는 UAlbertaBot 기본(저그 `Zerg_ZerglingRush`), UAlbertaBot 9풀 변형, Steamhammer 오버풀 변형, ZZZKBot과 진행 중인 변형 측정에서 나온 변형이다. 목록(2026-11-03 사용자 결정, `config/rl_rush1.toml` `[[ladder]]`, 쉬운 것부터, 출발 체크포인트의 승률 / 막음 비율, `reports/m6_rush_ladder.md` 7절·10절): T1 `ualbertabot9pool`(준비, 87% / 87%), T0 `zerg_rush`(47% / 33%), T1 `ualbertabot12poollings`(37% / 33%), T1 `ualbertabot`(기본 `Zerg_ZerglingRush`, 27% / 13%), T1 `ualbertabot9poollings`(27% / 13%), T1 `steamhammeroverpool`(3% / 3%), T1 `zzzkbot`(10판 0승) 7개. Steamhammer `11Pool`·`9HatchExpo9Pool9Gas` 변형은 오버풀만큼 어려워(0~3%) 넣지 않았다. 순서는 기록용이고 선택은 PFSP f_var(`ladder_pick = "f_var"`)가 승률로 한다 |
| 저그가 아닌 T1 | 이번 단계에서는 쓰지 않는다(`c1_t1 = 0`, 사용자 결정). 021-R4의 run 풀 조건(종족마다 2개 이상)의 예외다. 이 run의 상대 종족은 사다리의 저그와 모델 상대(TvT)뿐이다 |
| anchor | 2차와 같다(`[league] anchor = "checkpoints/ac1-vtrace-s1/ckpt-3133569"`, 출발 체크포인트) |
| 스냅샷 | 2차와 같다: `main_every_updates = 250`, `main_candidates = 4`, 후보 안 PFSP f_hard |
| 보상 | 2차와 같다: `r_value_stages = ["C0", "C1"]`, `r_value_vs_model = false`. 봇(사다리) 판만 r_value(0.05) 켬 |
| 붕괴 항복 | `[concede] enabled = true`, 나머지 기본값(화면 서플라이 ≤ 4가 3분 이후 120초 연속, 042-R14) |
| z | 없음 |
| 맵 | 투혼·서킷브레이커 균등 |
| 상한 | agent step 상한 없음. 중단은 사용자 결정 |
| 점검 | 100만 agent step마다 학습과 동시에: T0 `zerg_rush`, UAlbertaBot 기본(저그), Steamhammer 오버풀 변형, ZZZKBot 각 10판, stochastic, 투혼, seed_base 43000. 승·패·무와 함께 막음 비율(`reports/m6_rush_ladder.md` 2절 정의: 게임 1분 이후 우리 SCV가 4기 이하가 된 적이 없는 판의 비율, 리플레이 재생으로 셈)을 보고한다. 평가 게임이라 붕괴 항복은 쓰지 않는다. 첫 중간 보고는 500만 step(메인 임시 결정, 1·2차와 같음) |
| 사다리·비중 변경 | 자동 변경 없음. 점검 추이를 보고 사용자가 정한다. 정상 종료(042-R10) → 설정 수정 → 재시작(재개 시 설정) |

### V-trace vs PPO 채택 기준
- PPO 비교 설정: clip 0.2, GAE λ 0.95, 같은 배치를 2 epoch, value clip 없음. 나머지(KL·엔트로피·옵티마이저)는 같다.
- 판정: 042-AC1 설정에서 같은 agent step(아래 AC1)까지 3 seed 평균 최종 승률이 **3%p 넘게 높은 쪽**을 채택한다. 차이가 3%p 이하이면 비동기 actor와 잘 맞는 V-trace를 채택한다. 어느 seed든 발산(NaN, 또는 KL(π‖π_BC)이 계속 증가해 승률이 BC 미만으로 떨어짐)하면 그 알고리즘은 탈락한다.
- 판정(2026-11-02): 3 seed 평균 V-trace 75.5%, PPO 78.2%, 차이 2.7%p ≤ 3%p라 **V-trace 채택**. 발산한 seed는 없다(`reports/m6_ac1.md`).

## Requirements
| ID | 요구사항 |
|---|---|
| 042-R1 | **MUST** 학습 재개(resume)를 지원한다. 저장 대상은 가중치, 옵티마이저, 리그 상태, 승률 테이블, RNG 상태다. 다른 학습 호스트(ROCm↔CUDA, 2026-10-23. 이전 Vulkan↔CUDA)에서도 재개할 수 있어야 한다(061-R10, 061-AC5) |
| 042-R2 | **MUST** 에피소드 결과·보상 항목·KL·엔트로피·처리량을 TensorBoard 호환 로그 또는 CSV로 남긴다. 학습한 agent step 누계(`trained_steps`, 재개해도 run 누적이라 체크포인트에 기록)와 큐 자리를 기다린 시간(`queue_wait_s`)도 `train.csv`에 남긴다(2026-10-31). `queue_wait_s`는 그 행 에이전트의 큐에서 그 에이전트의 직전 행 이후 끝난 기다림의 초 합이다(기다림은 끝날 때 센다, 게임 마지막 세그먼트의 기다림 포함) |
| 042-R3 | **MUST** Aborted 에피소드는 학습에서 제외한다(020-R4). 세그먼트는 끝나는 대로 큐에 넣고, 게임이 Aborted로 끝나면 그 게임의 세그먼트 중 아직 학습에 쓰지 않은 것을 큐에서 회수한다(2026-10-17 사용자 결정, AMB-042-6: 게임 끝까지 붙잡으면 정책 지연으로 대부분 버려짐). Ctrl+C(R10)는 learner를 먼저 멈추므로 완전히 제외된다. 크래시처럼 드문 경우 이미 학습에 쓴 세그먼트는 되돌리지 않으며, 그 수를 로그에 남긴다. **역압 기다림(2026-10-31)**: 큐 자리를 기다리던 게임이 Aborted로 끝나거나 중지되면 그 기다림을 취소하고 세그먼트를 넣지 않는다(`revoked`로 센다). 회수한 게임 id는 기억해 두고, 그 뒤에 늦게 도착한 그 게임의 세그먼트도 넣지 않는다. Ctrl+C(R10)로 큐를 닫으면 기다리는 게임을 모두 풀고 세그먼트는 넣지 않는다(R10 순서는 그대로). 학습 루프가 오류로 끝날 때도 큐를 닫아 기다리는 게임이 남지 않게 한다 |
| 042-R4 | **MUST** 정책 지연(behaviour vs target 버전 차이)을 모니터링하고, 임계값을 넘는 trajectory는 버린다 |
| 042-R5 | **SHOULD** 커리큘럼 단계 C0(T0 봇 위주) → C1(T0+T1+셀프) → C2(풀 리그) 전환은 승률 조건으로 자동화한다. 시작 단계와 자동 전환 여부는 설정 `[curriculum] start_stage`·`auto_advance`(기본 `"C0"`·켬)로 정하고, 자동 전환을 끈 run은 사용자 결정으로만 단계를 바꾼다(2026-11-03, AMB-042-13) |
| 042-R6 | **MUST** 체크포인트 승격은 [050](050-evaluation.md)의 회귀 게이트를 통과해야 한다 |
| 042-R7 | **MUST** 에이전트 3개(Main, Main Exploiter, League Exploiter)는 학습 호스트의 learner를 시분할로 쓴다. Main이 학습 시간의 50% 이상을 차지한다. 비율은 설정 파일로 관리한다 |
| 042-R8 | **MUST** RL 학습 시퀀스 길이는 T=64다(032-R5와 동일) |
| 042-R9 | **MUST** KL 계수, 엔트로피 계수, 커리큘럼 전환 승률 기준, 스냅샷 주기는 튜닝 대상이다. 설정 파일로 관리하고 체크포인트에 기록한다 |
| 042-R10 | **MUST** 정상 종료(Ctrl+C, 2026-10-16 사용자 결정). 첫 Ctrl+C(SIGINT, Windows는 Ctrl+C 이벤트)를 받으면 다음 순서로 멈춘다. ① 새 게임 배정을 멈춘다. ② 진행 중인 게임을 중단하고 `Aborted`로 기록해 학습에서 뺀다(042-R3). ③ 체크포인트를 저장하고 push한다(061-R4 "정상 종료 시"). ④ 잠금을 해제한다(061-R5). 두 번째 Ctrl+C를 받으면 저장 없이 즉시 종료한다. 게임 호스트 구성(hosts 파일)을 바꾸려면 정상 종료 후 재시작한다. 재시작하면 hosts 파일을 새로 읽고 마지막 체크포인트에서 재개한다(R1) |
| 042-R11 | **MUST** 학습 중 리플레이를 저장한다(2026-10-18 사용자 결정). 모든 게임을 `save_replay`로 시작하고(게임 호스트가 임시로 저장, 020-R1 `Finished.replay`로 학습 호스트에 전달), 판정 뒤 **정기 표본(게임 id가 `[replays] sample_every`(기본 100)의 배수)** 이거나 **승자 불일치 게임(020-R13)** 이면 `<URANIUM_SHARED_DIR>/runs/<run_id>/replays/g<game_id>-<sample\|disagree>.rep`에 쓴다(임시 이름으로 쓰고 다시 읽어 바이트를 대조한 뒤 이름을 바꾼다). 나머지는 버린다. run이 끝나도 지우지 않고, 저장 예산(060-R7)을 넘어도 자동 삭제하지 않는다(경고만, 2026-10-19 사용자 결정, AMB-061-5. 총량 측정은 060-R7). `games.csv`에 `replay`(run 폴더 기준 경로)와 `replay_reason` 열을 남긴다 |
| 042-R12 | **MUST** agent step 상한(`--max-agent-steps`, 042-AC1·033-AC2는 300만(2026-11-01 사용자 결정, 이전 1,000만))에 도달하면 R10의 정상 종료 순서(①~④)로 멈춘다. agent step은 체크포인트(`TrainState.step`)에서 이어 세므로 재개해도 run 전체 누적이다. 진행 중이던 게임은 R10처럼 Aborted로 빠지므로 실제 학습량은 상한을 약간 넘거나 모자랄 수 있고, 마지막 체크포인트 step으로 기록한다 (2026-10-18). **세는 기준(2026-10-31 사용자 결정)**: 상한은 그대로 만든 act 호출 수(agent step)로 센다. 역압(기본 `queue_full = "wait"`, 큐 행)에서는 큐 넘침 버림이 기다림 상한을 넘을 때만 생기므로, 학습한 step과의 차이는 정책 지연 버림(R4), Aborted 회수(R3), 멈출 때 진행 중이던 판으로 줄어든다. 학습한 step 누계는 `train.csv` `trained_steps`(R2)로 남긴다 |
| 042-R13 | **MUST** RL learner는 win_open(032-R18)이 있는 모델에서 win_open을 win의 v-trace target에 MSE로 학습한다(가중치 `config/rl.toml` `[loss] win_open`, 기본 0.5, value와 같은 0.5×가중치×제곱오차 형태, 2026-10-21 사용자 결정, AMB-031-6). advantage·policy gradient에는 쓰지 않고, 전역 clip은 041-R19처럼 win_open 파라미터를 따로 한다. RL 게임(OpenBW)에서 항복 모드는 plan 기본값(020-R14 `shadow`)을 따른다. 붕괴 항복(042-R14)은 이 항복 모드와 별개의 규칙이다 |
| 042-R14 | **MUST** RL 학습 게임 붕괴 항복(2026-11-03 사용자 결정, AMB-042-15). `[concede] enabled`면 RL 학습 게임의 모델 정책 세션은 위 "붕괴 항복" 절의 조건(자기 서플라이 화면 표시 ≤ `max_supply`가 게임 `after_s` 이후 `hold_s` 연속)에서 010-R13 `surrender`로 떠나고, 그 판은 항복한 쪽 패배로 판정해 학습한다(020-R18, 033 r_win −1). 자기 상태만 쓴다(맵핵 아님). 평가 게임에는 적용하지 않는다. `games.csv` `end_reason = "Concede"`로 기록한다 |

## Acceptance Criteria
- 042-AC1: 작은 설정(1맵, T0 1종족)에서 BC 정책 대비 승률 개선을 재현 가능하게 보인다. 3 seed 평균으로 측정한다.
  - 설정(2026-10-17): 투혼, 상대는 BC 정책의 승률이 20~80%인 T0 종족 하나 → **저그**(2026-10-31 사용자 결정: 출발 체크포인트 `bc-s2e2-pt/ckpt-11387`의 투혼 stochastic 50판 33승 2패 15무, 66%, Wilson 95% 52~78%), seed당 **300만** agent step(2026-11-01 사용자 결정: V-trace s1 학습 곡선이 270만 step 안에 BC 68.3% → 약 76%로 오른 뒤 1,000만까지 평가 오차 안에서 평평 — `reports/timeline.md` 11-01, 이전 1,000만. 이미 1,000만까지 돈 V-trace s1은 300만에 가장 가까운 체크포인트 `ckpt-3133569`로 판정·비교한다), 학습 전후 각 200판 평가(050-R1 두 모드). 개선 = 3 seed 모두 학습 후 승률의 Wilson 95% 하한이 BC 승률의 상한보다 높음.
  - 판정 샘플링(2026-10-31 사용자 결정): 개선 판정은 **확률적 샘플링** 승률로 한다(RL actor가 확률적으로 두며 학습이 키우는 것이 이 승률이다. BC argmax는 0%라 20~80% 조건이 성립하지 않는다). argmax는 BC 기준선과 처음 네 run(V-trace s1·s2, PPO s1·s2)에서 모두 0/200이어서, 2026-11-02 사용자 결정으로 이후 AC1 평가에서 뺀다(050-R1의 예외). argmax 쪽에서 나아질 징후가 보이면 다시 넣을지 사용자에게 묻는다.
  - 학습 전 기준선(2026-10-31 사용자 결정): RL은 `last_failed` 실제 값을 쓰므로(030-R10), BC 체크포인트를 `last_failed` 0 고정과 실제 값으로 각 200판(같은 seed) 재고 **높은 쪽 승률**을 BC 승률로 쓴다. 학습 후 평가는 실제 값이다.
  - z(2026-10-31 사용자 결정): AC1의 6 run은 z 테이블 없이(모든 게임 z 없음, pseudo-reward 꺼짐, 033-R8) 진행한다. BC 기준선(z 없음)과 조건이 같고, z가 판 시작 때 고정돼 정찰에 따른 빌드 변경을 담지 못하므로 유연성을 위해 일단 z 없이 간다. 033-AC2·리그에서 z를 쓸지는 그 전에 다시 묻는다.
  - 규모(2026-10-17 사용자 결정, AMB-042-7): V-trace·PPO 각 3 seed(6 run)로 알고리즘을 고른다. 033-AC2 ablation 3 run은 본 BC 체크포인트와 z 테이블이 나온 뒤 실행한다. 처리량은 TBD(M6)로 측정해 run 시간을 다시 보고한다.
  - 결과(2026-11-02, `reports/m6_ac1.md`): **6 run 모두 seed별 개선 기준 미달**(BC 68.3%, 기준 = 학습 후 Wilson 하한 > 74.4%). 판정 규칙대로 V-trace 채택(3 seed 평균 75.5% vs PPO 78.2%, 차이 2.7%p ≤ 3%p). 다음 run은 위 "T1·셀프 run"이었고(1차 실패, 2차 `t1self-s2`도 정지), 그다음은 "러시 커리큘럼 run"(`rush-s1`)이다.
- 042-AC2: 72시간 연속 학습 중 크래시나 메모리 누수 없이 resume이 가능하다.
- 042-AC3: Main이 이전 승격 체크포인트 대비 승률 ≥ 55%일 때만 승격한다.
- 042-AC4: 학습 중 Ctrl+C 한 번이면 체크포인트 push와 잠금 해제까지 마치고 종료한다. 이어서 hosts 파일의 게임 호스트를 하나 빼거나 더해 재시작하면, `--take-over` 없이 같은 step에서 재개하고 바뀐 호스트 구성으로 게임을 배정한다(042-R10).
- 042-AC5 (2026-11-03, 042-R14): 붕괴 항복 판정 단위 테스트 — `after_s` 경계(4,319 프레임 결정은 세지 않음·4,320부터 셈), `hold_s` 경계(2,879 불가·2,880 가능), 무너짐 아닌 결정 하나에 끊김, 서플라이 2배 단위(8 무너짐·9 아님), `enabled = false`면 판정 없음, 한 판 한 번, 설정 검사. 통합: OpenBW 1판에서 항복 판이 항복한 쪽 패배로 집계되고(Aborted 아님) `games.csv` `end_reason = Concede`, r_win −1이 마지막 실제 결정에 붙는다.

## Risks & Open Questions
> **[SPEC DEFERRED(M6)] AMB-042-2**: TvP·TvZ 실력이 T0·T1 봇 수준에서 정체될 수 있다(셀프플레이는 TvT만 가능). 지금은 T0·T1 P/Z 봇으로 진행한다(2026-10-15 결정). M6 중간 점검 때 아래 내용을 사용자에게 보고하고 대응을 다시 확인받는다.
> - 보고 내용: TvP·TvZ 승률이 정체되었는지, 또는 T1 봇 상대 승률이 포화되었는지
> - 대응 후보: 프로토스·저그를 플레이하는 상대 전용 정책을 추가할지
>
> M6 시작 시 재확인(2026-10-17): T0·T1 P/Z 봇으로 시작한다. M6 중간 점검에서 위 내용을 보고하고 다시 확인받는다(Deferred 유지).

- **러시 커리큘럼 run의 종족 범위(2026-11-03)**: `rush-s1`의 봇 상대는 저그 러시뿐이고 나머지는 테란 모델 상대(TvT)다. TvP 판과 저그 운영형 상대가 없어 그쪽 실력이 떨어져도 점검(저그 러시 4종)에 드러나지 않는다. 다른 종족·T1 봇 상대 확인 시점은 사용자에게 묻는다(AMB-042-2와 함께).

- **초기 설정값 (확정 2026-10-17, AMB-042-3)**: 위 "초기 설정값" 절의 수치는 사용자 위임으로 Claude가 문헌 기준으로 정했다. 042-AC1 이후 조정을 제안할 수 있다.

- **바깥 LSTM 지원(2026-10-27, 032-R20 ④)**: BC 본학습 모델은 구조 변형 모두 켬이라 바깥 LSTM(`outer`, 032 §2.7)이 있다. 세그먼트가 바깥 구간 안 아무 위치에서나 시작하므로, 학습기와 BC 참조 forward는 032 §2.7의 위상별 구간 계산을 쓴다(사용자 확인, 구현 완료). 세그먼트 길이·겹침(R8)은 바꾸지 않았다. Core 계산이 구간 수만큼(최대 3배) 늘어난다. CUDA 호스트 합성 입력 측정(2026-10-27, 2줄 × 64 step, 엔티티 128, bf16, forward+backward)에서 구간 안에서 시작하는 세그먼트는 161 ms로, BC 경로 123 ms보다 약 32% 길었다(기본 구성 113 ms). 메모리는 같았다. 실제 RL update 시간은 042-AC1 준비 때 잰다.

- 학습 호스트 GPU 하나에서 추론과 학습이 경쟁한다. 다른 호스트의 GPU를 추론 전용으로 쓰는 옵션은 [060](060-infra.md) R5를 따른다.

- **학습 예산 (확정 2026-10-15, AMB-042-1)**
  - 기간 제한 없이 성능이 오르는 동안 계속 학습한다.
  - 리그는 Main + Main Exploiter + League Exploiter 전체 구성을 쓴다.
  - 042-R1(resume)과 060-R6(자동 재시작)이 필수 전제다.
  - 학습은 커리큘럼 C2 이후에도 종료 조건 없이 진행한다. 중단은 사용자가 결정한다.
