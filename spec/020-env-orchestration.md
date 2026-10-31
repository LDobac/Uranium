# 020 — Env Orchestration

- **Status**: Approved (2026-10-15, 2026-10-23 개정: PyTorch 전환(사용자 결정, AMB-000-4) — R5 정책 버전은 `model.safetensors` 해시, R7·운영 메모 Linux ROCm 호스트, 모델 plan은 Python 진입점·추론 백엔드 선택. 2026-10-16 개정: 020-AC1 기준, Windows 호스트 로컬 hostd, 2026-10-17 개정: 스케줄러 crate `uranium-runner`, plan 플레이어 `model`과 slot별 정책 라우팅(020-R11), 정책 버전 기록 / 020-R12 동적 Job 공급자(RL 리그, 3단계 계획 승인) / 020-R13 승자 불일치 판정(AMB-020-3, 사용자 결정), 2026-10-18 개정: `Finished.replay` 리플레이 바이트 반환(020-R1, 042-R11), 020-R13 상호 퇴장 = Aborted·hostd 드롭 시간(011-R17·R18, 사용자 결정) / summary.md 정책 표시·정책별 합산(사용자 결정, Risks 절 메모). 2026-10-19 개정: R3 에이전트 무응답 30초 중단 삭제 → OpenBW 드롭(011-R18)·상호 퇴장 Aborted(R13)로 정합(C-9, 사용자 결정), 장비 표기 중립화(001-R10). 2026-10-21 개정: 020-R14 plan 항복 옵션(사용자 결정, AMB-031-6). 2026-10-26 개정: 020-R15 평가 작업(추론 장비 이어받기·마무리 정지·재개, 사용자 결정, AMB-020-5), 020-R16 플레이어 이름, R4 브리지 미접속 판, R1 제어 프로토콜 버전 3, AC6·AC7. 2026-10-31 개정: 020-R17 브리지 런처에만 `URANIUM_BRIDGE_EXT`(사용자 결정, 011-R23))
- **Related**: [011](011-bridge-openbw.md), [021](021-opponent-pool.md), [042](042-rl-training.md), [060](060-infra.md)

## Purpose
여러 OpenBW 게임을 로컬이나 원격 호스트에서 병렬로 실행하고, 에피소드 수명주기를 관리한다.

## 구성
- **uranium-hostd**: 각 OpenBW 호스트에서 실행되는 데몬이다. 학습 호스트의 `uranium-env`가 gRPC가 아닌 단순 TCP 제어 프로토콜(postcard)로 지시한다.
- **게임 쌍**: `BWAPILauncher` 2개로 구성된다.
  - P1: 우리 봇 브리지
  - P2: 상대. 우리 봇 브리지(셀프플레이) 또는 OSS 봇 `.so`
  - 두 런처는 Linux에서 `OPENBW_LAN_MODE=LOCAL`(같은 호스트 소켓), Windows에서 `OPENBW_LAN_MODE=TCP`(포트 2개)로 매칭한다.
- **uranium-env**: 학습 호스트의 게임 스케줄러 바이너리다(crate `uranium-runner`, 2026-10-17 분리, AMB-001-3). 에이전트 서버를 내장하고, 매치업 계획을 여러 hostd에 분배하고, 결과를 모은다.
- hostd와 env는 Linux와 Windows 모두에서 돈다(001-R8).

## Requirements
| ID | 요구사항 |
|---|---|
| 020-R1 | **MUST** hostd API: `StartGame{map, p1, p2, start_locs?, seed, agent_addrs, save_replay}`, `StartReplay{rep_bytes, perspective_slot, agent_addr, privileged: bool}`(040-R12 스트리밍, 011 재생 모드), `StopGame{id}`, `Status`, `ListCapabilities{max_concurrent_games, max_concurrent_replays, oss_bots}`. `save_replay` 게임이 끝나면 hostd는 slot 0이 저장한 리플레이(두 클라이언트가 같은 게임을 기록하므로 하나면 충분) 바이트를 `Finished.replay`에 담아 돌려준다. 원격 게임 호스트의 리플레이를 학습 호스트가 모으기 위한 것이다. `URANIUM_HOSTD_REPLAY_DIR`이 있으면 그 폴더에 파일도 남기고, 없으면 작업 폴더의 임시 파일을 보낸 뒤 지운다. 제어 프로토콜 버전 2 (2026-10-18, 042-R11). 버전 3 (2026-10-26, 020-R16): `GameSpec.players[i].name`(선택) |
| 020-R2 | **MUST** 게임별 작업 디렉터리와 포트를 격리한다. 같은 호스트에서 N개 게임이 동시에 돌아도 충돌이 없어야 한다 |
| 020-R3 | **MUST** 타임아웃 처리 |
| 020-R4 | **MUST** 크래시 처리: 런처 비정상 종료 시 에피소드를 `Aborted`로 표시하고 학습 데이터에서 제외한다. 같은 설정이 연속 3회 크래시하면 격리하고 경보를 낸다. **브리지 미접속 판**(2026-10-26): hostd가 `Finished`를 보고했지만 Hello 검사를 통과한 세션 수가 브리지 플레이어 수보다 적으면, 브리지가 에이전트에 접속하지 못하고 나간 판이다. 스케줄러는 이 판을 `Finished`가 아니라 결과 없는 실패(`Aborted`, `detail = "bridge never connected (<접속 수> of <브리지 수> sessions)"`)로 기록하고, summary에서 승패 대신 따로 센다. 평가 작업(020-R15)은 이 판을 다시 배정한다 |
| 020-R5 | **MUST** 에피소드 메타데이터를 기록한다: id, 맵, 시작 위치, 상대 종류·버전, 정책 버전, 결과, 길이, 리플레이 경로. 모델 정책의 정책 버전은 체크포인트 id(체크포인트 폴더 이름 + `model.safetensors` SHA-256 앞 12자리, 061-R10. 2026-10-22까지는 `model.bpk`)와 샘플링 모드다 |
| 020-R12 | **MUST** `uranium-runner`는 라이브러리로도 쓸 수 있어야 한다. 게임 배정은 Job 공급자 인터페이스(다음 Job 요청, 시작 못 한 Job 되돌림, 게임 결과 통지)로 받는다. matchup plan은 정적 공급자, RL 리그(042)는 상대와 맵을 그때그때 정하는 동적 공급자다. 정상 종료(042-R10)를 위해 공급자 중지와 진행 중 게임 `StopGame`을 지원한다 (2026-10-17) |
| 020-R13 | **MUST** 같은 게임의 두 세션이 보고한 승자가 다르면 **더 늦게 끝난 세션(GameEnd 프레임이 더 큰 쪽)의 승자**를 게임 결과로 쓴다. 프레임이 같으면 먼저 받은 보고를 쓴다. 불일치 건수는 따로 집계해 보고서에 남긴다. 이 판정은 평가 집계(`uranium-env` summary), RL 리그 승률 표와 r_win(042, 033) 모두에 쓴다 (2026-10-17 사용자 결정, AMB-020-3). **상호 퇴장**: 두 세션이 모두 `Surrender`(상대가 섬멸되지 않고 떠남, 011-R17)로 자기 승리를 보고하면 연결 끊김(OpenBW 드롭)으로 보고 게임을 `Aborted`로 처리한다. 평가 집계에서는 승패에서 빼고 "dropped" 열로 세며, RL에서는 그 게임의 세그먼트를 회수한다(042-R3). 진짜 항복(봇의 GG 퇴장)은 떠난 쪽 세션이 상대를 승자로 보고하므로 기존 규칙대로 판정한다 (2026-10-18 사용자 결정). hostd는 모든 런처에 OpenBW 드롭 시간 `URANIUM_OPENBW_DROP_TIMEOUT_MS`(기본 600,000ms, 011-R18)를 넘긴다 |
| 020-R11 | **MUST** plan의 플레이어 종류에 `model`(체크포인트로 플레이하는 테란 정책)을 둔다. 스케줄러는 `StartGame`을 보내기 전에 에이전트 서버의 라우팅 표에 `(game_id, slot) → 정책`을 등록하고, 세션은 `Hello.game_id`(010-R9)와 `GameSpec.players` 인덱스로 정책을 찾는다. 인덱스는 hostd가 붙이는 플레이어 이름 `uranium-p<i>`에서 읽는다. BWAPI 플레이어 id(`Hello.slot`)는 런처 순서와 다를 수 있기 때문이다. 이름이 없으면 `Hello.slot`을 쓴다. 2026-10-26부터 이름 형식은 020-R16(`p<i>-<표시 이름>`)이고, 기존 `uranium-p<i>`도 읽는다. 표에 없는 세션은 서버 기본 정책(T0)을 쓴다 (2026-10-17) |
| 020-R6 | **SHOULD** 호스트 목록과 호스트별 동시 게임 수는 설정 파일로 관리한다. 실제 주소가 들어가므로 커밋하지 않는 로컬 파일(`URANIUM_HOSTS_FILE`)에 두고, 형식 예시만 `config/hosts.example.toml`로 커밋한다(001-R10, 2026-10-15 변경) |
| 020-R8 | **MUST** hostd는 OS별 부분(LAN 매칭 설정, 프로세스 정리)을 플랫폼 계층으로 분리한다. Linux는 LOCAL 소켓과 프로세스 그룹 + `PR_SET_PDEATHSIG`를 쓰고, Windows는 TCP 포트 쌍(`URANIUM_HOSTD_PORT_BASE`부터 할당)과 Job Object(kill-on-close)를 쓴다 |
| 020-R9 | **MUST** hostd 설정은 `.env`에서 읽는다(001-R9): `URANIUM_HOSTD_LISTEN`, `URANIUM_BWAPI_DIR`, `URANIUM_BRIDGE_LIB`, `URANIUM_DATA_DIR`, `URANIUM_MAPS_DIR`, `URANIUM_HOSTD_MAX_GAMES`, `URANIUM_HOSTD_PORT_BASE`(Windows), `URANIUM_CACHE_DIR`(측정 파일, 020-R10) |
| 020-R10 | **MUST** 호스트별 동시 게임 수는 측정으로 정한다. 게임 호스트에서 `uranium-env calibrate`를 한 번 실행하면 동시 게임 수 N을 1부터 늘려 가며 T0 대전(단계마다 2N판)을 돌린다. 세션별 브리지 p99(011-AC2 지표)의 중앙값이 300µs 이하인 가장 큰 N을 권장값으로 `<URANIUM_CACHE_DIR>/hostd_calibration.toml`에 저장한다. hostd의 동시 게임 수는 다음 순서로 정한다: `.env` `URANIUM_HOSTD_MAX_GAMES` → 측정 파일 → 측정 전 기본값 `max(1, (코어 수 − 1) / 2)`(경고 출력). 학습 프로세스와 함께 돌릴 때의 CPU 예산(020-R7)은 M5/M6에서 같은 명령을 확장해 정한다 (2026-10-16 사용자 결정) |
| 020-R7 | **SHOULD** Linux 학습 호스트(ROCm, 2026-10-23. 이전 Vulkan)의 로컬 hostd도 지원한다. 이 경우 CPU 코어를 추론 서버와 나눠 쓰는 예산을 정한다([060](060-infra.md)) |
| 020-R14 | **MUST** plan 항복 옵션(2026-10-21 사용자 결정, AMB-031-6): `surrender = "off" \| "shadow" \| "on"`(031-R12 모드, OpenBW plan 기본 `"shadow"` — 학습·평가 게임은 나가지 않는다)과 `surrender_chat`(문자열, 기본 `"gg"`, 빈 문자열이면 채팅 없이 퇴장, UTF-8 79바이트 초과는 plan 로드 오류(010-R13)). 모델 정책 슬롯에만 적용된다. summary는 모델 정책별로 열 두 개를 더한다: 판정이 성립한 게임 수(Shadow·On 공통), 그중 그 정책이 이긴 게임 비율(Shadow에서만 의미 있음 — 오항복률). `results.jsonl`의 세션 통계에 첫 성립 프레임을 남긴다 |
| 020-R15 | **MUST** 평가 작업(2026-10-26 사용자 결정, AMB-020-5). 여러 plan을 차례로 도는 긴 평가를, 추론 장비가 빠졌다 다시 합류해도 이어 갈 수 있게 한다. ① **작업 폴더**: 공용 드라이브(061) `eval/<job>/job.toml`에 단계(`stages`, plan 경로 목록, 이 순서대로)와 호스트 목록 파일(`hosts`, 020-R6 형식)을 둔다. job 폴더 바로 아래에 장비별 기록(`games/<host>.jsonl`, `results/<host>.jsonl`, hostd 판 번호로 기록)과 진행 중 표(`inflight/<host>.json`), 빠지는 호스트 표시(`leaving/`)를 두고, 단계마다 `<job>/<단계 이름>/` 아래에 리플레이(`replays/g<plan 판 번호>.rep`)와 ⑨의 합친 기록·요약을 둔다. 같은 파일에 두 장비가 동시에 덧붙이지 않는다. ② **작업자**: `python -m uranium eval-worker --job <dir> --backend <cuda\|rocm\|cpu> --priority <n> [--local-hostd <addr>]`. 시작할 때 에이전트 서버를 연다. 모델(020-R11 체크포인트·샘플링 쌍)은 판 배정 권한을 얻은 뒤 남은 판이 있는 단계의 것을 한 번에 올리고, 첫 판 전에 모든 (배치 2의 거듭제곱 ≤ max_batch, 엔티티 수) 모양을 미리 돌린다(warm-up: 게임 도중 처음 보는 모양의 compile·그래프 캡처가 모든 판을 멈추게 하므로). 권한을 놓은 뒤 진행 중인 판이 끝나면 내린다(대기 작업자가 GPU 메모리를 차지하지 않게, 2026-10-26 구현 결정). 모든 모델이 올라온 단계의 판만 배정한다(모델 서버가 없는 판이 T0 봇으로 도는 일을 막음). ③ **판 배정 권한**: 한 시점에 한 작업자만 새 판을 배정하고 그 판의 추론을 맡는다. 권한은 061 잠금(heartbeat·만료)을 id `eval-<job>`으로 쓴다. 더 높은 `--priority` 작업자가 오면 061-R14와 같이 양보를 요청한다. 권한을 쥔 작업자는 요청을 보면 새 판 배정을 멈추고 권한을 바로 놓은 뒤 진행 중인 판을 끝까지 마친다(판 도중에는 추론을 넘기지 않는다). 권한을 받은 작업자는 바로 새 판을 배정한다. 빈 자리가 없으면 hostd의 capacity 거부와 재시도로 기다린다. 권한이 없는 작업자는 10초마다 다시 시도한다. heartbeat는 권한을 쥔 동안 모델을 올리는 중에도 계속한다. ④ **판 상태**: 원장(모든 장비의 `games` 기록과 `inflight` 표)에서 계산한다. 최종 기록이 있으면 끝남이고, 살아 있는 작업자(진행 중 표를 수 초마다 다시 쓴다. 내용이 10분 동안 그대로면 죽은 것으로 본다, 061-R5와 같은 관측 만료)의 진행 중 표에 있으면 진행 중이고, 나머지는 남음이다. 죽은 작업자의 진행 중 판은 남음으로 돌아간다. 다시 하는 판은 브리지 미접속(020-R4), hostd 연결 끊김, 강제 정지(`Stopped`)이고, 같은 seed로 다시 돈다(`attempt` 증가, 이전 기록은 지우지 않고 `retry = true`로 남김). `Timeout`·게임 crash `Aborted`·`Rejected`(capacity 외)는 최종 기록이다. 한 판의 실패한 시도(재시도 대상 중 브리지 미접속·hostd 끊김. 사용자의 두 번째 Ctrl+C로 멈춘 `Stopped`는 세지 않는다)가 `max_attempts`(job 설정, 기본 **3**, 2026-10-26 사용자 결정 AMB-020-6)에 이르면 그 판은 마지막 기록으로 끝난 것으로 보고(결과 없음: 합친 기록에 `gave_up`, 그 시도의 세션은 요약에서 뺌) 경고를 남기며, 상태·요약에 포기한 판 수를 따로 적는다. 공용 드라이브의 기록·진행 중 표를 읽지 못하거나 권한 heartbeat가 공용 드라이브에 닿지 않으면 "모름"으로 보고(061-R13), 마지막 대기열을 유지한 채 새 판 배정만 멈춘다. 기록 쓰기는 몇 번 다시 시도하고, 판을 받은 뒤 어느 경로로 끝나도 결과 통지나 되돌림을 한다. 남은 판은 plan 순서로 배정하고, 앞 단계의 남은 판이 없을 때 다음 단계로 넘어간다. ⑤ **정지**: Ctrl+C(SIGINT. 학습 프로세스와 같은 처리라 SIGTERM은 받지 않는다) 한 번이면 마무리 정지다. 새 판을 받지 않고 권한·양보 요청을 놓은 뒤 진행 중인 판을 끝까지 기록하고 종료한다. 두 번째 Ctrl+C는 즉시 정지다. 진행 중인 판에 `StopGame`을 보내고 남음으로 돌린 뒤 종료한다. 세 번째는 그대로 프로세스를 끝낸다(멈춘 경우 대비). ⑥ **호스트 빠짐·재참여**: 작업자는 호스트 목록을 60초마다 다시 확인해 돌아온 호스트를 다시 쓴다. 진행 중에 hostd 연결이 끊긴 판은 남음으로 돌린다. `--local-hostd <addr>` 작업자는 첫 Ctrl+C에 어느 상태에서든 `<job>/leaving/<addr>`(파일 이름은 `:` 등을 `_`로 바꿈, 내용이 주소)를 쓰고, 배정자는 5초마다 이 표시를 읽어 그 호스트에 새 판을 보내지 않는다. 다시 시작하면 지운다. `--hostd`로 띄운 이 PC의 hostd는 작업자가 Ctrl+C가 닿지 않게 따로 띄우고, 끄기 전에 그 hostd에서 도는 판이 0이 될 때까지 기다린다(다른 작업자의 판일 수 있음, 두 번째 Ctrl+C면 기다리지 않음). 모든 작업자가 같은 hosts 파일을 읽으므로 호스트는 LAN 주소로 적는다. ⑦ **hostd 판 번호**: hostd에 보내는 `game_id`는 `시도 번호(0부터) × 10,000,000 + 단계 순번(1부터) × 100,000 + plan 판 번호`다(단계가 바뀌는 순간 두 작업자의 판이 hostd 작업 폴더·소켓 이름에서 겹치지 않고, 다시 한 판의 세션 기록이 이전 시도와 섞이지 않게). 장비별 기록은 이 번호로 쓰고, ⑨의 합친 기록은 plan 판 번호(`game_id`)와 이 번호(`hostd_id`)를 함께 남긴다. 단계 이름은 `stage`에 남긴다. ⑧ **추론 장비 기록**: `games` 기록마다 `infer_host`와 `backend`(정밀도는 032-R8)를 남긴다. ⑨ **상태·요약**: `python -m uranium eval-status --job <dir>`는 단계별 끝남·진행 중·남음·재시도 수와 작업자·권한 상태를 보이고, 단계별 `summary.md`(020-R13·R14 형식, 장비별 기록을 합침, 다시 한 판은 마지막 시도만, 추론 backend별 판 수 표 추가)를 다시 쓴다. hostd가 정상 종료로 보고했지만 브리지 세션 하나라도 끝을 보고하지 못한 판(판 도중 연결이 끊김: 호스트 네트워크, 클라이언트 crash)은 합친 기록에서 결과 없음(`Aborted`, `no_result = "session lost"`, 원래 결과는 `hostd_outcome`)으로 바꾸고 그 판의 세션은 요약에서 뺀다. 남은 쪽이 "상대 나감"으로 승리를 보고하기 때문이다. 다시 돌리지 않으며 summary에 그 판 수를 적는다(2026-10-27 사용자 결정) |
| 020-R16 | **MUST** 플레이어 이름(2026-10-26 사용자 결정): 리플레이에서 어느 정책이 어느 플레이어인지 알 수 있게, hostd는 `GameSpec.players[i].name`이 있으면 플레이어 이름을 `p<i>-<name>`으로, 없으면 `uranium-p<i>`로 붙인다. 스케줄러가 채우는 기본 이름은 모델이면 체크포인트의 run 폴더 이름(예: `core-abl-gtrxl-pt`), T0이면 `T0-<종족>`(시나리오 빌드면 `T0-<종족>-<빌드>`), OSS 봇이면 봇 이름이다. plan 플레이어의 `name`으로 바꿀 수 있다. BW 이름 길이 상한을 넘으면 `p<i>-` 뒤를 잘라 상한에 맞춘다(인덱스는 잘리지 않는다). 에이전트는 `p<i>-`와 `uranium-p<i>`에서 인덱스를 읽는다(020-R11) |
| 020-R17 | **MUST** 브리지 확장 변수(2026-10-31 사용자 결정, `reports/m6_action_audit.md` §11 U-1·U-2, 011-R23): hostd는 우리 브리지(`URANIUM_BRIDGE_LIB`)를 AI로 띄우는 대전 런처의 환경에만 `URANIUM_BRIDGE_EXT=1`을 넣는다. OSS 봇(T1) 런처의 환경에서는 이 변수를 지운다(hostd 자신의 환경에 있어도 넘기지 않는다). 두 런처는 같은 BWAPI 빌드를 쓰므로 BWAPI 패치 0016의 동작은 이 변수로만 갈린다. 재생 모드 런처(`StartReplay`)는 명령을 보내지 않으므로 넣지 않아도 된다 |

### 020-R3 타임아웃 규칙
| 상황 | 기준 | 처리 |
|---|---|---|
| 게임 길이 | 최대 게임 시간 60분(= 86,400 프레임) 초과 | **무승부** (보상 0, `TimeLimitDraw`) |
| 에이전트 무응답 | 별도 중단 기준 없음(2026-10-19 사용자 결정, C-9, AMB-020-4) | lockstep 학습은 에이전트 응답을 기다린다(001-R5). 멈춤이 OpenBW 드롭 시간(011-R18, 기본 10분)을 넘으면 클라이언트가 서로를 끊고, 두 세션이 모두 상대 퇴장으로 승리를 보고하면 `Aborted`로 처리한다(020-R13). 추론 배치가 5초를 넘으면 시각과 함께 로그를 남긴다(011 관측과 대응) |

### 020-R4 크래시 처리
| 상황 | 처리 |
|---|---|
| 런처 비정상 종료 | 에피소드를 `Aborted`로 표시하고 학습 데이터에서 제외 |
| 같은 설정이 연속 3회 크래시 | 해당 설정을 격리하고 경보 |

### 020-R11 plan 플레이어 종류
```toml
p0 = { kind = "t0", race = "terran" }                         # T0 스크립트 봇(021)
p1 = { kind = "oss", name = "iron", race = "terran" }          # T1 OSS 봇(021)
p0 = { kind = "model", checkpoint = "<dir>", sample = "argmax" }  # 모델 정책(테란 고정), sample = "argmax" | "stochastic"(050-R1)
```
- `checkpoint`가 상대 경로면 `URANIUM_SHARED_DIR` 기준으로 찾는다(061). 같은 체크포인트·샘플링 모드를 쓰는 세션은 추론 서버 하나를 공유한다.
- 실행 진입점(2026-10-23): 모델 플레이어가 있는 plan은 `python -m uranium run --plan <plan>`으로 돌린다(PyTorch 추론 스레드 + Rust 러너·에이전트 스레드, 001-R6). 스크립트 봇·OSS 봇만 있는 plan은 Rust 바이너리 `uranium-env run`으로도 돌릴 수 있고, 모델 플레이어가 있으면 Python 진입점을 안내하고 거부한다. 추론 백엔드는 `--backend <auto|cuda|rocm|cpu>`, 없으면 `.env` `URANIUM_AGENT_BACKEND`, 그것도 없으면 `auto`다(060-R1). (이력: `uranium-env run --backend <cuda|vulkan|flex>`, 기본 `vulkan`.)
- 추론 배치 파라미터(`max_batch`, `max_wait_ms`)는 CLI로 바꿀 수 있고 기본값은 측정해 정한다(TBD(M6)). lockstep이라 정확성에는 영향이 없고 처리량만 바뀐다. 정확성 제약은 OpenBW 드롭 시간(011-R18)뿐이다.

## 처리량 목표 (M3에서 실측)
| 지표 | 목표 |
|---|---|
| 동시 게임 수 | 호스트별 측정값(020-R10). M3 실측: 런처 수가 코어 수를 넘으면 브리지 p99가 10배 이상 늘어난다(`reports/m3.md`) |
| 평균 게임 길이 | 약 15분 = 21,600 프레임, 평균 delay 8프레임 기준 약 2,700 결정/게임 |
| 전체 결정 처리량 | TBD(M3). 추론 서버 한계가 병목일 것으로 예상([060](060-infra.md)) |

## Acceptance Criteria
- 020-AC1: 원격 게임 호스트 2대와 학습 호스트의 로컬 hostd(3대, 각자 측정한 동시 게임 수)로 T0·T1 혼합 200판(`config/plans/m3-remote.toml`)을 돌렸을 때 Aborted 비율이 1% 미만이다. 결과에는 호스트별·상대별 판당 시간을 기록한다. (2026-10-16 사용자 결정으로 "원격 2대·동시 16게임·1시간"에서 변경)
- 020-AC2: 같은 seed, 같은 상대, 같은 정책(결정론적 샘플링)으로 실행했을 때 결과가 재현된다. 상대는 결정론적인 T0·셀프플레이로 한정한다. OSS 봇은 내부 난수·시간 의존으로 재현되지 않으며 그대로 둔다(AMB-021-3).
- 020-AC3: hostd를 kill -9 한 뒤 재시작하면 고아 프로세스가 남지 않는다. 프로세스 그룹 정리로 보장한다.
- 020-AC4: `StartReplay`로 스트리밍한 RawFrame이 같은 리플레이의 저장 shard와 일치한다(040-AC6).
- 020-AC5: `uranium-env calibrate` 한 번으로 측정 파일이 만들어지고, `URANIUM_HOSTD_MAX_GAMES` 없이 재시작한 hostd가 그 값을 쓴다.
- 020-AC6 (2026-10-26, 020-R15): 실제 호스트에서 작은 평가 작업(단계 2개)을 돌리며 ① CUDA 작업자 마무리 정지(Ctrl+C 1회) → 진행 중인 판이 끝까지 기록되고 그 호스트에 새 판이 가지 않음 ② 대기 작업자가 권한을 받아 새 판부터 추론 ③ CUDA 작업자 재합류 → 대기 작업자가 양보하고 마무리 ④ 즉시 정지(Ctrl+C 2회) → 끊긴 판이 다시 배정됨 ⑤ 브리지 미접속 판이 다시 배정됨을 확인한다. 끝나면 모든 plan 판이 최종 기록을 정확히 하나씩(다시 한 판은 마지막 시도) 갖고, 기록의 추론 장비가 실제와 같다. 판 상태 계산·권한 넘김·번호 오프셋은 단위 테스트로도 검사한다. — **통과(2026-10-26)**: drill 56판·drill2 6판(`reports/m6_arch_ablation_pt.md` 4절), ⑤는 단위 테스트.
- 020-AC7 (2026-10-26, 020-R16): 모델 대 모델·모델 대 T0 게임의 리플레이를 `uranium-replay`로 읽으면 플레이어 이름이 `p<i>-<run 폴더 이름>`·`p<i>-T0-<종족>`이고, 정책 라우팅이 이름 형식과 무관하게 맞다(이름 자르기·두 형식 해석 단위 테스트 포함). — **통과(2026-10-26)**: drill-pair 리플레이 `p0-core-abl-gtrxl-pt`·`p1-arch-abl-terrain-pt`.

## Risks & Open Questions
- **slot 식별 (2026-10-17 실측)**: Windows 호스트(TCP LAN)에서 BWAPI 플레이어 id(`Hello.slot`)가 런처 순서와 다르게 배정됐다(T0 테란 대 T0 저그 4판 중 2판). 그래서 정책 라우팅(020-R11)과 시나리오 빌드(050)는 hostd가 붙이는 플레이어 이름 `uranium-p<i>`로 `GameSpec.players` 인덱스를 찾는다. 같은 게임의 두 세션이 승자를 다르게 보고한 판도 관측됐다. 원인(2026-10-29 확인, `reports/m6_bc_eval.md` 3절): 진 쪽이 섬멸 직전에 보낸 명령이 남아 있으면 이긴 쪽 OpenBW가 진 쪽 연결을 끊고, 진 쪽 프로세스가 이를 상대 드롭으로 처리해 자기 승리나 승자 없음을 보고한다. 진 쪽이 거의 매 프레임 명령하는 argmax 모델 판에서 나타난다(2026-10-29 평가 argmax 300판 중 90판). OpenBW 패치(011-R22)로 고쳤다. 판정은 020-R13을 따르고, 보고서는 그런 판을 "winner disagree"로 따로 센다.
- **summary.md 형식 (2026-10-18 사용자 결정, 구현 메모)**: 모델 대 모델 대전(032-R17 맞대결)을 읽을 수 있게 `uranium-env` summary를 바꾼다. 대전 동작은 바꾸지 않는다. ① matchup 표의 p0·p1 모두 모델이면 `model <core> \`<run 폴더>/<체크포인트 폴더>\` (샘플링)`과 정책 버전(020-R5 `policy` 값)을 표시한다. core는 체크포인트 `meta.json`에서 읽는다. ② 서로 다른 두 모델 정책의 게임은 승자 세션의 `policy`로 p0·p1 승을 가른다(이전에는 "mirror decided"로만 셌다). 같은 정책끼리는 그대로 mirror다. ③ "Per policy (slot swaps combined)" 표: (정책, 상대)별로 슬롯 교대 matchup을 합친 승·패·무, Wilson 95%(050-R2), dropped를 맵별과 전체(`all`)로 낸다. ④ "Model sessions" 표에 core·run 열을 더한다. 테스트 `report::tests::model_vs_model_summary_per_policy_032_r17`.


- LAN 모드에서 두 런처의 프레임 동기 오버헤드를 측정해야 한다. 셀프플레이는 같은 에이전트 서버로 두 브리지가 접속하는 구조가 된다.
- 원격 호스트의 MPQ와 맵 배포는 사용자가 직접 준비한다. hostd는 시작 시 파일 해시를 검증한다.

- **호스트 구성 (확정 2026-10-15, AMB-020-1)**
  - 학습 호스트 외에 **게임 전용 Linux PC 2대 이상**을 쓴다.
  - 호스트별 코어 수와 동시 게임 수는 hostd `ListCapabilities`로 자동 감지하고, `config/hosts.toml`로 덮어쓸 수 있다.
  - Linux 학습 호스트(ROCm)의 로컬 hostd는 기본 비활성이다. 그 호스트를 learner로 쓰는 동안에는 추론과 학습을 전담한다.
  - **Windows CUDA 호스트**는 학습·추론과 함께 로컬 hostd로 게임도 돌린다(2026-10-16 사용자 결정). 원격 Linux 게임 호스트와 함께 `URANIUM_HOSTS_FILE`에 둘 다 등록하고, 에이전트 서버는 원격 브리지가 접속할 수 있는 주소로 listen한다(`URANIUM_AGENT_LISTEN`, `URANIUM_AGENT_PUBLIC_ADDR`). 학습 중 로컬 동시 게임 수는 hosts 파일의 `max_games`로 줄일 수 있으며, 값은 학습 부하를 측정해 정한다(TBD(M5)).
  - 020-AC1은 원격 2대 + 학습 호스트 구성 200판으로 판정한다(2026-10-16 변경).

- **시간 초과 처리 (확정 2026-10-15, AMB-020-2)**
  - 60분(86,400 프레임)에 게임을 중단하고 **무승부**로 처리한다. 보상은 0이고 점수 판정은 하지 않는다.
  - `EndReason::TimeLimitDraw`로 기록한다.
  - 2026-10-15에 점수 판정안에서 무승부로 변경했다.
