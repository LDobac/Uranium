# 043 — Ladder Replay Collector

- **Status**: Approved (2026-10-16, AMB-043-1~8 확정. 043-AC1 실측 반영. 랭크 A 이상 = bucket ≥ 6)
- **Related**: [040](040-replay-dataset.md), [061](061-shared-storage.md), [001](001-system-architecture.md), [013](013-version-mapping.md)

## Purpose
SC:R 래더(매치메이킹) 경기 리플레이를 자동으로 모아 040 파이프라인의 대상 맵(S2) 데이터를 늘린다.
공개 사이트(cwal.gg 등)는 robots.txt가 자동 수집을 막고 있다(`reports/dataset_sources.md`). 그래서 같은 데이터의 원천인 **로그인된 SC:R 클라이언트의 로컬 웹 API**를 직접 쓴다.

## 배경: SC:R 로컬 웹 API (비공식)
- 배틀넷에 로그인한 SC:R 클라이언트는 `127.0.0.1`의 임의 포트에 HTTP 서버를 연다. 공식 문서는 없고, 커뮤니티가 리버스 엔지니어링으로 정리했다(bw-web-api, bw-web-api-rs, MIT).
- 엔드포인트(2026-10-16 실측, `reports/m4_ladder_probe.md`). 접두사는 `/web-api`다:

  | 용도 | 경로 |
  |---|---|
  | 리더보드 목록 | `/web-api/v1/leaderboard` |
  | 리더보드 순위 | `/web-api/v1/leaderboard/{leaderboard_id}?offset=&length=` (표 형식 `columns`+`rows`) |
  | 프로필 (최근 리플레이 목록) | `/web-api/v2/aurora-profile-by-toon/{toon}/{gateway}?request_flags=scr_profile` → `replays[]` 최근 25개, `link`가 `MM-<UUID>`면 래더 |
  | 경기 정보 + 리플레이 링크 | `/web-api/v1/matchmaker-gameinfo-playerinfo/{link}` → `replays[]`(양측 각 1개) `md5`, `url` |
  | 플레이어 래더 전적 | `/web-api/v1/matchmaker-gameinfo-by-toon/{toon}/{gateway}/{gamemode}/{season}` (실측한 경기들은 playerinfo `replays`가 비어 있었다. 수집 경로로 쓰지 않는다) |

- 수집 경로: 리더보드 → 대상 플레이어 프로필의 `replays[]`(래더, `replay_humans == 2`) → playerinfo → `url`(`https://storage.googleapis.com`, 인증 없음). md5를 API 값과 대조한다.
- GCS 리플레이 파일은 약 30일(커뮤니티 보고 17~46일) 뒤 사라진다. 프로필 목록도 최근 25개뿐이다. 그래서 수집기는 주기적으로 돌아야 한다.
- 응답 스키마는 클라이언트 버전마다 바뀔 수 있다(043-R9·R10).

## Scope & Non-goals
- Scope:
  - 로컬 API 포트 탐지
  - 리더보드 → 플레이어 → 래더 전적 → 경기 → 리플레이 순서의 수집
  - 중복 제거, 재개, 속도 제한
  - `raw/<src>/` 저장과 수집 기록(records)
- Non-goals:
  - 공개 사이트 크롤링
  - 클라이언트 메모리 읽기·수정, 패킷 가로채기
  - 게임 플레이 자동화(래더 대기열 참가 등)
  - 리플레이 재배포, 웹 UI
  - 리플레이 파싱·필터(040의 `uranium-replay`가 맡는다)

## Requirements
| ID | 요구사항 |
|---|---|
| 043-R1 | **MUST** 새 crate `uranium-ladder`(바이너리 `uranium-ladder`)로 구현한다. SC:R이 도는 Windows x86_64에서 빌드·실행되어야 한다. Linux 빌드도 깨지지 않게 한다(테스트용) |
| 043-R2 | **MUST** 클라이언트가 이미 연 로컬 HTTP API만 호출한다. 클라이언트 프로세스 메모리 접근, 인증 토큰 추출, 배틀넷 서버 직접 호출은 하지 않는다. 리플레이 파일은 API가 준 링크로만 받는다 |
| 043-R3 | **MUST** 포트는 `URANIUM_SCR_API_PORT`가 있으면 그 값을 쓰고, 없으면 `StarCraft` 프로세스의 LISTEN 포트에서 탐지한다. 후보가 여러 개면 리더보드 목록 응답이 유효한 포트를 고른다 |
| 043-R4 | **MUST** 속도·양 한도(확정 2026-10-16, AMB-043-4): 로컬 API 호출과 리플레이 다운로드를 합쳐 **0.5 req/s 이하**(`URANIUM_LADDER_RPS`, 기본 0.5). 리플레이 다운로드는 **하루(로컬 시각 기준) 1,000개 이하**(`URANIUM_LADDER_DAILY_MAX`, 기본 1000). 한도에 닿으면 그날 수집을 마치고 종료한다(재실행 시 043-R6으로 이어감). 총량 상한은 두지 않고 디스크 예산(060-R7)만 지킨다 |
| 043-R5 | **MUST** 수집 대상은 계획 파일(`config/ladder/<plan>.toml`)로 지정한다. 기본값(확정 2026-10-16, AMB-043-3): ① 게이트웨이 **Korea(30) + Asia(45)**(2026-10-16, AMB-043-7) ② 추적 대상 플레이어는 각 게이트웨이 현재 시즌 1v1 리더보드에서 **랭크 문자 A 이상**(S·A). API에는 문자가 없고 `bucket`(1–7)과 시즌별 경계값(`matchmaked_season_buckets`, 시즌 20: 0, 1163, 1397, 1561, 1742, 2028, 2244, 9999)만 있다. 경계값 7구간이 랭크 F~S 7단계와 맞으므로 **F=1 … A=6, S=7로 보고 `min_bucket = 6`** 을 쓴다(2026-10-16 사용자 결정: 사용자 랭크가 없어 대조 없이 추정대로 확정). 계획 파일에 `min_bucket`이 없으면 `collect`는 실행을 거부한다 ③ 매치업 **전체**(테란이 없는 경기도 받는다. S1·평가 분석용, 040 필터가 거른다) ④ 맵 **래더 맵 전체** ⑤ 시즌: 과거 시즌은 받을 수 없으므로(배경 절) 조건에서 뺀다. 대신 **하루 1회 `collect`를 실행해 최근 경기를 누적**한다(2026-10-16, AMB-043-8). 실행 수단(수동, Windows 작업 스케줄러)은 범위 밖이다. 상대 플레이어의 랭크는 제한하지 않는다. 사람 1:1 래더 경기만 받는다(`link`가 `MM-`, `replay_humans == 2`) |
| 043-R6 | **MUST** 중복 제거와 재개: 경기 ID 단위로 기록한다. 이미 받은 경기, 이미 실패로 기록된 경기, `raw/` 전체 SHA-256과 같은 파일은 다시 받지 않는다. 중단 후 재실행하면 이어서 진행한다 |
| 043-R7 | **MUST** 저장: 리플레이는 `<URANIUM_SHARED_DIR>/datasets/replays/raw/<src>/`(040-R13)에 둔다. 파일명은 경기 ID의 해시로 정하고 플레이어 이름은 쓰지 않는다. `src`는 **`scr_ladder`**(확정 2026-10-16, AMB-043-5, 040-R14에 추가). 리플레이 원본은 배틀태그·채팅을 포함한 그대로 보관한다(AMB-043-6) |
| 043-R8 | **MUST** 수집 기록(`<URANIUM_SHARED_DIR>/datasets/replays/ladder/collect.jsonl`. `build` 기록과 섞이지 않게 `<src>*.jsonl` 이름을 쓰지 않는다)에 경기마다 경기 ID, SHA-256, 게이트웨이, 시즌, 맵 이름, 양측 종족, 양측 toon(평문)·MMR·랭크, 기록된 승자, 수집 시각, 상태(ok/failed/skipped와 사유)를 남긴다. 040 파이프라인은 랭크·MMR로 S2 "상위 래더" 기준을 적용하고, 승자는 040-R8 ④ 디싱크 판정에 쓴다. 식별자 평문 기록은 공용 드라이브 안에서만 허용한다(확정 2026-10-16, AMB-043-6). 이 파일은 git에 넣지 않는다 |
| 043-R9 | **MUST** API 응답 스키마가 예상과 다르면(필드 누락, 형식 변경) 추측하지 말고 해당 경기를 `failed: schema`로 기록한다. 연속 실패가 임계값(기본 20)을 넘으면 중단하고 원인을 보고한다 |
| 043-R10 | **MUST** 응답 스키마 정의(필드 이름, 경로 접두사)는 코드에 흩뿌리지 않고 한 모듈의 타입·상수로 모은다. 클라이언트 업데이트로 바뀌면 그 모듈만 고친다(013-R1 정신) |
| 043-R11 | **MUST** 콘솔 로그, 리포트, 커밋, 테스트 픽스처에 IP, 호스트 이름, 배틀태그, toon, 플레이어 이름, 계정 정보를 쓰지 않는다(001-R10, 040-R14). 평문 식별자는 043-R8의 수집 기록에만 둔다 |
| 043-R13 | **MUST** 실행 환경(확정 2026-10-16, AMB-043-2): 사용자 **본계정**으로 로그인된 SC:R이 떠 있는 Windows PC. 수집기는 게임 조작을 하지 않는다 |
| 043-R12 | **SHOULD** 테스트는 실제 클라이언트 없이 저장된 응답 픽스처(개인정보 제거)와 가짜 HTTP 서버로 한다 |

## Interfaces
```
uranium-ladder probe [단계 인자]             # 포트 탐지, 단계별 원본 저장, 응답 형태 요약(값은 가림) → 043-AC1
uranium-ladder shape <json> [--pointer /p]   # 저장한 응답의 형태(043-R11)
uranium-ladder collect --plan config/ladder/<plan>.toml [--limit N] [--dry-run]   # 하루 1회
uranium-ladder status  [--out <md>]          # 수집 기록 집계(상태별·맵별·매치업별 개수)
```
- 수집 결과는 기존 명령으로 이어 처리한다: `uranium-replay build <raw/<src>> --src <src> --records <jsonl>`(040).
- 환경변수(`.env.example`에 추가): `URANIUM_SCR_API_PORT`(선택), `URANIUM_LADDER_RPS`(기본 0.5), `URANIUM_LADDER_DAILY_MAX`(기본 1000).

## Acceptance Criteria
- 043-AC1: (스파이크) 로그인된 SC:R에서 `probe`가 포트를 탐지하고, 리더보드·전적·경기 정보 응답을 받아 리플레이 1개를 내려받는다. 결과(경로 접두사, 응답 필드, 링크 형식, 필요 헤더)를 `reports/m4_ladder_probe.md`에 개인정보 없이 기록한다.
- 043-AC2: 픽스처 기반 테스트로 전적 → 경기 → 다운로드 흐름, 중복 제거, 재개, 스키마 불일치 처리를 검증한다.
- 043-AC3: 실제 수집으로 계획 범위의 리플레이 100개 이상을 받는다. 그 결과를 `uranium-replay build`에 넣어 040-AC5 형식으로 디싱크 비율을 리포트한다.
- 043-AC4: 속도 제한 준수: 수집 로그에서 초당 요청 수가 설정값을 넘지 않는다.

## Risks & Open Questions
- **데이터 정책 예외 (확정 2026-10-16, AMB-043-1)**
  - 040의 정책(AMB-040-3)은 "사이트 이용약관·robots.txt 준수"다.
  - Blizzard EULA에는 승인되지 않은 서드파티 프로그램으로 데이터를 수집하는 행위를 금지하는 조항이 있고, 로컬 API 자동 호출이 여기에 걸릴 수 있다(해석은 불확실).
  - 사용자는 이 위험을 알고 043 범위의 수집을 040 정책의 **예외로 허용**했다. 내부 학습 전용·재배포 금지는 그대로다.
- 게이트웨이(확정 2026-10-16, AMB-043-7): API에서 30 Korea와 45 Asia는 별도 게이트웨이이며 둘 다 수집한다.
- 과거 시즌(확정 2026-10-16, AMB-043-8): 링크는 프로필의 최근 25경기에서만 나오고 파일은 약 30일 뒤 사라진다. 그래서 하루 1회 주기 수집으로 바꿨다. 25경기보다 많이 두는 플레이어의 일부 경기는 놓칠 수 있다.
- 결과 판정: playerinfo의 `game_result`는 플레이어마다 보고가 다르다(한쪽은 `Win`/`Loss`, 다른 쪽은 `Disconnect`/`Undecided`인 경우가 있다). 기록에는 양쪽 보고를 모두 남기고, 040-R8 ④에는 두 보고가 `Win`/`Loss`로 일치할 때만 승자를 쓴다.
- **계정 제재 위험**: 사용자 본계정을 쓴다(AMB-043-2). 043-R4의 보수적 한도(0.5 req/s, 일 1,000개)로 트래픽을 줄이지만 위험이 없어지지는 않는다. 클라이언트가 오류·경고를 주면 즉시 멈추고 보고한다(043-R9).
- 클라이언트 업데이트로 API가 바뀌거나 막힐 수 있다. 043-R9·R10으로 영향 범위를 한 모듈로 줄인다.
- 리플레이 링크가 클라이언트 세션 쿠키·헤더를 요구할 수 있다. 그 경우 043-R2 범위(API가 준 정보만 사용) 안에서 가능한지 스파이크에서 확인하고, 안 되면 사용자에게 보고한다.
