# 043 — Ladder Replay Collector

- **Status**: Draft (2026-10-16)
- **Related**: [040](040-replay-dataset.md), [061](061-shared-storage.md), [001](001-system-architecture.md), [013](013-version-mapping.md)

## Purpose
SC:R 래더(매치메이킹) 경기 리플레이를 자동으로 모아 040 파이프라인의 대상 맵(S2) 데이터를 늘린다.
공개 사이트(cwal.gg 등)는 robots.txt가 자동 수집을 막고 있다(`reports/dataset_sources.md`). 그래서 같은 데이터의 원천인 **로그인된 SC:R 클라이언트의 로컬 웹 API**를 직접 쓴다.

## 배경: SC:R 로컬 웹 API (비공식)
- 배틀넷에 로그인한 SC:R 클라이언트는 `127.0.0.1`의 임의 포트에 HTTP 서버를 연다. 공식 문서는 없고, 커뮤니티가 리버스 엔지니어링으로 정리했다(bw-web-api, bw-web-api-rs, MIT).
- 이 spec에서 쓰는 엔드포인트:

  | 용도 | 경로 |
  |---|---|
  | 리더보드 목록 | `/v1/leaderboard` |
  | 리더보드 순위 | `/v1/leaderboard/{ladder}?offset=&length=` |
  | 플레이어 래더 전적 | `/v1/matchmaker-gameinfo-by-toon/{toon}/{gateway}/{ladder}/{season}` |
  | 경기 정보 + 리플레이 링크 | `/v1/matchmaker-gameinfo-playerinfo/{match_id}` |
  | 프로필 | `/v2/aurora-profile-by-toon/{toon}/{gateway}?request_flags=scr_profile` |

- 경로는 bw-web-api README 표기를 따랐다. 실제 경로 앞에 접두사(예: `/web-api`)가 붙을 수 있다. 경로 접두사, 포트, 응답 스키마, 리플레이 링크 호스트, 필요한 헤더는 클라이언트 버전마다 달라질 수 있다. 정확한 값은 스파이크(043-AC1)로 실측한다: TBD(M4).

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
| 043-R4 | **MUST** 속도 제한: 로컬 API와 리플레이 다운로드 각각 기본 1 req/s 이하(`URANIUM_LADDER_RPS`). 상한과 일일 한도는 AMB-043-4로 정한다 |
| 043-R5 | **MUST** 수집 대상(게이트웨이, 순위/MMR 기준, 매치업, 맵, 시즌)은 계획 파일(`config/ladder/<plan>.toml`)로 지정한다. 기본값은 AMB-043-3으로 정한다 |
| 043-R6 | **MUST** 중복 제거와 재개: 경기 ID 단위로 기록한다. 이미 받은 경기, 이미 실패로 기록된 경기, `raw/` 전체 SHA-256과 같은 파일은 다시 받지 않는다. 중단 후 재실행하면 이어서 진행한다 |
| 043-R7 | **MUST** 저장: 리플레이는 `<URANIUM_SHARED_DIR>/datasets/replays/raw/<src>/`(040-R13)에 둔다. 파일명은 경기 ID의 해시로 정하고 플레이어 이름은 쓰지 않는다. `src` 값은 AMB-043-5로 정한다 |
| 043-R8 | **MUST** 수집 기록(`records.jsonl`, `raw/<src>/` 옆)에 경기마다 경기 ID 해시, SHA-256, 게이트웨이, 시즌, 맵 이름, 양측 종족, 양측 MMR/순위, 기록된 승자, 수집 시각, 상태(ok/failed/skipped와 사유)를 남긴다. 040 파이프라인은 MMR·순위로 S2 "상위 래더" 기준을 적용하고, 승자는 040-R8 ④ 디싱크 판정에 쓴다. 플레이어 식별자 저장 범위는 AMB-043-6으로 정한다 |
| 043-R9 | **MUST** API 응답 스키마가 예상과 다르면(필드 누락, 형식 변경) 추측하지 말고 해당 경기를 `failed: schema`로 기록한다. 연속 실패가 임계값(기본 20)을 넘으면 중단하고 원인을 보고한다 |
| 043-R10 | **MUST** 응답 스키마 정의(필드 이름, 경로 접두사)는 코드에 흩뿌리지 않고 한 모듈의 타입·상수로 모은다. 클라이언트 업데이트로 바뀌면 그 모듈만 고친다(013-R1 정신) |
| 043-R11 | **MUST** 로그와 리포트에 IP, 포트 외 호스트 정보, 배틀태그, 플레이어 이름, 계정 정보를 쓰지 않는다(001-R10, 040-R14) |
| 043-R12 | **SHOULD** 테스트는 실제 클라이언트 없이 저장된 응답 픽스처(개인정보 제거)와 가짜 HTTP 서버로 한다 |

## Interfaces
```
uranium-ladder probe                         # 포트 탐지, 엔드포인트별 응답 형태 요약(값은 가림) → 043-AC1
uranium-ladder collect --plan config/ladder/<plan>.toml [--limit N] [--dry-run]
uranium-ladder status  [--src <src>]         # 수집 기록 집계(상태별·맵별·매치업별 개수)
```
- 수집 결과는 기존 명령으로 이어 처리한다: `uranium-replay build <raw/<src>> --src <src> --records <jsonl>`(040).
- 환경변수(`.env.example`에 추가): `URANIUM_SCR_API_PORT`(선택), `URANIUM_LADDER_RPS`(기본 1).

## Acceptance Criteria
- 043-AC1: (스파이크) 로그인된 SC:R에서 `probe`가 포트를 탐지하고, 리더보드·전적·경기 정보 응답을 받아 리플레이 1개를 내려받는다. 결과(경로 접두사, 응답 필드, 링크 형식, 필요 헤더)를 `reports/m4_ladder_probe.md`에 개인정보 없이 기록한다.
- 043-AC2: 픽스처 기반 테스트로 전적 → 경기 → 다운로드 흐름, 중복 제거, 재개, 스키마 불일치 처리를 검증한다.
- 043-AC3: 실제 수집으로 계획 범위의 리플레이 100개 이상을 받는다. 그 결과를 `uranium-replay build`에 넣어 040-AC5 형식으로 디싱크 비율을 리포트한다.
- 043-AC4: 속도 제한 준수: 수집 로그에서 초당 요청 수가 설정값을 넘지 않는다.

## Risks & Open Questions
- [SPEC AMBIGUOUS] AMB-043-1 **데이터 정책 예외**
  - 040의 확정 정책(AMB-040-3)은 "사이트 이용약관·robots.txt 준수"다.
  - Blizzard EULA에는 승인되지 않은 서드파티 프로그램으로 데이터를 수집하는 행위를 금지하는 조항이 있다. 로컬 API를 자동 호출하는 것이 여기에 걸릴 수 있다(해석은 불확실).
  - 이 수집을 040 정책의 예외로 허용할지 사용자가 결정해야 한다. 허용 범위는 내부 학습 전용·재배포 금지를 유지한다.
- [SPEC AMBIGUOUS] AMB-043-2 **실행 PC와 계정**
  - 로그인된 SC:R이 계속 떠 있어야 한다. 어느 PC에서 돌릴지, 어떤 배틀넷 계정으로 돌릴지(본계정/별도 계정) 정해야 한다.
  - 계정 제재 위험은 계정 선택과 직결된다.
- [SPEC AMBIGUOUS] AMB-043-3 **수집 범위 기본값**
  - 게이트웨이: 전체 또는 특정 지역
  - 대상 플레이어: 리더보드 상위 N명 또는 MMR 하한
  - 매치업: 테란 포함 경기만 또는 전체
  - 맵: 투혼·서킷브레이커 계열만 또는 래더 맵 전체
  - 시즌: 현재 시즌만 또는 과거 시즌까지
- [SPEC AMBIGUOUS] AMB-043-4 **속도·양 한도**: 기본 1 req/s를 유지할지, 일일 다운로드 한도, 총 수집량 상한(디스크 예산 060-R7 안에서).
- [SPEC AMBIGUOUS] AMB-043-5 **`src` 태그**: 040-R14의 기존 `scr_public`을 쓸지, 새 태그 `scr_ladder`를 추가할지(040-R14 개정 필요).
- [SPEC AMBIGUOUS] AMB-043-6 **플레이어 식별자 보관**: 래더 경기라 리플레이 자체에 배틀태그·채팅이 들어 있다.
  - 수집 기록(records)에 toon/게이트웨이 같은 식별자를 남길지(플레이어 단위 중복 제거·추적용), 해시로만 남길지.
  - 리플레이 원본의 채팅을 그대로 보관할지.
- 클라이언트 업데이트로 API가 바뀌거나 막힐 수 있다. 043-R9·R10으로 영향 범위를 한 모듈로 줄인다.
- 리플레이 링크가 클라이언트 세션 쿠키·헤더를 요구할 수 있다. 그 경우 043-R2 범위(API가 준 정보만 사용) 안에서 가능한지 스파이크에서 확인하고, 안 되면 사용자에게 보고한다.
