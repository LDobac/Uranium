# M4 래더 수집 스파이크 (043-AC1)

- 일자: 2026-10-16
- 환경: Windows PC, 사용자 본계정으로 로그인한 SC:R, `uranium-ladder probe` (debug 빌드)
- 속도: 모든 요청 0.5 req/s(043-R4). 이번 스파이크의 다운로드는 1개다.
- 이 리포트에는 형태만 적는다(043-R11). 원본 응답은 git 밖(작업 임시 폴더)에 두었다.

## 결과 요약
| 단계 | 경로 | 결과 |
|---|---|---|
| 포트 탐지 | `tasklist` + `netstat -ano` | 성공. LISTEN 포트 1개, 관리자 권한 불필요 |
| 경로 접두사 | `/v1/leaderboard` | 접두사 없음은 404, **`/web-api`** 는 200 |
| 리더보드 목록 | `/web-api/v1/leaderboard` | 200, 24KB. `gamemodes`(1=1v1), `gateways`, `leaderboards`(114개) |
| 게이트웨이 | `/web-api/v1/gateway` | 200. 10 U.S. West, 11 U.S. East, 20 Europe, **30 Korea**, **45 Asia** (각각 별도) |
| 리더보드 페이지 | `/web-api/v1/leaderboard/{id}?offset=0&length=100` | 200. 표 형식 `columns` 13개 + `rows`. Korea 1v1 현재 시즌(season_id 20)은 리더보드 12988 |
| 전적 | `/web-api/v1/matchmaker-gameinfo-by-toon/{toon}/{gw}/{gamemode}/{season}` | 200, 16경기. 각 경기는 `MM-<UUID>` 키 아래 `match_created`(센티널 u64 max), `players[2]` |
| 경기 정보 (전적의 경기) | `/web-api/v1/matchmaker-gameinfo-playerinfo/{MM-…}` | 200이지만 **4경기 모두 `replays: []`** |
| 프로필 | `/web-api/v2/aurora-profile-by-toon/{toon}/{gw}?request_flags=scr_profile` | 200, 150KB. `replays[]` 25개(래더 `MM-` 23개, 사설 방 숫자 2개). 각 항목은 `attributes`, `create_time`(int, 일부는 u32 max 센티널), `link`. url·md5는 없음 |
| 경기 정보 (프로필의 link) | 같은 playerinfo 경로 | 200. **`replays[]` 2개(양측 플레이어 각 1개)**, 각 항목에 `md5`(32), `url`(129자) |
| 다운로드 | `url` | `https://storage.googleapis.com` 200, 42KB, 인증 없음. md5가 API 값과 일치. `uranium-replay inspect` 파싱 성공(`Modern121`) |

- 로컬 API 응답에는 HTTP 헤더가 거의 없다(이름 목록이 비어 있음).
- 받은 리플레이 1개는 11프레임에서 끝난 경기(초반 이탈)였다. 040 필터(5분 미만 제외)에서 빠진다.

## 응답 형태 (발췌)
- 리더보드 `columns`: `rank, last_rank, gateway_id, points, wins, losses, disconnects, toon, battletag, avatar, feature_stat(주종족), rating, bucket`
- 프로필 `replays[i].attributes`: `game_creator, game_id, game_name, game_save_id, game_speed, game_sub_type, game_type, map_era, map_height, map_title, map_width, replay_description, replay_humans, replay_map_number, replay_max_players, replay_min_players, replay_opponents, replay_player_names, replay_player_races, replay_player_types, replay_result`
- playerinfo `replays[i]`: 위 `attributes` + `create_time`, `link`(빈 문자열), `md5`, `url`
- playerinfo 플레이어 항목: `game_info.attributes`(map_name, map_md5, map_file_name 등), `game_result`, `is_winner`, `score{base, bucket_new, bucket_old, delta, win_streak}`, `info_attributes`

## spec 043과 다른 점 (사용자 확인 필요)
1. **리플레이 링크의 출처는 프로필의 최근 `replays[]`(25개)다.** 전적 경로의 경기는 `replays`가 비어 있었다. 커뮤니티 보고(screpdb #341)에 따르면 GCS 파일은 17~46일(아마 30일) 뒤 404가 된다. 그래서 **과거 시즌 수집은 사실상 불가능**하고, 최근 경기를 계속 모으는 방식이 된다.
2. **API에 랭크 문자(S/A/…)가 없다.** `bucket` 숫자만 있다(Korea 상위 100명은 모두 7). 랭크 A 이상을 bucket 값으로 바꾸는 대응표는 실측해야 한다: TBD(M4).
3. **게이트웨이 30 Korea와 45 Asia가 별도다.** "아시아(한국)만"이 어느 쪽인지 확인이 필요하다.

## `collect` 검증 실행 (2026-10-16)
- 계획: Korea + Asia, bucket 7(S)만, `--limit` 5·3·4로 세 번. 랭크 A 이상(`min_bucket = 6`) 확정 전의 안전한 부분집합이다.
- 추적 대상(S): Korea 369명, Asia 40명.
- 결과: 리플레이 12개 저장(ok 12, skipped 0, failed 0). 모두 `uranium-replay inspect`로 파싱된다(`Modern121`).
  - 길이 5분 이상은 3개다. 나머지는 초반 이탈 경기라 040 필터에서 빠진다. 실제로 쓸 수 있는 비율은 약 1/4로 예상된다.
  - 매치업: TvT 4, TvP 3, TvZ 2, ZvZ 2, 기타 1. 맵: KnockOut 4, Backrooms 2, Radeon 2, 투혼 2, 기타 2.
- 첫 실행에서 `raw/` 인덱스 생성(80,630개 SHA-256)에 약 6분 걸렸다. 이후 실행은 캐시 덕분에 전체 40초대다.
- 실측 중 발견해 고친 점:
  - 일부 프로필 응답에 멀티바이트 중간에서 잘린 이름(유효하지 않은 UTF-8)이 있어 파싱이 실패했다(처음 3명 중 2명). 손실 허용 디코딩(U+FFFD)으로 처리한다.
  - `info_attributes.race`는 고른 종족이라 `Random`일 수 있다. 실제 종족은 본인 `game_result`의 `attributes.race`에서 읽는다.
- 속도 추정(0.5 req/s): 프로필 1회 + 새 경기당 2회 요청. A 이상 수백~천여 명을 하루에 한 바퀴 돌면 수십 분~2시간 정도 걸린다. 다운로드는 하루 1,000개에서 멈춘다.

## 커뮤니티 참고
- screpdb #341: playerinfo는 참가하지 않은 경기의 URL도 준다. GCS 파일 수명은 약 30일, 메타데이터는 46일 이상 남는다. HTTP 200에 `Internal error` 본문이 오는 일시 오류가 있다.
- scfingerprint #59: link가 `MM-…`이면 래더, 숫자면 사설 방이다. `replay_humans == 2`가 사람 1:1이다.

## 043 수용 기준 현황 (2026-10-17)
| AC | 상태 |
|---|---|
| 043-AC1 스파이크 | 완료(이 리포트) |
| 043-AC2 테스트 | 완료 |
| 043-AC3 100개 이상 수집 → build → 디싱크 리포트 | 완료: 2,000개 수집, 전체 build로 재생 1,052시점 디싱크 15.2%, 디싱크 10% 초과 맵 제외 후 5.6%(`reports/m4_scr_desync.md`, 040-R16) |
| 043-AC4 속도 제한 준수 | 매일 수집 로그로 확인 중 |

- 수집분의 맵: 현재 래더 맵 풀 9종(투혼 1.4 포함). 서킷브레이커는 없다.
- SC:R 최근 맵(CHK VER 64·206, `STRx`)과 opcode 0x5d는 변환기에서 처리한다(013 개정, `f10ff57`).
