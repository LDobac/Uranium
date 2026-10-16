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

## 커뮤니티 참고
- screpdb #341: playerinfo는 참가하지 않은 경기의 URL도 준다. GCS 파일 수명은 약 30일, 메타데이터는 46일 이상 남는다. HTTP 200에 `Internal error` 본문이 오는 일시 오류가 있다.
- scfingerprint #59: link가 `MM-…`이면 래더, 숫자면 사설 방이다. `replay_humans == 2`가 사람 1:1이다.
