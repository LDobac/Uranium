# 리플레이 출처 (040-R14, 내부 기록)

- 일자: 2026-10-16
- 데이터 정책(040, AMB-040-3):
  - 내부 학습 전용이며 재배포하지 않는다.
  - 자동 수집은 robots.txt와 약관이 허용하는 곳에서만 1 req/s 이하로 한다.
  - 그 밖의 출처는 링크만 남기고 사용자가 직접 받는다(2026-10-16 사용자 결정).
  - 예외: SC:R 로컬 웹 API 래더 수집(`scr_ladder`, spec 043)은 EULA 해석 위험을 감수하고 허용한다(AMB-043-1). 속도는 0.5 req/s, 하루 1,000개다.
- 받은 파일은 `<URANIUM_SHARED_DIR>/datasets/replays/raw/<src>/`에 둔다(040-R13). 파일명과 플레이어 이름은 리포트에 쓰지 않는다.

## 확보한 출처
| src | 내용 | 수량 | 라이선스 / 근거 | 확보 방법 |
|---|---|---|---|---|
| `stardata` | STARDATA 원본 1.16.1 리플레이 | 65,646 | BSD + 특허 허여 (facebookresearch/StarData) | `original_replays.tar.gz` 다운로드(5.3GB, SHA-256 `037cb688…642a`) |
| `personal` | 사용자 본인 SC:R 리플레이 | 576 (테란 1:1 5분 이상 140) | 사용자 소유 | 사용자 PC의 리플레이 폴더에서 복사 |
| `community` | StarCraft AI wiki 리플레이 묶음 4종 (`raw/community/<묶음>/`) | 14,406 (손상 79) | 연구용 공개 묶음, 명시 라이선스 없음. 내부 학습 전용 | 묶음 파일 직접 다운로드(크롤링 아님), 각 `.sha256` 기록 |
| `scr_ladder` | SC:R 래더 1:1 (Korea·Asia, 추적 대상 랭크 A 이상, 매치업·맵 전체) | 12 (2026-10-16 검증 실행, 매일 누적) | 내부 학습 전용, 040 정책 예외(AMB-043-1) | `uranium-ladder collect`(본계정 SC:R 로컬 API → 공개 GCS 링크). 기록은 `ladder/collect.jsonl`(`reports/m4_ladder_probe.md`) |

### community 묶음 상세 (2026-10-16)
| 묶음 | 원본 수 | STARDATA와 중복 | 새 리플레이 | 파이프라인 표본 결과 |
|---|---|---|---|---|
| Synnaeve TL·GG·ICCup gosu (R3) | 7,708 | 4,458 | 3,248 | 앞쪽 300개는 중복·필터. 전체 실행에서 처리 |
| Webber (R1) | 5,158 | 1,212 | 약 3,500 | 새 리플레이 121개 중 118개 디싱크. 1.16.1 이전 버전 리플레이로 판단해 **제외** |
| ICCup 유저 (R4) | 1,031 | 687 | 344 | 저장 49, 디싱크 36 (42%). 옛 맵(Lost Temple 등)이 많음 |
| Oregon State PvT (R5) | 509 | 452 | 37 | 저장 24, 디싱크 2 |

- 파싱: 14,327/14,406. 실패 79개는 screp로도 읽히지 않는 손상 파일이다.
- OpenBW가 처리하지 못하는 명령은 게임 속도(0x0F)뿐이고 변환 때 빠진다.
- 중복은 파일 SHA-256으로 판정하고 `build --skip-hashes`로 건너뛴다.

## 자동 수집 후보 조사 (공개 SC:R 리플레이)
| 사이트 | 결과 |
|---|---|
| cwal.gg | robots.txt `User-agent: * / Disallow: *` → 자동 수집 불가. 같은 원천(SC:R 로컬 API)을 직접 쓰는 `scr_ladder` 수집으로 대체(043) |
| repmastered.app | "Account Suspended" 응답 → 현재 운영되지 않음 |
| bwreplays.com | 도메인 매물 페이지 → 사이트 없음 |
| ICCup (`/starcraft/replays.html`) | 스타크래프트 섹션 없음("존재하지 않는 페이지") |
| Reps.ru (`/replays.php`) | 404 |
| GosuGamers 리플레이 아카이브 | 404 |
| TeamLiquid 리플레이 | robots.txt가 `/replay/download.php` 금지 → 수동 |
| ygosu | 게시글 첨부 형태, 프로 리플레이 재게시 제한 → 수동 |

자동 수집이 가능한 공개 SC:R 리플레이 사이트는 아직 찾지 못했다. 찾으면 이 표에 추가하고 사용자 확인 후 수집한다. 사이트 대신 SC:R 클라이언트의 로컬 API로 래더 경기를 직접 모은다(`scr_ladder`, 043).

## 사용자가 직접 받을 수 있는 경로
- **SC:R 클라이언트**: 게임 안에서 받은 리플레이는 `Maps\Downloads` 폴더에 저장된다(Liquipedia 안내). 래더 상위 게임이나 대회 리플레이를 이 경로로 모아 `raw/scr_public/`에 넣는다.
  - https://liquipedia.net/starcraft/Portal:Beginners/Replays
- **커뮤니티 리플레이 스레드**: TeamLiquid Brood War 포럼의 리플레이 공유 글(첨부 파일). 글마다 다운로드 허용 여부를 확인한다.
  - https://tl.net/forum/brood-war/584283-where-do-yall-get-replays-these-days
- **연구용 리플레이 모음 목록**: StarCraft AI wiki의 데이터 마이닝 문서. 대부분 1.16 시대 자료이고, 링크가 살아 있는지는 확인하지 못했다.
  - https://www.starcraftai.com/wiki/StarCraft_Brood_War_Data_Mining

## 필요한 것
- 040-AC5: SC:R 리플레이 100개 이상. 대상 맵(투혼·서킷브레이커) 위주가 좋다.
  - 사용자 리플레이 중 테란 1:1은 140개이고 모두 투혼 계열이다.
  - 서킷브레이커 SC:R 리플레이는 아직 없다.
  - `scr_ladder` 매일 수집으로 채운다. 첫 검증 12개 중 투혼은 2개였다. 5분 이상 경기는 약 1/4이었다(초반 이탈이 많음). 현재 래더 맵 풀에 서킷브레이커가 없으면 이 경로로는 얻을 수 없다.
