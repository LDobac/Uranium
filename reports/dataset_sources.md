# 리플레이 출처 (040-R14, 내부 기록)

- 일자: 2026-10-16
- 데이터 정책(040, AMB-040-3):
  - 내부 학습 전용이며 재배포하지 않는다.
  - 자동 수집은 robots.txt와 약관이 허용하는 곳에서만 1 req/s 이하로 한다.
  - 그 밖의 출처는 링크만 남기고 사용자가 직접 받는다(2026-10-16 사용자 결정).
- 받은 파일은 `<URANIUM_SHARED_DIR>/datasets/replays/raw/<src>/`에 둔다(040-R13). 파일명과 플레이어 이름은 리포트에 쓰지 않는다.

## 확보한 출처
| src | 내용 | 수량 | 라이선스 / 근거 | 확보 방법 |
|---|---|---|---|---|
| `stardata` | STARDATA 원본 1.16.1 리플레이 | 65,646 | BSD + 특허 허여 (facebookresearch/StarData) | `original_replays.tar.gz` 다운로드(5.3GB, SHA-256 `037cb688…642a`) |
| `personal` | 사용자 본인 SC:R 리플레이 | 576 (테란 1:1 5분 이상 140) | 사용자 소유 | 사용자 PC의 리플레이 폴더에서 복사 |

## 자동 수집 후보 조사 (공개 SC:R 리플레이)
| 사이트 | 결과 |
|---|---|
| cwal.gg | robots.txt `User-agent: * / Disallow: *` → 자동 수집 불가 |
| repmastered.app | "Account Suspended" 응답 → 현재 운영되지 않음 |
| bwreplays.com | 도메인 매물 페이지 → 사이트 없음 |

자동 수집이 가능한 공개 SC:R 리플레이 출처는 아직 찾지 못했다. 찾으면 이 표에 추가하고 사용자 확인 후 수집한다.

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
