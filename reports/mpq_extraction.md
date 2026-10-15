# 게임 데이터 확보 (AMB-011-1)

## 조사 결론
- 비공식 배포본은 쓰지 않음(저작권).
- 2017년 Blizzard 공식 무료 1.18 설치본: 현재도 공식 링크가 살아 있는지 확인 불가.
- **채택 경로: SC:R 무료 설치본의 CASC 저장소에서 OpenBW가 읽는 파일만 추출.**

## OpenBW가 읽는 파일 (bwgame.h `global_init`)
`arr/{units,weapons,upgrades,techdata,flingy,sprites,images,orders}.dat`, `arr/images.tbl`,
`scripts/iscript.bin`, `triggers/Melee.trg`, `Tileset/{badlands,platform,install,AshWorld,Jungle,Desert,Ice,Twilight}.{vf4,cv5}`,
그리고 `images.tbl`이 가리키는 `unit/*` (GRP, LO* 오버레이).

## 도구
- `casc` CLI (casc-cli 0.1.6, CascLib 기반): `cargo install casc --version 0.1.6 --locked` (libclang 필요).
- `cargo run -p casc-extract -- <SC:R>/Data <out_dir>`: 전체 목록에서 대소문자·역슬래시·`namespace:`·`SD/` 차이를 흡수해 실제 이름을 찾고, 원래 경로로 추출, `manifest.toml`(크기·SHA-256) 생성. 누락 시 종료 코드 2.

## 상태 (2026-10-15)
- 도구 완성, 단위 테스트 5개 통과. **실제 추출은 사용자 데이터 복사 대기.**

## 사용자 작업
Windows SC:R 설치 폴더(예: `C:\Program Files (x86)\StarCraft`)에서 아래를 BC250 `~/sc_data/scr/`로 복사:
- `.build.info` (숨김 파일, CASC 열기에 필요)
- `Data/` (CASC 저장소)
- `Maps/` (래더 맵)
