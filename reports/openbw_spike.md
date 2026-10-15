# M1(b) OpenBW 헤드리스 스파이크

## 상태 (2026-10-15)
| 항목 | 결과 |
|---|---|
| OpenBW 소스 | `third_party/openbw` @ `4b046d5f65302b10cb0a745f0fecd37ec85b20a8` |
| BWAPI(OpenBW 포크) | `third_party/bwapi` @ `48124ba8ed1b4d52b3dfd52acbaf34afb9a37fe2` (BWAPI 4.2) |
| 빌드 | **성공** — `tools/openbw/build.sh` (Release, `OPENBW_ENABLE_UI=0`) → `BWAPILauncher`, `libBWAPI.so`, `libOpenBWData.so`, `ExampleAIModule.so` |
| 빌드 우회 | CMake 4: `-DCMAKE_POLICY_VERSION_MINIMUM=3.5`, 최신 GCC: `-DCMAKE_CXX_FLAGS="-include cstdint"`(BWAPI 헤더 누락) |
| 데이터 로더 패치 | `third_party/patches/0001-openbw-loose-data-files.patch` — `URANIUM_DATA_DIR`가 설정되면 MPQ 없이 풀린 파일 폴더에서 읽음. BWAPI의 세 호출 지점(`BWData.cpp` 185/270/914)이 모두 `data_files_directory()`를 쓰므로 OpenBW 쪽 한 곳만 수정 |
| 데이터 없음 동작 | MPQ 경로: `file_reader: failed to open ./Patch_rt.mpq` / 폴더 경로: `data_files_loader: arr/units.dat: file not found (loose dir: ...)` — 명확한 에러로 종료 확인 |
| 데이터 추출 | **성공** — CASC 53,051개 중 필요한 956개 추출, 누락 0. `.dat` 크기가 1.16.1 구조와 일치(units 19876, flingy 3135=209×15, sprites 3229 등) |
| 맵 버전 | 투혼: 구버전(e364…)과 **1.3**(f39d…, 2024 시즌) 2종 → 1.3 채택. 서킷브레이커: **1.0** 1종(ba95…). `config/assets.toml`에 고정 |
| 맵 로드 | **성공** — 두 맵 모두 로드(맵 이름 "투혼 1.3", "Circuit Breakers 1.0" 정상). STRx 패치 불필요 |
| 게임 진행 | **진행 확인** — 싱글플레이, ExampleAIModule(T) vs 비활성 컴퓨터(Z), 헤드리스 20초 실행, 크래시 없음 |
| 이상 징후 | ExampleAIModule의 SCV 채취 명령이 `Unit_Does_Not_Exist`로 반복 실패(가장 가까운 미네랄을 찾지 못함). 데이터 크기는 정상이라 원인 미확정 — 게임 상태를 들여다볼 프로브가 필요 |
| fps·결정론·게임 완주 | **미측정** — 동봉 예제 AI는 게임을 끝내지 않고 프레임 수도 출력하지 않음. 프로브 또는 M2 브리지 필요 |
