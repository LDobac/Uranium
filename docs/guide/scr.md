# SC:R 실전 실행

StarCraft: Remastered 로컬·LAN 커스텀 게임에서 봇 실행. 요구사항 [spec 012](../spec/2026_10_15_22_24_012-bridge-remastered.md).

## 지원 환경
- **SC:R 1.23.10.13515 x64만.** 브리지가 `StarCraft.exe` SHA-256 확인, 다르면 꺼진 채 로드([012-R6](../spec/2026_10_15_22_24_012-bridge-remastered.md)). 런처가 패치를 받으면 동작 안 함 → 지원 빌드 추가 여부 결정 필요([AMB-012-2](../spec/README.md)).
- **samase 로더 0.10.28-64**(`samase64.exe`, SHA-256 `8cc8d84d2c8c533ee048760b13cf49ed369502a821840f6e35ebe537afec668c`). 소스 비공개 실행 파일, 최신 64비트 빌드는 제3자 저장소 `BoneHorror/sezam`의 `bin/samase64.exe`. 받은 뒤 해시 확인(`tools\scr\run.ps1`이 실행마다 확인).
- **로컬·LAN 커스텀 게임만**. 배틀넷 공개 게임·래더 대상 아님([AMB-012-3](../spec/README.md)).
- 외부 프로그램 주입 → **계정 제재 위험**(사용자가 감수 결정).

## 설치
1. `samase64.exe`를 저장소 밖 폴더에 받고 해시 확인.
2. `.env`에 `URANIUM_SCR_SAMASE=<samase64.exe 경로>`. 나머지 키는 `.env.example` SC:R bridge 절(기본값으로 충분).
3. 브리지 빌드 `tools\scr\build.ps1`: 단위 테스트 → release DLL을 `runs\scr\mod\samase\`(또는 `URANIUM_SCR_MOD_DIR`)에 복사, `plugins` 목록 파일 작성.
4. Python 환경에 `uranium` 패키지(`uranium._core` 포함)와 CUDA PyTorch: 저장소 루트에서 `uv pip install -e .`([AMB-000-4](../spec/README.md)). 상세 [commands.md](commands.md#pythonpytorch-amb-000-4-001-r8).
5. 대상 맵(투혼·서킷브레이커) 사전계산이 `URANIUM_MAPS_DIR`에 있어야 함(없으면 세션 거부, [AMB-031-4](../spec/README.md)). SC:R이 연 맵 파일과 같은 파일로 `uranium-replay mapgen <맵 파일>`.

## 실행 순서
1. **에이전트**: `python -m uranium scr --checkpoint <체크포인트> [--sample stochastic]`(백엔드 기본은 그 PC GPU, CUDA bf16, [032-R8](../spec/2026_10_15_22_24_032-model-architecture.md)). 전체 옵션 [commands.md](commands.md#에이전트-서버기타).
   - 기본 listen `127.0.0.1:7700`, 항복 `on`(채팅 `gg`).
   - 결과 `runs/scr/sessions.jsonl`에 게임마다 한 줄. 요약 `uranium-agent summarize runs/scr/sessions.jsonl --out <md>`.
2. **SC:R**: `tools\scr\run.ps1`. 배틀넷 앱 로그인 필요, 다른 SC:R은 먼저 닫음.
3. **게임 시작은 사람이**: 싱글 → 커스텀 게임, 또는 멀티 → LAN(UDP). 테란 선택, 상대(내장 AI 또는 사람) 넣고 시작. samase `--map` 자동 시작은 이 빌드에서 실패.
4. 시작되면 브리지가 에이전트에 접속해 플레이. 봇 클라이언트의 마우스·키보드는 건드리지 않음.
5. **끝나면 결과 화면에서 나감.** 봇 항복은 "gg" 채팅 → 명령 정지 → 항복 기록, 창은 사람이 닫음([012-R11](../spec/2026_10_15_22_24_012-bridge-remastered.md)). 다음 게임은 메뉴에서 다시, 게임마다 새로 접속.

## 기록·문제 해결
- 브리지 로그 `URANIUM_SCR_LOG_DIR\uranium_bridge_scr.log`(run.ps1 기본 `runs\scr\logs`): 시작 검사([012-R6](../spec/2026_10_15_22_24_012-bridge-remastered.md)), 게임 정보(맵·플레이어), 게임 종료, 프레임 훅 지연(012-AC4), 관측 → 명령 지연·만료 Act 수(012-AC6).
- samase 로그: 모드 폴더의 상위 폴더 `samase.log`.
- 화면에 "uranium: agent not connected"(012-R16): 에이전트가 떠 있는지, `URANIUM_SCR_AGENT_ADDR` 확인.
- "012-R6: the bridge stays off": exe 해시·samase 버전·API 중 하나 불일치, 이유는 로그.
- 정적 맵 확인: `URANIUM_SCR_DUMP_GAMEINFO=<파일>` → 게임 시작 때 `GameInfo`를 postcard 메시지로 저장.
