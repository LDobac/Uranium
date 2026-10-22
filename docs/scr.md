# SC:R 실전 실행 (spec 012)

우리 봇을 StarCraft: Remastered 로컬·LAN 커스텀 게임에서 돌리는 방법이다. 요구사항은 [spec 012](../spec/012-bridge-remastered.md)를 따른다.

## 지원 환경
- **SC:R 1.23.10.13515 x64만 지원한다.**
  - 브리지가 `StarCraft.exe`의 SHA-256을 확인하고, 다르면 꺼진 채로 로드된다(012-R6).
  - 배틀넷 런처가 패치를 받으면 동작하지 않는다. 그때는 지원 빌드를 추가할지 정해야 한다(AMB-012-2).
- **samase 로더 0.10.28-64**(`samase64.exe`, SHA-256 `8cc8d84d2c8c533ee048760b13cf49ed369502a821840f6e35ebe537afec668c`)를 쓴다.
  - 소스가 공개되지 않은 실행 파일이고, 최신 64비트 빌드는 제3자 저장소 `BoneHorror/sezam`의 `bin/samase64.exe`에 있다.
  - 받은 뒤 해시를 확인한다. `tools\scr\run.ps1`이 실행할 때마다 확인한다.
- **로컬·LAN 커스텀 게임만** 지원한다. 배틀넷 공개 게임과 래더는 대상이 아니다(AMB-012-3).
- 외부 프로그램을 게임에 넣으므로 **계정 제재 위험**이 있다. 사용자가 위험 감수를 결정했다(2026-10-22).

## 설치
1. `samase64.exe`를 저장소 밖 폴더에 받고, 해시를 확인한다.
2. `.env`에 `URANIUM_SCR_SAMASE=<samase64.exe 경로>`를 넣는다. 나머지 키는 `.env.example`의 SC:R bridge 절에 있고, 기본값으로 충분하다.
3. 브리지를 빌드한다: `tools\scr\build.ps1`.
   - 단위 테스트를 돌린 뒤 release DLL을 `runs\scr\mod\samase\`(또는 `URANIUM_SCR_MOD_DIR`)에 복사한다.
   - `plugins` 목록 파일도 쓴다.
4. 에이전트를 CUDA로 빌드한다: `cargo build --release -p uranium-agent --features cuda`.
5. 대상 맵(투혼·서킷브레이커) 사전계산이 `URANIUM_MAPS_DIR`에 있어야 한다. 없으면 에이전트가 세션을 거부한다(AMB-031-4).
   - SC:R이 연 맵 파일과 같은 파일로 만든다: `uranium-replay mapgen <맵 파일>`.

## 실행 순서
1. **에이전트**: `uranium-agent scr --checkpoint <체크포인트> --backend cuda [--sample stochastic]`.
   - 기본 listen 주소는 `127.0.0.1:7700`, 항복 모드는 `on`(채팅 `gg`)이다.
   - 결과는 `runs/scr/sessions.jsonl`에 게임마다 한 줄씩 쌓인다. `uranium-agent summarize runs/scr/sessions.jsonl --out <md>`로 요약한다.
2. **SC:R**: `tools\scr\run.ps1`.
   - 배틀넷 앱에 로그인돼 있어야 한다. 다른 SC:R이 떠 있으면 먼저 닫는다.
3. **게임 시작은 사람이 한다.**
   - 싱글 플레이 → 커스텀 게임, 또는 멀티플레이 → LAN(UDP)으로 게임을 만든다. 테란을 고르고 상대(내장 AI 또는 사람)를 넣어 시작한다.
   - samase `--map` 자동 시작은 이 빌드에서 실패한다.
4. 게임이 시작되면 브리지가 에이전트에 접속해 플레이한다. 봇이 플레이하는 클라이언트의 마우스·키보드는 건드리지 않는다.
5. **게임이 끝나면 결과 화면에서 나간다.**
   - 봇이 항복하면 "gg" 채팅 뒤 명령을 멈추고 항복으로 기록한다. 게임 창은 사람이 닫는다(012-R11).
   - 다음 게임은 메뉴에서 다시 시작한다. 게임마다 새로 접속한다.

## 기록과 문제 해결
- **브리지 로그**: `URANIUM_SCR_LOG_DIR\uranium_bridge_scr.log`(run.ps1 기본 `runs\scr\logs`). 남는 내용은 다음과 같다.
  - 시작 검사 결과(012-R6)
  - 게임 정보(맵·플레이어)
  - 게임 종료
  - 프레임 훅 지연(012-AC4)
  - 관측 → 명령 처리 지연과 만료 Act 수(012-AC6)
- **samase 로그**: 모드 폴더의 상위 폴더에 `samase.log`가 생긴다.
- **봇이 움직이지 않을 때** 화면에 "uranium: agent not connected"가 뜨면 에이전트에 접속하지 못한 것이다(012-R16). 에이전트가 떠 있는지, `URANIUM_SCR_AGENT_ADDR`가 맞는지 확인한다.
- **"012-R6: the bridge stays off"**: exe 해시, samase 버전, API 중 하나가 맞지 않는다. 로그에 이유가 있다.
- **정적 맵 확인용**: `URANIUM_SCR_DUMP_GAMEINFO=<파일>`을 주면 게임 시작 때 `GameInfo`를 postcard 메시지로 저장한다.
