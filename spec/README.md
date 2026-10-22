# Uranium Specs

Uranium은 스타크래프트1 **테란 밀리 AI 봇** 프로젝트다. 이 폴더는 구현의 단일 진실 공급원(Single Source of Truth)이다.

## 규칙
1. **Spec 먼저.** 코드 변경은 해당 spec이 `Approved` 상태여야 시작할 수 있다. 동작을 바꾸려면 spec을 먼저 수정하고 리뷰한 뒤 코드를 고친다.
2. 문서는 공통 템플릿을 따른다: `Status / Purpose / Scope & Non-goals / Requirements / Interfaces / Acceptance Criteria / Risks & Open Questions / Related`.
3. 요구사항 키워드:
   - **MUST**: 반드시 지켜야 하며 Acceptance Criteria로 검증한다.
   - **SHOULD**: 강력 권장이며, 벗어나려면 사유를 기록한다.
   - **MAY**: 선택 사항.
4. 요구사항에는 ID(`<spec번호>-R<n>`, 예: `010-R3`)를 붙인다. 코드 주석, 테스트, PR에서는 이 ID로 참조한다.
5. Status 값:
   - `Draft`: 작성 중
   - `Review`: 리뷰 대기
   - `Approved`: 구현 가능
   - `Implemented`: 구현 및 AC 통과
   - `Superseded`: 대체됨
6. 수치 중 아직 측정하지 않은 값은 `TBD(M1)`처럼 측정할 마일스톤을 함께 적는다. 이 표기는 **측정만 하면 정해지는 값**에 쓴다.
7. **사용자 결정이 필요한 모호한 부분**은 `[SPEC AMBIGUOUS] AMB-<spec번호>-<n>`으로 표기한다. 해당 항목은 사용자 확인 전까지 구현 근거로 쓸 수 없다. 확정되면 다음 순서로 처리한다.
   1. 본문에 결정 내용을 반영한다.
   2. 태그를 제거한다.
   3. 아래 레지스트리의 상태를 `Resolved`로 바꾸고 날짜와 결정 내용을 적는다.
8. `[SPEC AMBIGUOUS]`가 남아 있는 spec은 `Approved`가 될 수 없다.
8-1. **보류(Deferred)** — 도입 2026-10-15
   - 결정 시점이 뒤 마일스톤으로 정해진 항목은 `[SPEC DEFERRED(Mx)] AMB-<번호>-<n>`으로 표기한다. 이 표기는 사용자 확인을 거쳐야 붙일 수 있다.
   - 레지스트리 상태는 `Deferred (Mx)`다.
   - Deferred 항목은 spec의 `Approved`를 막지 않는다. 그러나 **그 항목에 의존하는 코드는 작성할 수 없다.**
   - 해당 마일스톤을 시작하기 전에 반드시 사용자에게 다시 질문해 확정한다.
8-2. **개인·보안 정보 금지** — 도입 2026-10-15. spec과 모든 문서에 IP 주소, 호스트 이름, 계정, 이메일, 개인 경로, 자격 증명을 쓰지 않는다. 자리표시자(`<server>`)를 쓰고 실제 값은 git에 올리지 않는 `.env`에 둔다([001-R9·R10](001-system-architecture.md)). 장비 종류·상태(보드·GPU 모델명, 장비 번호, 특정 PC 이름·사양)도 spec에 쓰지 않고 OS·백엔드(Linux/Windows, Vulkan/CUDA)로만 표기한다(2026-10-19, 001-R10).
9. **spec 번호는 구현 순서가 아니라 영역 분류다.**
   - 앞 두 자리가 영역(대역)을 나타낸다.
   - 같은 대역 안에서는 다른 문서가 의존하는 문서에 앞번호를 주는 것을 권장한다. 강제는 아니다.
   - 대역 사이의 빈 번호(예: 002, 014)는 이후 추가할 문서를 위해 비워 둔다. 새 문서는 해당 영역 대역의 빈 번호를 쓴다.
   - 한 번 부여한 번호는 바꾸지 않는다. 요구사항 ID와 AMB ID가 번호에 묶여 있기 때문이다. 폐기한 spec은 `Superseded`로 남긴다.
   - **구현 순서는 아래 "마일스톤" 표가 정한다.**

   | 대역 | 영역 |
   |---|---|
   | 00x | 전체 개요·아키텍처 |
   | 01x | 게임 연동 (프로토콜, 브리지, 버전 매핑) |
   | 02x | 게임 실행 환경 (오케스트레이션, 상대 풀) |
   | 03x | 모델 (관측, 액션, 아키텍처, 보상) |
   | 04x | 데이터·학습 (리플레이, 모방학습, RL) |
   | 05x | 평가 |
   | 06x | 인프라 |
   | 07x–09x | 예약 (새 영역이 필요하면 사용자 확인 후 지정) |

## Ambiguity Registry

| ID | 내용 | 상태 |
|---|---|---|
| AMB-000-1 | 승률 목표 수치 | Resolved (2026-10-15: G1 80%·G2 90% 초안 확정, M5 이후 조정 제안 가능, APM 제한 없음) |
| AMB-000-2 | SC:R 실사용 환경(로컬/LAN vs 배틀넷)·상대 | Resolved (2026-10-15: 로컬/LAN 커스텀 한정, 상대 사람/내장 AI) |
| AMB-000-3 | 맵 파일 버전, 상대 Random 종족 지원 | Resolved (2026-10-15: 최신 공개 버전(1.16.1 호환 검증 후 해시 고정), Random 지원) |
| AMB-001-1 | SC:R 실전 추론 위치 | Deferred (M7) |
| AMB-001-2 | OpenBW 브리지 C++ shim 허용 여부 | Resolved (2026-10-15: 얇은 C++ shim 허용) |
| AMB-001-3 | 에이전트가 모델을 쓰면 생기는 crate 순환(agent → model → env → agent) 해결 | Resolved (2026-10-17: 스케줄러를 새 crate `uranium-runner`로 분리, 바이너리 이름 `uranium-env`·CLI 유지, `uranium-env`는 라이브러리만) |
| AMB-001-4 | 장비 라벨(보드 이름·장비 번호)을 문서에 쓰는 것이 001-R10 위반인지(BC 전 리뷰 G-D9) | Resolved (2026-10-19: git 추적 파일(spec·코드·주석·추적 config·docs·CLAUDE.md)에는 장비 종류·상태를 쓰지 않고 OS·백엔드로만 표기, 허용 위치는 git 미추적 config와 리포트 — 001-R10) |
| AMB-011-1 | MPQ·맵 파일 확보 경로 (사용자 미보유) | Resolved (2026-10-15: 사용자 구버전 확보 + Claude SC:R CASC 추출 조사 병행) |
| AMB-012-1 | samase_scarf 분석 실패 시 대안 선택 | Resolved (2026-10-22: 대안 불필요 — M7 스파이크에서 samase 0.10.28-64가 SC:R 1.23.10.13515 x64에서 012-AC1 목록을 모두 제공, 명령 송신·시야 비트 실게임 확인) |
| AMB-012-2 | 대상 SC:R 빌드 정책, 기존 samase 환경 | Resolved (2026-10-22: SC:R 1.23.10.13515 하나로 고정(exe 해시, 다른 빌드는 012-R6 비활성), 기존 samase 환경 없음 → M7 스파이크에서 로더 확보·설치 절차 조사, 실전 PC는 CUDA 학습 호스트(Windows)와 같은 PC) |
| AMB-012-4 | SC:R 항복 시 게임 나가기(`0x57`로는 로컬 클라이언트가 빠지지 않음, samase에 수단 없음) | Resolved (2026-10-22: 채팅 후 브리지가 명령을 멈추고 `GameEnd(Surrender)`를 보낸 뒤 연결을 닫는다. 게임 창은 사람이 닫는다. 메뉴 키 입력·내부 함수 호출은 쓰지 않는다 — 012-R11) |
| AMB-012-3 | 실행 환경·안티치트 정책 (→AMB-000-2) | Resolved (2026-10-15: 로컬/LAN 한정) |
| AMB-020-1 | OpenBW 원격 호스트 사양 | Resolved (2026-10-15: 게임 전용 Linux PC 2대 이상, 사양 자동 감지) |
| AMB-020-2 | 최대 게임 시간·무승부 규칙 | Resolved (2026-10-15: 60분 초과 시 무승부, 보상 0 — 점수 판정안에서 변경) |
| AMB-020-3 | 같은 게임의 두 세션이 다른 승자를 보고할 때의 판정 | Resolved (2026-10-17: 더 늦게 끝난(GameEnd 프레임이 더 큰) 세션의 승자를 쓴다, 불일치 건수는 따로 기록. 평가 집계·리그 승률 표·r_win 모두 적용, 020-R13) |
| AMB-020-4 | 020-R3 "에이전트 무응답 30초 → 게임 중단"(미구현)과 011-R18 드롭 시간 10분의 모순(리뷰 C-9) | Resolved (2026-10-19: 30초 중단 삭제, 011-R18 드롭 → 상호 퇴장 Aborted(020-R13)로 정합, 5초 초과 추론 로그 유지 — 020-R3) |
| AMB-021-1 | BWAPI 4.4 봇 대응 방향, T1 후보 목록 | Resolved (2026-10-15: 단계적 접근 — 4.2 호환 봇+T0 먼저, 4.4 봇은 포팅 비용 측정 후 재확인. 2026-10-16 재확인: Steamhammer·McRave·Stardust·ZZZKBot·Iron 포팅) |
| AMB-021-2 | 상대 봇 맵핵 허용 | Resolved (2026-10-15: 기본 금지, 학습 후반 강화용 별도 상대로만 허용, 평가 미사용) |
| AMB-021-3 | OSS 봇 결정론(020-AC2를 T1 상대에도 요구할지) | Resolved (2026-10-16: 그대로 둔다 — 상대 다양성으로 보고 패치하지 않음, 020-AC2는 T0·셀프플레이로 검증) |
| AMB-030-1 | `enemy_last_seen_age` 채널 의미: (a) 타일을 마지막으로 본 뒤 경과 시간 vs (b) 그 타일에서 적을 마지막으로 본 뒤 경과 시간, 정규화 | Resolved (2026-10-16: 사용자 위임으로 Claude 결정 — (a) 정찰 기록, 채널명 `last_seen_age`, 보임 0·미탐색 1·그 외 log(1+경과)/log(1+86400)) |
| AMB-030-2 | BC 전 리뷰 E절 관측 항목: hazard 반지름 고정 16px, 시야 밖 자원 미포함, 평가 대전 `last_failed`, Random 상대 종족, 증강 `entity_cell` 1타일 어긋남 | Resolved (2026-10-19: 효과별 실제 반지름·플레이그·인스네어 포함(030 §3), 맵 공개 자원은 마지막 관측 상태로 포함(010-R5 예외·030-R2), 평가 대전 `last_failed` 0 고정·RL부터 실제 값(030-R10), Random 상대는 첫 적 유닛 관측 전 unknown(030-R11), `entity_cell`은 변환 픽셀로 재계산(030-R4)) |
| AMB-030-3 | 시야 밖 자원(E-3)의 초기값 출처와 엔티티 대응 | Resolved (2026-10-19: 기술 결정(메인). 맵 사전계산(`MapPrecompute`, CHK 해시별, CHK 유닛 배치의 자원 종류·위치·초기량)을 BC·라이브 공통으로 쓰고, 레이블 target과는 타일 위치로 매칭, 한 번도 못 본 자원은 `seen_frame = 0`(경과 프레임 feature = 현재 프레임 ÷ 2048). 브리지·proto 무변경 — 030 §2, 010-R5, 031) |
| AMB-030-4 | 탄환 효과의 시야 누출(OpenBW shim이 CompleteMapInformation을 켜서 BWAPI가 모든 탄환을 보이는 것으로 처리 → 시야 밖 탄환이 hazard 채널에 들어감, 2026-10-22 M7 조사 중 발견) | Resolved (2026-10-22: 사람 화면과 같은 규칙 — 중심 타일이 지금 보이는 효과만. 에이전트 feature 추출에서 `RawFrame.visibility`로 거르고(재build 없음, BC 본학습 전 반영), OpenBW 브리지·SC:R 브리지도 같은 규칙으로 보낸다. 끝난 코어 비교 학습은 그대로 둔다 — 030 §3, 010-R5) |
| AMB-031-1 | executor(규칙) 범위 | Resolved (2026-10-15: 최소 — 일꾼·건물 선택만, 위치 보정·심시티 스냅·자동 채취 없음) |
| AMB-031-2 | APM 제한 수치 (→AMB-000-1) | Resolved (2026-10-15: APM 제한 없음, delay 1–32 비균등 16 bucket) |
| AMB-031-3 | 맵 사전계산 채널(base_loc·choke·wall_candidate) 적용 범위 | Resolved (2026-10-16: 대상 맵만이 아니라 모든 맵에 맵 무관 알고리즘으로 적용) |
| AMB-031-4 | 라이브 게임 맵 사전계산 키(BC는 CHK 해시, 라이브 `GameInfo.map_hash`는 맵 파일 해시) | Resolved (2026-10-17: 대상 맵 2개만 맵 파일 해시 키로 오프라인 생성(`uranium-replay mapgen`), 그 외 맵은 사전계산 없음 + 경고, 브리지·프로토콜 변경 없음) |
| AMB-031-5 | 모델 정책에서 executor가 액션을 거부할 때 동작 | Resolved (2026-10-17: 명령 없이 샘플된 delay만 진행, 거부 건수 기록) |
| AMB-031-6 | GG 항복(승률이 현저히 낮을 때 채팅 후 퇴장) 판단 기준·임계값·적용 범위·채팅 | Resolved (2026-10-21: 상대 정보 없이 학습한 승률 추정 win_open(032-R18) + 규칙 조건(병력 격차 1/3 또는 붕괴), P ≤ 5%가 30초 연속·10분 이후, SC:R 실전만 기본 On·OpenBW는 Shadow, 채팅은 plan 옵션 `surrender_chat` 기본 "gg" — 031-R12, 010-R13, 011-R21, 012-R11, 020-R14, 041-R19, 042-R13) |
| AMB-032-1 | 정책 구조 (단일 vs 계층형) | Resolved (2026-10-15: 단일 end-to-end) |
| AMB-032-2 | 매치업별 모델 구성 | Resolved (2026-10-15: 단일 공유 모델 + 종족 임베딩) |
| AMB-032-3 | 전략 조건 z 포함 여부 | Resolved (2026-10-15: 포함) |
| AMB-032-4 | 시간 축 코어 | Resolved (2026-10-15: LSTM 시작, M5에서 Transformer 메모리와 비교. 비교 절차는 AMB-032-6) |
| AMB-032-5 | 032-AC1의 CUDA 판정: Burn/cubek CUDA matmul이 f32를 TF32로 계산해 Flex 대비 최대 1.3e-3 차이(Vulkan은 1e-6) | Resolved (2026-10-16: 허용 오차 확대 — CUDA ≤ 1e-2, Vulkan ≤ 1e-4 유지) |
| AMB-032-6 | 시간 축 코어 비교 절차(AMB-032-4가 "M5에서 비교"를 남긴 채 닫혀 절차·AC가 없었음, 2026-10-18 발견) | Resolved (2026-10-18: BC 본학습 전 짧은 비교. 코어마다 S1 5,300 update 1회(약 2일), 짧은 run 전용 스케줄(warmup 5% + cosine), val 500게임 bc-eval, 맞대결 맵당 50판 + 각자 T0 상대 조합당 50판(코어당 300판), 확률적 샘플링, 판정은 결과 보고 후 사용자 결정 — 032-R17·AC7) |
| AMB-032-7 | Burn autotune·fusion 채택(리뷰 F-2-1·G-D5)과 backward 허용 오차(리뷰 H, `cross_device.rs` 1e-3이 spec에 없음), 코어 비교 val 500게임 선정(G-D8) | Resolved (2026-10-19: CUDA·Vulkan 모두 autotune+fusion, 코어 비교·본학습 같은 설정(041-R15). backward AC Vulkan 1e-3, CUDA는 autotune·fusion을 켠 재측정 뒤 여유를 둔 값으로 확정(D-a, 032-AC8 → 2026-10-19 실측 3.7e-3~4.9e-3, CUDA 1e-2). val은 고정 seed 셔플 500게임, seed 1개 유지·한계는 리포트(032-R17 ②)) |
| AMB-032-8 | 032-AC2 Linux Vulkan 호스트 추론 지연 기준(초안 40ms, 2026-10-20 실측 p95 f32 179~187ms)과 호스트 역할 | Resolved (2026-10-20: CUDA 호스트가 유휴일 때는 주 장비, 바쁠 때는 Linux Vulkan 호스트가 주 장비. AC2 판정은 CUDA 호스트에만(같은 호스트 Vulkan보다 빠름), Linux Vulkan 호스트는 실측 기록만·40ms 폐기. BC는 작업자 우선순위·양보(061-R13·R14)로 구현됨, RL 적용은 M6 — 032-AC2·R17, 060-R8) |
| AMB-034-1 | 무력화 유닛 명령 불가 규칙이 일부 클래스에만 적용(리뷰 B-11), SetRally 다중 건물 선택(리뷰 E) | Resolved (2026-10-19: 무력화 규칙을 모든 선택 자격·executor 선택에 적용(034-R6), 건물 대상 명령은 건물 1개만 선택, SetRally max_select 1(034-R7)) |
| AMB-040-1 | 리플레이 출처 | Resolved (2026-10-15: 공개 데이터셋·커뮤니티·공개 프로/래더·AI 봇 리플레이) |
| AMB-040-2 | 리플레이 품질 필터 기준 | Resolved (2026-10-15: S1 APM≥80·5분+, S2 프로/상위 래더 APM≥150) — S2 기준은 AMB-040-7로 대체(2026-10-19) |
| AMB-040-3 | 리플레이 라이선스·출처 표기·크롤링 범위 | Resolved (2026-10-15: 내부 학습 전용, 재배포 금지, 약관·robots.txt 준수) |
| AMB-040-4 | BC 데이터 저장 방식 (디스크 < 1TB) | Resolved (2026-10-15: RawFrame 압축 저장 S2→S1 순, 예산 초과분은 재생 스트리밍) |
| AMB-040-5 | 040-R8 디싱크 판정 기준(1.16.1은 승패 기록 없음) | Resolved (2026-10-16: 무효 명령 비율·마지막 프레임 미도달·이탈자 우세·SC:R 기록 승자 불일치 중 하나면 디싱크. 임계값 2026-10-16 확정: 무효 15% 초과, 종료 프레임 미달, 이탈 직전 1.5배) |
| AMB-040-6 | BC shard의 정적 맵 정보 출처 (feature 추출용) | Resolved (2026-10-16: shard game마다 `GameInfo`(static_map 포함) 저장, schema_version 상향 후 재빌드) |
| AMB-040-7 | S2 품질 기준(040 품질 표 테란 APM ≥ 150·프로/상위 래더, 043-R8)과 041-R13("대상 맵 train 분할만, 소스 무관")의 모순(리뷰 A-4·G-D3) | Resolved (2026-10-19: APM ≥ 150 등 S2 기준은 적용하지 않음. 041-R13 유지, 040 품질 표·043-R8을 맞춤) |
| AMB-040-8 | 040-R4 "스팸 명령 병합"이 Train 연타 등 게임에서 모두 적용되는 명령까지 합침(리뷰 A-2) | Resolved (2026-10-19: 병합 전면 제거, 근접 프레임 delay 최소 1 정규화는 유지 — 040-R4) |
| AMB-040-9 | value 정답 `result`가 LEAVE로 끝난 게임에서 0(리뷰 A-3), 040-R8 ④ 미구현 | Resolved (2026-10-19: 브리지 승자가 없으면 첫 LEAVE 이탈자 패배, 그래도 없으면 SC:R 기록 승자(`result_source = recorded`, D-b), 판정 불가는 result 0 + value 가중치 0(041-R7), 기록 승자가 다른 방법 결과와 다르면 디싱크 — 040-R8) |
| AMB-040-10 | 데이터 수정 방식(재build vs 로더 수정)과 shard 쓰기 검증(리뷰 G-D1·G-D2, F-2-3) | Resolved (2026-10-19: 전체 재build(A-1·A-2·A-3·E-2·E-3), shard는 로컬 우선 → 공용 복사 → 재읽기 해시 검증 → rename(040-R17), 첫 스텝은 게임 시작 관측(040-R18)) |
| AMB-040-11 | 표본 build에서 무효 레이블 비율이 비정상인 시점(scr_ladder APM 2,376 → 55%, 957 → 24.6%, 나머지 2.3–3.9%)의 처리(APM 상한 또는 제외) | Resolved (2026-10-19: APM 상한이 아니라 데이터 품질 기준으로, build에서 무효 레이블(BC 레이블 검사와 같은 함수) 비율 ≥ 20%인 테란 시점을 `filtered`로 기록하고 제외 — 040-R19) |
| AMB-040-12 | R-4 재build에서 040-R17 검증을 통과한 shard 7개가 공용 드라이브에서 0으로 채워지거나 잘림(Linux CIFS 클라이언트가 자기 캐시에서 다시 읽음), 한 출처 안 같은 리플레이 파일이 두 게임으로 들어감 | Resolved (2026-10-20 사용자 결정: 다시 읽기는 fsync 뒤 캐시 우회로 서버 내용을 읽고(체크포인트 push 061-R4 포함), manifest를 만들기 전 shard를 모두 디코드해 보며, 같은 파일은 출처 전체 정렬 목록 기준 첫 파일만 한 게임 — 040-R17·R20, 061-R4. 손상 파일은 버리지 않고 해당 게임을 다시 build. Linux 마운트 설정은 그대로) |
| AMB-041-1 | BC 혼합 선택 레이블 판정(`select_cap`: 일부 유닛만 자격이 있는 선택) | Resolved (2026-10-17: 사용자 위임으로 Claude 결정 — 게임 엔진 동작 기준. 자격 없는 유닛을 레이블에서 빼고 남는 유닛이 없을 때만 무효(041-R11). 함께 확인한 엔진 규칙: 어택땅 자격 = 이동 가능(034), 선택 추적 = 게임 선택 규칙(040-R3), 떠 있는 건물 우클릭 = 이동(040-R11), 랠리는 건설 중에도 가능(034)) |
| AMB-041-2 | 041-AC2(BC vs T0) 측정 판수·z·샘플링 모드 | Resolved (2026-10-17: 조합(종족×맵)당 50판, z 없음, argmax·확률적 두 모드 보고, Wilson 95%) |
| AMB-041-3 | BC train/val/test 분할 방법·비율(040-R5 세부) | Resolved (2026-10-18: `replay_sha256` 게임 단위 결정론 분할(TvT 두 시점 같은 쪽), 대상 맵 80/10/10, 그 외 95/5, S1·S2·ablation 공통, 대상 맵 test는 학습 제외 — 041-R12) |
| AMB-041-4 | BC S1·S2 데이터 선택, 소스 혼합 가중치, 게임 순서 셔플 | Resolved (2026-10-18: S1 네 소스 train 분할 전부 가중치 없이, S2 대상 맵 train 분할만, 고정 seed 셔플 — 041-R13) |
| AMB-041-5 | grad clip 1.0의 의미(Burn은 파라미터 텐서별 클립, 리뷰 B-9·G-D4) | Resolved (2026-10-19: 전역 L2 norm clip 1.0, micro-lane gradient 합산 뒤 적용, 직접 구현, BC·RL learner 공통 — 041-R3, 042 옵티마이저 표) |
| AMB-041-6 | 041-R9 스트리밍 로더 미구현(리뷰 E-11) | Resolved (2026-10-19: 예산 초과 시에만 필요한 조건부 요구사항으로 — 041-R9) |
| AMB-042-1 | 학습 기간·장비 예산 | Resolved (2026-10-15: 기간 제한 없음, 전체 리그 구성) |
| AMB-042-2 | TvP·TvZ 상대 다양성 (P/Z 상대 정책 추가 여부) | Deferred (M6) (2026-10-17 M6 시작 재확인: T0·T1 봇으로 시작, M6 중간 점검에서 정체 여부 보고 후 재확인) |
| AMB-042-3 | RL 초기 설정값(정책 지연 임계값, 가중치 배포 K, 스냅샷 N, 커리큘럼 전환 조건, 헤드별 엔트로피, KL 감쇠, 시분할, PFSP, 종족·Random 비율, 배치·옵티마이저) | Resolved (2026-10-17: 사용자 위임으로 Claude 결정 — IMPALA·AlphaStar 기준 축소값을 042 "초기 설정값" 절·`config/rl.toml`에 기록, 전부 042-R9 튜닝 대상, 042-AC1 후 조정 제안 가능. 첫 RL 학습 호스트는 CUDA 호스트) |
| AMB-042-4 | actor가 기록할 behaviour 정보(전체 logits는 target_loc만 16,384개) | Resolved (2026-10-17: 선택 액션의 헤드별·joint behaviour log-prob만 기록, KL(π‖π_BC)은 learner가 BC 모델 재계산) |
| AMB-042-5 | BC 본학습 체크포인트 전 RL 파이프라인 검증 정책 | Resolved (2026-10-17: build된 shard 표본으로 짧은 BC를 돌린 체크포인트로 042-AC1 형식 연습) |
| AMB-042-6 | Aborted 게임 trajectory 처리(게임 끝까지 보류 vs 즉시 전송·회수) | Resolved (2026-10-17: 즉시 전송, Aborted면 미사용분 회수, Ctrl+C는 learner 먼저 정지, 042-R3 개정) |
| AMB-042-7 | 042-AC1 실험 규모 | Resolved (2026-10-17: V-trace·PPO 각 3 seed(6 run), 033-AC2 ablation은 본 BC·z 테이블 이후) |
| AMB-042-8 | 첫 RL 게임 호스트 구성 | Resolved (2026-10-17: 학습 호스트 로컬 hostd로 시작, 리플레이 build 후 원격 Linux 게임 호스트 두 대 추가) |
| AMB-042-9 | KL(π‖π_BC) 계산 방식(BC 재forward 비용 약 1.5배) | Resolved (2026-10-17: BC 모델 재forward로 정확한 분포 KL, GPU 메모리가 부족하면 사용자에게 보고) |
| AMB-043-1 | 래더 수집: 040 데이터 정책(약관 준수) 예외 허용 여부 | Resolved (2026-10-16: EULA 해석 위험 감수하고 예외 허용, 내부 학습 전용·재배포 금지 유지) |
| AMB-043-2 | 래더 수집: 실행 PC·배틀넷 계정 | Resolved (2026-10-16: 사용자 본계정, SC:R이 도는 Windows PC) |
| AMB-043-3 | 래더 수집: 범위 기본값(게이트웨이·순위/MMR·매치업·맵·시즌) | Resolved (2026-10-16: 아시아(한국)만, 추적 대상 랭크 A 이상, 매치업·맵 전체, 과거 시즌 포함) |
| AMB-043-4 | 래더 수집: 속도·일일·총량 한도 | Resolved (2026-10-16: 0.5 req/s, 일 1,000개, 총량은 디스크 예산만) |
| AMB-043-5 | 래더 수집: `src` 태그(`scr_public` vs `scr_ladder`) | Resolved (2026-10-16: `scr_ladder` 신설, 040-R14 개정) |
| AMB-043-6 | 래더 수집: 플레이어 식별자·채팅 보관 범위 | Resolved (2026-10-16: 수집 기록에 toon 평문(공용 드라이브만), .rep 원본 유지) |
| AMB-043-7 | 래더 수집: 게이트웨이 Korea(30)만 vs Korea+Asia(45) | Resolved (2026-10-16: Korea + Asia) |
| AMB-043-8 | 래더 수집: 과거 시즌 불가(링크는 최근 25경기, 파일 ~30일) → 주기 수집 여부 | Resolved (2026-10-16: 과거 시즌 조건 삭제, 하루 1회 주기 수집) |
| AMB-050-1 | 승격 게이트·평가 판수·회귀 시나리오 | Resolved (2026-10-15: 초안 수치 확정, M5 이후 조정 제안 가능) |
| AMB-050-2 | M6 3단계에서 구현할 승격 게이트 범위 | Resolved (2026-10-17: 직접 대전 ≥ 55%와 E-Regression만, E-Matrix·E-Elo는 4단계 평가 파이프라인과 함께) |
| AMB-060-1 | Windows 감시 프로세스 자동 시작 구성(리뷰 H Low) | Resolved (2026-10-19: 구성하지 않고 사용자가 직접 실행(D-c) — 060-R6) |
| AMB-061-1 | 원격 CUDA 호스트(VPN 경유) 공유 시 pull·push 정책 | Resolved (2026-10-15: (b) 정책 보강 → 061-R11 shard 재사용·재개 pull·push 재시도·잠금 재확인·미push 복구) |
| AMB-061-2 | 공유 제공 PC 자신의 로컬 캐시 복사 여부 | Resolved (2026-10-15: (b) 예외 추가 → 061-R12 `URANIUM_SHARED_LOCAL=1`이면 복사 없이 원본 읽기, 해시 검증 유지) |
| AMB-061-3 | 체크포인트 보존(061-R6 최신 5개)과 032-R17 ⑤ "비교 run은 지우지 않는다"의 모순(리뷰 C-1·G-D6) | Resolved (2026-10-19: run 설정 `keep_checkpoints`(없으면 5), 코어 비교 config는 전부 보존 — 061-R6) |
| AMB-061-4 | 061-AC1 RNG 비트 일치(dropout RNG 저장 불가)·AC5 1e-4와 CUDA 1e-2의 모순(리뷰 C-6·G-D7) | Resolved (2026-10-19: 불러온 가중치·옵티마이저 상태 비트 일치 + forward는 032-AC1 기준(Vulkan·Flex 1e-4, CUDA가 끼면 1e-2). AC1 "끊김 없이" = 같은 장비(같은 PC)에서 autotune 캐시를 유지·고정하면 비트 동일, 다른 장비나 캐시가 다르면 허용 오차 내, dropout RNG 제외(041-R15 autotune 캐시). 추가 2026-10-19: CUDA 호스트는 같은 PC·같은 캐시여도 032-AC1 허용 오차로 비교, 비트 동일은 Flex·Vulkan만(CUDA atomic add, Windows autotune 캐시 미저장 — 041-R15·061-AC1)) |
| AMB-061-5 | 여유 공간 경고(061-R8·060-R7 90%, 미구현)와 RL 리플레이 자동 삭제(060-R7)↔042-R11 "지우지 않는다" 모순(리뷰 C-3, D) | Resolved (2026-10-19: 공용 드라이브·로컬 캐시 물리 디스크 사용률 80% 경고, 학습 시작·체크포인트 저장마다, 중단 없음(061-R8). RL 리플레이는 지우지 않고 예산 초과는 경고만(060-R7)) |
| AMB-061-6 | 작업자 장치 리뷰 H: `init_run` DONE 미확인, 공유 I/O 오류 시 종료·Free 해석, mtime 기반 만료, 이중 SIGINT, 양보 요청 잔류·양보 중 push 실패 | Resolved (2026-10-19: H-1~H-4와 Medium·Low 모두 코드로. 작업자 자동 인수는 단조 시계로 heartbeat 불변을 stale_after 동안 관측, 수동 `--take-over`는 즉시 인수 + 경고(D-d), 미push 체크포인트 복구는 LATEST가 그 계통에 속할 때만, 자식은 별도 프로세스 그룹 — 061-R5·R11-5·R13·R14·R15, 060-R6) |

## 인덱스

| # | 문서 | 영역 | Status |
|---|---|---|---|
| 000 | [Overview](000-overview.md) | 목표·제약·용어 | **Approved** |
| 001 | [System Architecture](001-system-architecture.md) | 전체 구조 | **Approved** |
| 010 | [Game Protocol](010-game-protocol.md) | 브리지↔에이전트 프로토콜 | **Approved** |
| 011 | [Bridge: OpenBW](011-bridge-openbw.md) | OpenBW 래퍼 | **Approved** |
| 012 | [Bridge: Remastered](012-bridge-remastered.md) | SC:R 래퍼 | Draft |
| 013 | [Version Mapping](013-version-mapping.md) | 1.16.1 ↔ SC:R 정규화 | Draft (명령 인코딩 절만 **Approved** 2026-10-16) |
| 020 | [Env Orchestration](020-env-orchestration.md) | 게임 실행·병렬화 | **Approved** |
| 021 | [Opponent Pool](021-opponent-pool.md) | 학습 상대 | **Approved** |
| 030 | [Observation](030-observation.md) | 관측 feature | **Approved** |
| 031 | [Action Space](031-action-space.md) | 액션·executor | **Approved** |
| 032 | [Model Architecture](032-model-architecture.md) | Burn 모델 | **Approved** |
| 033 | [Reward](033-reward.md) | 보상 설계 | **Approved** |
| 034 | [Action Table](034-action-table.md) | 액션 목록·인자·마스크 클래스 | **Approved** |
| 040 | [Replay Dataset](040-replay-dataset.md) | 리플레이 데이터 | **Approved** |
| 041 | [Imitation Learning](041-imitation-learning.md) | BC 학습 | **Approved** |
| 042 | [RL Training](042-rl-training.md) | RL·리그 | **Approved** |
| 043 | [Ladder Replay Collector](043-ladder-replay-collector.md) | SC:R 래더 리플레이 수집 | **Approved** |
| 050 | [Evaluation](050-evaluation.md) | 평가·게이트 | **Approved** |
| 060 | [Infra: Training Hosts](060-infra.md) | 학습 호스트(Linux Vulkan·CUDA) | **Approved** |
| 061 | [Shared Storage](061-shared-storage.md) | 공용 드라이브 체크포인트·데이터 | **Approved** |

## 마일스톤

| M | 내용 | 관련 spec |
|---|---|---|
| M0 | spec 작성·리뷰 | 전체 |
| M1 | 리스크 스파이크: (a) Burn/Vulkan 벤치, (b) OpenBW 헤드리스 + SC:R CASC 데이터 추출. ~~(c) samase_scarf~~ → M7로 이동(2026-10-15) | 060, 011 |
| M2 | proto + OpenBW 브리지(대전 모드) + 에이전트 서버 최소판 + T0 테란 스크립트 봇으로 게임 완주·fps·결정론 측정 | 010, 011, 021(T0 테란) |
| M3 | hostd·env 스케줄러(Linux 구현, Windows는 `docs/windows.md` 체크리스트) + T0 P/Z 봇 + T1 1단계(UAlbertaBot)·2단계(5종 추가 포팅) + 동시 게임 수 측정(calibrate) | 020, 021 |
| M4 | 리플레이 파이프라인 (SC:R 리플레이 변환 포함), SC:R 래더 리플레이 수집 | 040, 043, 013(명령 인코딩 부분) |
| M5 | 모델 + BC. 데이터 규모가 필요 없는 부분(모델·feature·마스크·executor·보상·BC 골격)은 M4와 병행한다(2026-10-16 사용자 결정) | 030–034, 041 |
| M6 | RL + 리그 | 042, 050 |
| M7 | samase_scarf 분석 스파이크 + SC:R 브리지 + 실전 스모크 | 012, 013, 050 |
