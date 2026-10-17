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
8-2. **개인·보안 정보 금지** — 도입 2026-10-15. spec과 모든 문서에 IP 주소, 호스트 이름, 계정, 이메일, 개인 경로, 자격 증명을 쓰지 않는다. 자리표시자(`<server>`)를 쓰고 실제 값은 git에 올리지 않는 `.env`에 둔다([001-R9·R10](001-system-architecture.md)).
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
| AMB-011-1 | MPQ·맵 파일 확보 경로 (사용자 미보유) | Resolved (2026-10-15: 사용자 구버전 확보 + Claude SC:R CASC 추출 조사 병행) |
| AMB-012-1 | samase_scarf 분석 실패 시 대안 선택 | Deferred (M7) |
| AMB-012-2 | 대상 SC:R 빌드 정책, 기존 samase 환경 | Deferred (M7) |
| AMB-012-3 | 실행 환경·안티치트 정책 (→AMB-000-2) | Resolved (2026-10-15: 로컬/LAN 한정) |
| AMB-020-1 | OpenBW 원격 호스트 사양 | Resolved (2026-10-15: 게임 전용 Linux PC 2대 이상, 사양 자동 감지) |
| AMB-020-2 | 최대 게임 시간·무승부 규칙 | Resolved (2026-10-15: 60분 초과 시 무승부, 보상 0 — 점수 판정안에서 변경) |
| AMB-021-1 | BWAPI 4.4 봇 대응 방향, T1 후보 목록 | Resolved (2026-10-15: 단계적 접근 — 4.2 호환 봇+T0 먼저, 4.4 봇은 포팅 비용 측정 후 재확인. 2026-10-16 재확인: Steamhammer·McRave·Stardust·ZZZKBot·Iron 포팅) |
| AMB-021-2 | 상대 봇 맵핵 허용 | Resolved (2026-10-15: 기본 금지, 학습 후반 강화용 별도 상대로만 허용, 평가 미사용) |
| AMB-021-3 | OSS 봇 결정론(020-AC2를 T1 상대에도 요구할지) | Resolved (2026-10-16: 그대로 둔다 — 상대 다양성으로 보고 패치하지 않음, 020-AC2는 T0·셀프플레이로 검증) |
| AMB-030-1 | `enemy_last_seen_age` 채널 의미: (a) 타일을 마지막으로 본 뒤 경과 시간 vs (b) 그 타일에서 적을 마지막으로 본 뒤 경과 시간, 정규화 | Resolved (2026-10-16: 사용자 위임으로 Claude 결정 — (a) 정찰 기록, 채널명 `last_seen_age`, 보임 0·미탐색 1·그 외 log(1+경과)/log(1+86400)) |
| AMB-031-1 | executor(규칙) 범위 | Resolved (2026-10-15: 최소 — 일꾼·건물 선택만, 위치 보정·심시티 스냅·자동 채취 없음) |
| AMB-031-2 | APM 제한 수치 (→AMB-000-1) | Resolved (2026-10-15: APM 제한 없음, delay 1–32 비균등 16 bucket) |
| AMB-031-3 | 맵 사전계산 채널(base_loc·choke·wall_candidate) 적용 범위 | Resolved (2026-10-16: 대상 맵만이 아니라 모든 맵에 맵 무관 알고리즘으로 적용) |
| AMB-032-1 | 정책 구조 (단일 vs 계층형) | Resolved (2026-10-15: 단일 end-to-end) |
| AMB-032-2 | 매치업별 모델 구성 | Resolved (2026-10-15: 단일 공유 모델 + 종족 임베딩) |
| AMB-032-3 | 전략 조건 z 포함 여부 | Resolved (2026-10-15: 포함) |
| AMB-032-4 | 시간 축 코어 | Resolved (2026-10-15: LSTM 시작, M5에서 Transformer 메모리와 비교) |
| AMB-032-5 | 032-AC1의 CUDA 판정: Burn/cubek CUDA matmul이 f32를 TF32로 계산해 Flex 대비 최대 1.3e-3 차이(Vulkan은 1e-6) | Resolved (2026-10-16: 허용 오차 확대 — CUDA ≤ 1e-2, Vulkan ≤ 1e-4 유지) |
| AMB-040-1 | 리플레이 출처 | Resolved (2026-10-15: 공개 데이터셋·커뮤니티·공개 프로/래더·AI 봇 리플레이) |
| AMB-040-3 | 리플레이 라이선스·출처 표기·크롤링 범위 | Resolved (2026-10-15: 내부 학습 전용, 재배포 금지, 약관·robots.txt 준수) |
| AMB-040-4 | BC 데이터 저장 방식 (디스크 < 1TB) | Resolved (2026-10-15: RawFrame 압축 저장 S2→S1 순, 예산 초과분은 재생 스트리밍) |
| AMB-040-2 | 리플레이 품질 필터 기준 | Resolved (2026-10-15: S1 APM≥80·5분+, S2 프로/상위 래더 APM≥150) |
| AMB-040-5 | 040-R8 디싱크 판정 기준(1.16.1은 승패 기록 없음) | Resolved (2026-10-16: 무효 명령 비율·마지막 프레임 미도달·이탈자 우세·SC:R 기록 승자 불일치 중 하나면 디싱크. 임계값 2026-10-16 확정: 무효 15% 초과, 종료 프레임 미달, 이탈 직전 1.5배) |
| AMB-040-6 | BC shard의 정적 맵 정보 출처 (feature 추출용) | Resolved (2026-10-16: shard game마다 `GameInfo`(static_map 포함) 저장, schema_version 상향 후 재빌드) |
| AMB-043-1 | 래더 수집: 040 데이터 정책(약관 준수) 예외 허용 여부 | Resolved (2026-10-16: EULA 해석 위험 감수하고 예외 허용, 내부 학습 전용·재배포 금지 유지) |
| AMB-043-2 | 래더 수집: 실행 PC·배틀넷 계정 | Resolved (2026-10-16: 사용자 본계정, SC:R이 도는 Windows PC) |
| AMB-043-3 | 래더 수집: 범위 기본값(게이트웨이·순위/MMR·매치업·맵·시즌) | Resolved (2026-10-16: 아시아(한국)만, 추적 대상 랭크 A 이상, 매치업·맵 전체, 과거 시즌 포함) |
| AMB-043-4 | 래더 수집: 속도·일일·총량 한도 | Resolved (2026-10-16: 0.5 req/s, 일 1,000개, 총량은 디스크 예산만) |
| AMB-043-5 | 래더 수집: `src` 태그(`scr_public` vs `scr_ladder`) | Resolved (2026-10-16: `scr_ladder` 신설, 040-R14 개정) |
| AMB-043-6 | 래더 수집: 플레이어 식별자·채팅 보관 범위 | Resolved (2026-10-16: 수집 기록에 toon 평문(공용 드라이브만), .rep 원본 유지) |
| AMB-043-7 | 래더 수집: 게이트웨이 Korea(30)만 vs Korea+Asia(45) | Resolved (2026-10-16: Korea + Asia) |
| AMB-043-8 | 래더 수집: 과거 시즌 불가(링크는 최근 25경기, 파일 ~30일) → 주기 수집 여부 | Resolved (2026-10-16: 과거 시즌 조건 삭제, 하루 1회 주기 수집) |
| AMB-041-1 | BC 혼합 선택 레이블 판정(`select_cap`: 일부 유닛만 자격이 있는 선택) | Resolved (2026-10-17: 사용자 위임으로 Claude 결정 — 게임 엔진 동작 기준. 자격 없는 유닛을 레이블에서 빼고 남는 유닛이 없을 때만 무효(041-R11). 함께 확인한 엔진 규칙: 어택땅 자격 = 이동 가능(034), 선택 추적 = 게임 선택 규칙(040-R3), 떠 있는 건물 우클릭 = 이동(040-R11), 랠리는 건설 중에도 가능(034)) |
| AMB-042-1 | 학습 기간·장비 예산 | Resolved (2026-10-15: 기간 제한 없음, 전체 리그 구성) |
| AMB-042-2 | TvP·TvZ 상대 다양성 (P/Z 상대 정책 추가 여부) | Deferred (M6) (2026-10-17 M6 시작 재확인: T0·T1 봇으로 시작, M6 중간 점검에서 정체 여부 보고 후 재확인) |
| AMB-042-3 | RL 초기 설정값(정책 지연 임계값, 가중치 배포 K, 스냅샷 N, 커리큘럼 전환 조건, 헤드별 엔트로피, KL 감쇠, 시분할, PFSP, 종족·Random 비율, 배치·옵티마이저) | Resolved (2026-10-17: 사용자 위임으로 Claude 결정 — IMPALA·AlphaStar 기준 축소값을 042 "초기 설정값" 절·`config/rl.toml`에 기록, 전부 042-R9 튜닝 대상, 042-AC1 후 조정 제안 가능. 첫 RL 학습 호스트는 CUDA PC) |
| AMB-050-1 | 승격 게이트·평가 판수·회귀 시나리오 | Resolved (2026-10-15: 초안 수치 확정, M5 이후 조정 제안 가능) |
| AMB-061-1 | 원격 CUDA PC(WireGuard 경유) 공유 시 pull·push 정책 | Resolved (2026-10-15: (b) 정책 보강 → 061-R11 shard 재사용·재개 pull·push 재시도·잠금 재확인·미push 복구) |
| AMB-061-2 | 공유 제공 PC 자신의 로컬 캐시 복사 여부 | Resolved (2026-10-15: (b) 예외 추가 → 061-R12 `URANIUM_SHARED_LOCAL=1`이면 복사 없이 원본 읽기, 해시 검증 유지) |

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
| 060 | [Infra: Training Hosts](060-infra-bc250.md) | 학습 호스트(BC250·CUDA PC) | **Approved** |
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
