# M6 재build 준비: 전 장비 재빌드(R-1)와 표본 end-to-end(R-2) (2026-10-19)

## 배경
- `m6_pre_bc_review.md` G절의 결정·수정(T1~T4)이 master 6523286까지 반영됐다. 전체 재build(G-D1) 전에 계획 R.1(G-M1 재빌드)과 R.2(표본 end-to-end)를 한다.
- 6523286부터 브리지 C ABI `uranium_bridge_end`가 인자 4개, 프로토콜은 버전 2다(010-R12). 그래서 브리지 .so/.dll, 에이전트, hostd를 모든 PC에서 함께 다시 빌드한다(`m6_bc_preflight.md` 5차 배포 주의).
- 대상 장비: 학습 PC(Windows, CUDA), Linux 호스트 A(Vulkan, 학습 가능), Linux 호스트 B(게임·build 전용). 추출 전용 Windows PC는 참여하지 않는다.

## 예상
- shard schema 5 표본에서 다음을 예상했다.
  - 투혼 CP949 제목(`투혼 1.3`)이 `fighting_spirit`로 분류된다.
  - Train 연타 병합이 사라져 스텝(레이블)이 늘어난다.
  - `result = 0` 비율이 이전 약 70%에서 크게 준다.
  - 첫 스텝이 게임 시작 관측이다.
  - 맵 사전계산 v2에 자원 목록이 있다.
  - 로컬 staging을 거쳐 쓰고, 끝난 뒤 staging이 비어 있다.
  - 같은 리플레이를 다른 장비에서 build해도 결과가 같다.
- BC는 autotune·fusion(041-R15)으로 update가 이전보다 빨라진다(CUDA 약 26초, Vulkan 121–152초).

## 진행 기록

### 0. 사전 점검
- 학습 PC 드라이브(공용 드라이브 포함) 사용률이 82%라 중단하고 보고했다.
- 사용자가 병합된 작업 worktree를 정리했고, 사용률이 64%(여유 약 675GB)가 됐다. 이후 진행했다.
- Linux 호스트 A·B
  - 저장소는 master 7954346이었고 로컬 변경은 없었다.
  - hostd는 systemd가 아니라 수동 프로세스(`setsid nohup`, 홈의 시작 스크립트)로 돌고 있었고, 진행 중인 게임은 없었다.
- 학습 PC: 편집기의 `cargo check`만 돌고 있었다.

### 1. 재빌드 (R-1, G-M1)
| 장비 | 내용 | 결과 |
|---|---|---|
| Linux 호스트 A | hostd 중지 → `git pull --ff-only` → `build.sh` → `build_bridge.sh` → `cargo build --release`(hostd·replay·agent·runner·train) → hostd 재시작 | 6523286, hostd 수신 대기 정상(동시 5게임) |
| Linux 호스트 B | 같음 | 6523286, hostd 수신 대기 정상(동시 4게임) |
| 학습 PC | `build_bridge.ps1` → `cargo build --release --features cuda`(같은 5개) | 브리지 DLL·바이너리 빌드 성공 |

- `.env`에 추가한 키(기존 `.env`는 `.env.bak_20261019`로 백업)
  - 세 장비 모두: `URANIUM_SHARD_STAGING_DIR`(로컬 디스크), `URANIUM_AUTOTUNE_CACHE_DIR`(로컬 고정 폴더)
  - `URANIUM_CACHE_DIR`·`URANIUM_MAPS_DIR`는 이미 있었다.
- 공용 build 큐 스크립트(`run_queue.sh`·`.ps1`)는 실행 파일 경로를 인자로 받아서 고칠 필요가 없었다.
- proto v2 실게임 확인(아래 5절)에서 발견한 것:
  - Linux 호스트의 `URANIUM_MAPS_DIR`에는 대상 맵 `.scx`만 있고 v2 사전계산이 없다.
  - 그래서 모델 세션은 투혼에서 거부된다(031, 의도된 동작).
  - 실전·평가 전에 에이전트가 도는 장비의 `URANIUM_MAPS_DIR`에서 `uranium-replay mapgen <대상 맵 .scx>`를 돌려야 한다.
  - 이번에는 임시 폴더로만 확인했다.

### 2. 표본 build (R-2)
- 표본: 기존 records에서 37개를 골랐다.

  | 출처 | 개수 | 구성 |
  |---|---|---|
  | stardata | 17 | 투혼 CP949 제목 8, 영문 투혼 3, 기타 맵 4, 이전 디싱크 2 |
  | community | 8 | |
  | scr_ladder | 8 | 기록 승자 있음 6 |
  | personal | 4 | |

- 출력 위치: `<shared>/datasets/replays/_sample_20261019/{shards,records,maps}/<장비>`. `URANIUM_MAPS_DIR`도 이 아래로 지정했다.
- 장비별 build 범위
  - 학습 PC: 네 출처 전부
  - Linux 호스트 A: 네 출처 전부
  - Linux 호스트 B: stardata·scr_ladder
  - 모두 `--jobs 4`

| 지표 | 결과 |
|---|---|
| 결과 | 리플레이 37 → written 37시점, desync 2(이전에도 desync였던 2개, 디싱크율 5.4%) |
| 013-R7 왕복 검증 거부 | 0 |
| 맵 분류 | 이전 `other`였던 `투혼 1.3` 8개가 모두 `fighting_spirit`. 나머지 분류는 그대로. shard 기준 투혼 18시점 / 기타 19시점 |
| `result_source` (written 37) | bridge 11, leave 22, none 4(**10.8%**, 이전 표본 약 70%가 0). scr_ladder 기록 승자 8개는 모두 `recorded_winner: matched`(④ 대조 통과) |
| 스텝 수 | 같은 시점 기준 117,028 → 119,857(+2,829, +2.4%). 시점마다 증가분 − 1(첫 스텝)이 이전 `merged_spam`과 거의 같다(37시점 중 10시점은 정확히 같음). `merged_spam` 필드는 없어졌다 |
| Train 레이블 | 표본 합계 8,115. 이전 shard는 schema 4라 같은 게임 단위로 직접 비교하지 못했다. 위 스텝 증가가 병합 제거분이다 |
| 첫 스텝 | `shard-stats` `first_step_not_game_start` 0(네 출처) |
| `is_enemy` 검사 | `is_enemy_violations` 0, `game_info_bad` 0 |
| 맵 사전계산 v2 | 17맵 모두 `version = 2`, 자원 100–138개. 대상 맵은 mapgen 기준 투혼 122, 서킷브레이커 136 |
| 로컬 우선 쓰기 | 세 장비 모두 끝난 뒤 staging 폴더가 비어 있다(복사 → 해시 검증 → rename 완료) |
| `manifest` | 학습 PC 표본 4파일 생성·`--verify` 통과 |

- 장비 간 결정론
  - records: 학습 PC와 Linux 호스트 A의 39줄이 모두 같다.
  - shard-stats 전체: 두 장비가 같다.
  - shard 바이트: `--jobs 4`에서는 personal·scr_ladder는 같고, stardata·community는 달랐다. `--jobs 4`는 완료 순서대로 게임을 쌓기 때문이다. Linux 호스트 A·B끼리는 우연히 같았다.
  - `--jobs 1`로 stardata·community를 두 장비에서 다시 build하자 shard **바이트가 같았다**.
  - 맵 사전계산 파일: 장비 간 모두 같다.
- `label-check` 무효 레이블

  | 출처 | 비율 | 비고 |
  |---|---|---|
  | stardata | 3.92% | |
  | community | 2.78% | |
  | personal | 2.31% | |
  | scr_ladder | **40.3%** | |

  - stardata·community·personal은 10-17 기준 3.58%와 비슷하다.
  - scr_ladder는 두 시점이 거의 전부다.
    - APM 2,376 시점(37,441스텝): 무효 20,444, 대부분 `select_empty`
    - APM 957 시점: 무효 1,119(24.6%)
  - 이 두 시점을 빼면 scr_ladder도 4.5%다.
  - 비정상 APM 래더 게임의 선택 추적 문제로 보이고, 이번 코드 변경의 회귀로는 보이지 않는다. 이전 build에서도 두 시점 모두 written이었다.
  - 다만 전체 데이터에 같은 게임이 섞이면 레이블 손실(가중치 0)이 생긴다. APM 상한·제외 여부는 사용자 결정 대상이다.

### 3. BC 스모크 (실제 `config/bc.toml`, autotune+fusion, `--shards` 표본, 임시 `--out`)
| 항목 | 학습 PC CUDA | Linux 호스트 A Vulkan |
|---|---|---|
| update 수 | 8 | 4 |
| 첫 update (autotune 측정) | 265초 | 254초 |
| 이후 s/update | **약 10.6초** (이전 약 26.4초) | **약 86초** (이전 121–152초) |
| GPU 메모리 최대 | 약 11.3GB(기본 사용 약 0.9GB 포함) | VRAM+GTT 약 6.9GB |
| 프로세스 메모리 최대 | 작업 세트 약 16.1GB | RSS 약 11.6GB |
| 스왑 (G-M2) | — | 시작 0.7GB → 최대 **13.3GB** |
| 기타 | `CUDA_PATH`가 없는 셸에서 첫 시도가 libdevice 오류로 실패(문서에 있는 조건). 설정 후 정상 | autotune 후보 `matmul_naive`가 "Cube count too big"으로 작업 스레드 panic을 출력했지만 학습은 계속됐고 정상 종료 |

- 엔티티 수 N: `bc-profile` 4배치 기준 패딩 N 평균 185, 최대 192.
  - 3차 표본 초반은 N = 64(패딩 단위)였다.
  - 초반 배치는 자원 엔티티(맵 자원 100–138개 중 시야 밖)만큼 늘었다(E-3).
  - 게임 후반 N 분포는 측정하지 않았다.
- `no_result_games` 4(판정 불가 게임 value 가중치 0, 041-R7)가 로그에 나온다.
- G-M2: Linux 호스트 A는 BC 중 스왑을 13GB까지 썼다. 이 측정 하나로 원인을 구분하지 못했다(prefetch, autotune, 통합 메모리 GPU 할당 가운데 어느 쪽인지). 본학습 속도·안정성에 영향을 줄 수 있어 따로 측정이 필요하다.

### 4. 032-AC8·AC1 (Linux 호스트 A Vulkan, `cross_device.rs`, `URANIUM_TEST_DEVICE=vulkan`, release)
- 4개 모두 통과.
- AC8 LSTM backward: 최악 L2 상대 gradient 차이 **5.81e-4**(기준 1e-3).
- AC1 forward: 기본 크기 최악 5.13e-6(기준 1e-4). 탐욕 선택 일치 전부.

### 5. proto v2 실게임 (Linux 호스트 B, 설치된 hostd)
- random 체크포인트(LSTM, `checkpoints/smoke/random-lstm` 새로 생성, FEATURE_SCHEMA 4) 대 T0 테란, 투혼, 4,000프레임 제한 1판. 추론은 flex.
- 결과
  - 두 세션 모두 `final_frame`을 받았다(`[4001, 15, 0]`, `[4001, 33, 0]`).
  - `TimeLimitDraw`, 시야 위반 0.
- 처음 시도는 에이전트 쪽 투혼 v2 사전계산이 없어 세션이 거부됐다(1절 참조). 임시 폴더에 mapgen한 뒤 정상이었다.

### 6. 정리
- 표본 폴더 `_sample_20261019`를 삭제했다(계획 R.2).
- 그 밖에 지운 것: 임시 체크포인트(학습 PC·Linux 호스트 A), 표본 스캔 캐시, Linux 임시 파일.
- 남긴 것
  - 표본 records·build 로그·BC 로그·label-check 결과: 학습 PC 로컬 작업 폴더 `r12/kept`(약 340KB, 공용 드라이브 아님)
  - random 체크포인트: `checkpoints/smoke/random-lstm`(smoke-model plan 경로)
- 기존 shards·records·maps·claims는 건드리지 않았다.

## 전체 재build 전 남은 것
1. **결정 필요**: 비정상 APM 래더 시점(무효 레이블 40%의 원인)을 처리할지(APM 상한·제외).
2. 에이전트가 도는 장비의 `URANIUM_MAPS_DIR`에서 대상 맵 `mapgen`을 다시 돌려 v2를 만든다(실전·평가 전).
3. G-M2: Linux 호스트 A의 BC 스왑을 따로 측정한다.
4. 계획 R.3(기존 결과물 `_prev_20261019` 이동)부터 진행한다.
