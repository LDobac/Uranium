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
   - **결정·구현 (사용자 결정 2026-10-19, 040-R19, AMB-040-11)**
     - APM 상한이 아니라 데이터 품질 기준으로 정했다. build에서 테란 시점의 무효 레이블 비율이 **20% 이상**이면 shard에 쓰지 않고 `filtered`로 기록한다. 사유 형식은 `invalid_labels>=20% (<비율>%, <무효>/<전체>) (040-R19)`이다.
     - 무효 레이블 판정은 BC 로더·`label-check`와 같은 함수(`uranium_env::features::label`)를 쓴다.
     - 같은 커밋에서 재개 구멍도 고쳤다. 한 시점은 바로 기록되고(filtered 등) 다른 시점은 shard flush 전에 중단되면 그 시점이 영영 빠지던 문제다. 이제 재개는 (리플레이, 시점) 단위로 판정한다.
   - **표본 재build** (학습 PC, scr_ladder 8리플레이, `--jobs 1`)
     - 결과: written 8, filtered 2
     - 걸러진 두 시점: APM 2,376 시점 54.6%(20,444/37,441), APM 957 시점 24.6%(1,119/4,556)
     - 무효 수가 위 `label-check` 결과와 정확히 같다.
     - 출력: 학습 PC 로컬 `sc_data/t3c`
   - **비용**
     - 검사는 스텝당 약 0.54ms다(feature 전체 추출 포함).
     - 10시점 합계는 검사 29.5초, 재생 159초다. build 벽시계 189초의 **+15.6%**에 해당한다.
     - 학습과 똑같은 검사를 유지하려고 이 비용을 받아들였다(메인 결정).
2. 에이전트가 도는 장비의 `URANIUM_MAPS_DIR`에서 대상 맵 `mapgen`을 다시 돌려 v2를 만든다(실전·평가 전).
3. G-M2: Linux 호스트 A의 BC 스왑을 따로 측정한다.
4. 계획 R.3(기존 결과물 `_prev_20261019` 이동)부터 진행한다.
   - 사용자 결정(2026-10-19)으로 이동 없이 바로 삭제했다. 아래 R-3 절 참조.

## R-3 전체 재build 준비 (2026-10-19 16시, master d452c29)

### 배경
- 6523286 뒤 master에 T3c(040-R19 무효 레이블 20% 이상 시점 `filtered`, 레이블 검사 `uranium_env::features::label` 이동, (리플레이, 시점) 단위 재개)가 들어왔다. 브리지·proto 변경은 없다.
- 기존 shards·records는 재사용하지 않는다(사용자 결정 2026-10-19). `_prev` 이동 없이 삭제한다.
- 예상: 바이너리만 바꾸면 되고 hostd는 소스 변경이 없어 재시작이 필요 없을 것으로 봤다.

### 1. 재빌드
| 장비 | 내용 | uranium-replay sha256(앞 8자리) | hostd |
|---|---|---|---|
| 학습 PC | `cargo build --release --features cuda`(replay·train·agent·runner·hostd) | 2bef85e3 (train 2c96e483) | 실행하지 않음 |
| Linux 호스트 A | `git pull --ff-only`(6523286 → d452c29) → `cargo build --release`(같은 5개) | 1342618c (train 3cd8104e) | 바이너리가 바뀜(a3d46a13 → d0e61b01) → 게임 없음 확인 후 시작 스크립트로 재시작, 수신 대기 정상(동시 5게임) |
| Linux 호스트 B | 같음 | fdc63dd4 (train 7b2bb60b) | 바이너리 그대로(a3d46a13) → 재시작 안 함 |

- 브리지는 다시 빌드하지 않았다.
- hostd 소스(hostd·proto)는 6523286 이후 바뀌지 않았다. 그런데도 Linux 호스트 A만 hostd가 다시 링크됐다(6분 28초, 의존성 다수 재컴파일). 호스트 B는 49초였다.
- Linux 호스트 A·B는 rustc 1.99.0이 같지만 배포본이 다르다(rustup / 배포판 패키지). 그래서 바이너리 해시가 서로 다르다. 출력 결정론은 R-2에서 장비 간 같다고 확인했다.

### 2. 삭제 전 기록
- 옛 `build/jobs.txt`(78줄, 머리 줄 포함)는 내용을 그대로 보관했다가 다시 썼다.

  | 출처 | 파트 |
  |---|---|
  | stardata | 64 |
  | community | 11 (synnaeve_gosu 8, iccup_users 2, osu_pvt 1) |
  | scr_ladder | 1 |
  | personal | 1 |

- 옛 records 합계(레코드 줄 수, `outcome`별)

  | 출처 | written | filtered | desync | unplayable | error |
  |---|---|---|---|---|---|
  | stardata | 34,714 | 32,309 | 727 | 300 | 0 |
  | community | 393 | 6,464 | 1,556 | 12 | 1 |
  | scr_ladder | 609 | 1,458 | 38 | 0 | 3 |
  | personal | 154 | 434 | 5 | 8 | 0 |

  - `records/fix-desync/`(디싱크 오판 재처리): stardata written 91, community 3, scr_ladder 2

### 3. 사용 중 확인
- 학습 PC: uranium 프로세스가 없었다.
- Linux 호스트 A·B: hostd만 돌고 있었다. 게임 자식 프로세스, build·학습·bc-eval은 없었다.
- 추출 전용 Windows PC는 SSH 연결이 되지 않았다(꺼져 있는 것으로 보임). 이 PC는 SSH 세션에서 공용 드라이브를 쓸 수 없고, 이번 재build 참여 대상도 아니다.

### 4. 삭제 목록
| 위치 | 대상 | 크기 |
|---|---|---|
| 공용 | `datasets/replays/shards/`(네 출처, `fix-desync-p0`, 출처별 `manifest.json`) | 약 37GB |
| 공용 | `datasets/replays/records/`(`fix-desync/` 포함) | 24MB |
| 공용 | `datasets/replays/build/claims/`, `build/logs/`, `build/jobs.txt` | 약 0.4MB |
| 공용 | `datasets/replays/build/maps/`(v1 사전계산) | 11MB |
| 공용 | `datasets/replays/quarantine/`(`stardata_p16`) | 557MB |
| 학습 PC | `URANIUM_MAPS_DIR`의 v1 사전계산 `*.toml` 1,252개(`.scx` 2개는 유지) | 약 12MB |
| 학습 PC | `URANIUM_CACHE_DIR/scans/` 스캔 캐시 4개 | 24KB |
| 학습 PC | 로컬 앱 데이터 `uranium-build/2f2b0a0`(옛 build 큐용 바이너리 사본) | 209MB |
| Linux 호스트 A | `URANIUM_CACHE_DIR/datasets/`(BC용 옛 shard 캐시) | 약 37GB |
| Linux 호스트 A | `URANIUM_CACHE_DIR/scans/` | 비어 있음 |

- 확보한 공간
  - 학습 PC 드라이브(공용 드라이브 포함): 39.6GB, 사용률 65% → 63%
  - Linux 호스트 A: 39.3GB, 사용률 38% → 29%
- 학습 PC 공용 드라이브가 로컬 디스크라 학습 PC에는 따로 받은 shard 캐시가 없다.
- Linux 호스트 B에는 데이터 캐시·바이너리 사본이 없었다.
- 유지: `raw/`, `build/hashes/`, `build/run_queue.*`, `ladder/`, 맵 `.scx`, `URANIUM_CACHE_DIR/checkpoints`·`calibrate`, `checkpoints/smoke`

### 5. 새 큐
- 빈 `shards/`, `records/`, `build/claims/`, `build/logs/`, `build/maps/`를 다시 만들었다.
- `build/jobs.txt`는 옛 파일과 바이트가 같다(78줄, 작업 77개).
- 모든 raw 폴더와 skip-hashes 파일 4개(`skip_{iccup_users,osu_pvt,synnaeve_gosu,personal}.txt`)가 있다.
- Linux 호스트 A·B에서 공용 마운트로 빈 폴더와 78줄을 확인했다.

### 6. `.env` 확인
- 세 장비 모두 `URANIUM_SHARD_STAGING_DIR`가 있고, staging 폴더는 비어 있다.
- 세 장비 모두 `URANIUM_MAPS_DIR`가 있다. 큐 스크립트는 이 값을 `build/maps`로 덮어쓴다.

### 상태
- 큐는 시작하지 않았다. `run_queue`를 시작하면 바로 전체 재build(계획 R.4)로 넘어간다.

## R-4 전체 재build 시작 (2026-10-19 16:35, master d452c29)

### 배경과 예상
- R-3에서 비운 공용 경로에 옛 큐와 같은 `jobs.txt`(작업 77개)로 세 장비가 다시 build한다.
- 예상
  - written/filtered 비율은 옛 합계와 비슷하다.
  - 040-R19 걸러짐은 stardata에서 드물다. scr_ladder는 비정상 APM 시점이 있어 더 많다.
  - 끝까지 약 15시간 걸린다.

### 실행
| 장비 | 시작 | jobs | 방식 |
|---|---|---|---|
| Linux 호스트 A | 16:35(첫 파트 16:40) | 8 | `ssh -n -f` + `setsid nohup run_queue.sh` |
| Linux 호스트 B | 16:35(첫 파트 16:43) | 8 | 같음 |
| 학습 PC | 16:59 | 12 | 세션과 분리된 프로세스로 `run_queue.ps1` 실행 |
| Linux 호스트 C(4C/8T, RAM 14 GiB, swap 4 GiB) | 18:31(첫 파트 stardata_p07) | 8 | Linux 호스트 A와 같음. `.env` 키 같음, staging은 로컬 |

- 실행 파일: 각 장비 로컬 `uranium-build/d452c29/`에 복사한 d452c29 `uranium-replay`와 브리지
  - Linux는 `URANIUM_BRIDGE_LIB`로 브리지를 지정했다.
  - 학습 PC는 `-BridgeLib`로 지정했다.
- 학습 PC는 처음에 권한 거부로 띄우지 못했다. 사용자가 권한을 준 뒤 16:59에 시작했다.
- Linux 호스트 C는 18:31에 합류했다. d452c29 바이너리를 그 호스트에서 빌드했다. 합류 전 표본(scr_ladder 12, stardata 20)을 `--jobs 1`로 돌려 Linux 호스트 A와 비교했고, shard·records가 바이트 단위로 같았다.
- 시각은 각 장비 시계 기준이다. 학습 PC 시계가 Linux보다 약 8분 늦다.
- Linux 호스트 A·B: build를 방해하지 않게 `nice`로 대상 맵 `mapgen`을 돌렸다. 각 장비 `URANIUM_MAPS_DIR`에 v2 사전계산이 생겼다(서킷브레이커 자원 136, 투혼 122). 1절의 남은 일 2를 Linux 두 대에서 해결했다.

### 첫 파트 결과
| 장비 | 파트 | 걸린 시간 | 리플레이 | written | filtered | desync | unplayable | error | 040-R19 |
|---|---|---|---|---|---|---|---|---|---|
| Linux 호스트 A | scr_ladder p0 | 53분 | 2,000 | 570 | 1,494 | 41 | 0 | 3 | 36 |
| Linux 호스트 B | stardata p00 | 58분 | 1,026 | 519 | 536 | 4 | 3 | 0 | 2 |
| 학습 PC | stardata p01 | 29분 | 1,026 | 562 | 484 | 9 | 10 | 0 | 3 |

- 옛 합계와의 비교
  - stardata written/filtered 비율은 0.97·1.16이다. 옛 합계는 1.07이었다.
  - scr_ladder는 written 570 / filtered 1,494 / desync 41 / error 3이다. 옛 build는 609 / 1,458 / 38 / 3이었다.
  - error는 모두 013-R7 왕복 검증 실패(유닛 인덱스 범위 초과)다. 옛 build와 같은 수준이다.
- shard 쓰기 검증 실패는 없었다. staging 폴더는 파트 사이에 비어 있었다.
- 040-R19 걸러짐 비율(written + 040-R19 기준)
  - stardata: 0.4%·0.5%
  - scr_ladder: **5.9%**(36/606)
- scr_ladder 040-R19 걸러짐 36건
  - 무효 비율은 최소 20.1%, 중앙값 24.5%, 최대 54.6%다.
  - 20–30%가 27건, 30–50%가 8건, 50% 이상이 1건이다.
  - APM은 295–2,542, 중앙값 약 740이다.
- 학습 PC는 jobs 12에서 `unplayable`(브리지 연결 전 런처 종료)이 10건이다. 옛 stardata 평균은 파트당 약 4.7건이었다. 부하 영향인지 다음 파트에서 본다.
- `resumed_skip`은 시작할 때 읽은 같은 출처 records 줄 수다. 실제로 건너뛴 수가 아니다.

### 예상 완료
- 처리량은 학습 PC 약 29분/파트, Linux 호스트 A·B 약 55분/파트다. 합쳐서 약 14분/파트다.
- 남은 74파트 중 stardata가 약 62개다. community·personal은 중복 제외가 많아 빠르다.
- 완료는 약 15–16시간 뒤인 10-20 오전 9–10시경으로 예상한다.

## R-4 재개 (2026-10-20 00:10경, 바이너리 master aa187b2)

### 배경
- 10-19 밤 액션 감사 F7·F12 수정(T9, 레이블러 2, shard schema 6, `Unload_Unit`)을 기다리려고 아직 시작하지 않은 50파트를 보류했다.
  - 보류 방법: claim 파일 내용을 `hold-f7f12`로 만들어 두었다. 목록은 `<shared>/datasets/replays/build/held_f7f12.txt`.
  - 네 장비 큐는 진행 중이던 파트를 끝내고 `queue empty`로 멈췄다(마지막 22:02–23:05).
- 사용자 결정(2026-10-20): 남은 마스크·브리지 수정을 기다리지 않고 지금 재개한다.
- 결과적으로 재build는 두 레이블러가 섞인다(040 "Shard 포맷", schema 5는 레이블러 1로 읽음).
  - 옛 레이블러(d452c29, schema 5) 27파트: scr_ladder p0, stardata p00–p25
  - 새 레이블러(aa187b2, schema 6, 레이블러 2) 50파트: stardata p26–p63 38개, community 11개(synnaeve_gosu 8, iccup_users 2, osu_pvt 1), personal 1개

### 재빌드
| 장비 | 내용 | uranium-replay / 브리지 sha256(앞 8자리) |
|---|---|---|
| 학습 PC | `build_bridge.ps1` → `cargo build --release -p uranium-replay` | 로컬 앱 데이터 `uranium-build/aa187b2/` |
| Linux 호스트 A·B | `git pull --ff-only` → `cargo build --release -p uranium-bridge-openbw -p uranium-replay` → 브리지 cmake를 별도 폴더 `bridge/openbw/build-aa187b2`로 빌드 | 11a5fce7 / 884a7a14(두 장비 같음) |
| Linux 호스트 C | 저장소 `git pull --ff-only`, 새 소스 worktree `~/uranium-build/src-aa187b2`(OpenBW·BWAPI 빌드 결과는 src-d452c29에서 복사) → `build_bridge.sh` → `cargo build --release -p uranium-replay` | ed85d088 / 15a71297 |

- 실행 파일은 각 장비 `uranium-build/aa187b2/`에 복사했다(d452c29와 같은 방식).
- Linux 호스트 A·B는 hostd가 저장소의 `bridge/openbw/build/liburanium_bridge.so`(PROTOCOL 2)를 쓴다. 이 파일을 덮어쓰지 않으려고 브리지를 별도 cmake 폴더로 빌드했다. hostd는 재시작하지 않았다. PROTOCOL 3 hostd·agent 재배포는 남은 실게임 쪽 수정 뒤에 한다.

### 표본 확인
- 학습 PC에서 scr_ladder 리플레이 3개를 새 바이너리로 build했다(출력은 작업 tmp, 공용 shards·records 아님, `--jobs 3`, 8.6초).
- 결과: written 2시점(1리플레이), filtered 2(제외 맵 1, APM 1), error 0
- records 전부 `labeler_version: 2`, unknown 0
- `shard-stats`: `labeler_versions`가 2만 있음(schema 6 헤더)

### 재개
- `held_f7f12.txt`의 50개 claim을 내용이 `hold-f7f12`인 것만 지웠다(50개 모두 해당). 다른 claim·done 파일은 그대로다.
- 네 장비 큐를 이전과 같은 이름·jobs로 다시 띄웠다(학습 PC 12, Linux 8).

  | 장비 | 첫 파트 |
  |---|---|
  | 학습 PC | stardata_p29 |
  | Linux 호스트 A | stardata_p26 |
  | Linux 호스트 B | stardata_p27 |
  | Linux 호스트 C | stardata_p28 |

- 시작 직후 네 파트의 records 줄은 모두 `labeler_version: 2`이고 error는 0이다.
- 예상 완료: 네 장비 합쳐 약 12분/파트, 50파트 → 약 10시간, 10-20 오전 10시경

## R-5 재build 뒤 검증과 shard 손상 수리 (2026-10-20, master 7b0879e)

### 배경과 예상
- R-4 전체 재build가 10-20 07:56에 끝났다. 77파트 모두 `.done`이고, 네 장비 큐는 모두 `queue empty`로 멈췄다.
  - 레이블러가 섞여 있다. 레이블러 1(shard schema 5)이 27파트, 레이블러 2(schema 6, aa187b2)가 50파트다.
  - 040-R19 무효 레이블 필터는 액션 감사 마스크 수정(`m6_action_audit.md` §11) 이전 마스크로 판정했다. 받아들이기로 했고, 다시 build하지 않는다.
- 10-19에 받은 래더 리플레이 약 1,000개는 scr_ladder 파트를 build한 뒤에 `raw/`에 들어와서 shard에 없었다.
- 할 일은 다음과 같다.
  - 래더 증분 build
  - manifest와 검증, `shard-stats`
  - 분할 재집계와 `dataset_stats.md`
  - 숫자 표
  - `[steps]` 갱신과 BC 스모크
  - 맵 사전계산 동기화
  - 고정 바이너리 사본 정리
- 예상
  - 투혼 시점 수는 헤더 CP949 수정 뒤라 약 11,100이다.
  - 040-R19 걸러짐은 stardata에서 드물고, scr_ladder는 약 6%다.
  - 검증은 통과한다.

### 1. 래더 증분 build (학습 PC, 08:35–08:58, f513b92)
- jobs.txt의 `scr_ladder-p0`와 같은 인자를 썼다. 출력 폴더는 `shards/scr_ladder/scr_ladder-p0`, records는 `scr_ladder_scr_ladder-p0.jsonl`, `URANIUM_MAPS_DIR=build/maps`, `--jobs 12`.
- 재개가 기존 records 2,108줄을 건너뛰고 새 리플레이 1,000개만 처리했다.
  - written 378, filtered 694, desync 38, error 1
  - filtered 사유: 제외 맵 377, APM 145, 5분 미만 88, 인원 45, 040-R19 39
  - 새 shard는 `shard_00043`~`00062`다.
- 로그: `<shared>/datasets/replays/build/logs/r5-incremental/`
- 10:08–10:35에는 다른 세션이 오늘 수집분 1,000개를 Linux 호스트 B에서 f513b92로 build했다.
  - 결과: written 285, filtered 738, desync 19, records 4,261줄
  - 새 shard는 `shard_00063`~`00080`이다.

### 2. 발견: 검증을 통과한 shard 7개가 공용 드라이브에서 손상됨
- 첫 `shard-stats`에서 stardata 4파트가 "Unknown frame descriptor" 또는 "Data corruption detected"로 실패했다.
- 그래서 shard 2,023개를 모두 끝까지 압축 해제해 보았고, 7개가 풀리지 않았다.

  | 파트(쓴 장비) | 파일 | 크기 | 손상 모양 |
  |---|---|---|---|
  | scr_ladder p0 (Linux 호스트 A) | shard_00014 | 18,016,519 | 앞 8MiB가 0 |
  | scr_ladder p0 (Linux 호스트 A) | shard_00034 | 18,078,644 | 앞 8MiB가 0 |
  | stardata p00 (Linux 호스트 B) | shard_00010 | 22,208,552 | 앞 8MiB가 0 |
  | stardata p00 (Linux 호스트 B) | shard_00023 | 4,194,304 | 정확히 4MiB에서 잘림 |
  | stardata p03 (Linux 호스트 A) | shard_00022 | 20,972,798 | 0x400000부터 16MiB가 0 |
  | stardata p08 (Linux 호스트 A) | shard_00003 | 19,573,667 | 앞 16MiB가 0 |
  | stardata p27 (Linux 호스트 B) | shard_00004 | 20,853,606 | 0x400000부터 8MiB가 0 |

- 손상 시각은 10-19 17:00–18:44와 10-20 00:22다.
- 모두 Linux 호스트 A·B가 CIFS 마운트로 쓴 파일이다. 학습 PC(공유를 제공하는 PC)와 Linux 호스트 C가 쓴 파일은 깨끗했다.
- 손상 구간이 모두 4MiB(마운트 `wsize`) 단위다.
- 원인 조사
  - Linux 호스트 A·B는 커널 6.18 LTS, cifs 모듈 2.57이다. 마운트 옵션은 `vers=3.1.1, cache=strict, hard, rsize=wsize=4MiB, actimeo=1, closetimeo=1`이다.
  - 해당 시간대 커널 로그에 CIFS·SMB 메시지(재연결, I/O 오류)가 하나도 없다. OOM도 없다.
  - Linux 호스트 C는 커널 7.0, cifs 2.59이고 마운트 옵션은 같으며 손상이 없었다.
  - 학습 PC의 System 로그와 SMBServer 로그에도 관련 이벤트가 없다.
  - 결론: 클라이언트가 쓰기 요청 일부를 조용히 잃었다. 그런데 040-R17 검증은 자기 페이지 캐시에서 다시 읽어 해시가 맞았다고 본다. 로그로 확정할 증거는 없다.
  - 사용자 결정(2026-10-20): Linux A·B 마운트 설정은 그대로 둔다.
- 영향: 읽을 수 없는 게임은 132시점이었다. scr_ladder 43, stardata p00 36, p03 18, p08 19, p27 16.

### 3. 코드 수정 (spec 개정, 사용자 결정 2026-10-20, AMB-040-12)
master 67488fc·1ce972c·7b0879e. `cargo test --workspace` 통과 303, 실패 0.

- **040-R17 다시 읽기를 서버 기준으로**
  - 복사본을 fsync한 뒤 클라이언트 캐시를 거치지 않고 읽는다(`uranium_replay::share_io`).
    - Linux: `posix_fadvise(DONTNEED)` + `O_DIRECT`
    - Windows: `FILE_FLAG_NO_BUFFERING`
    - 이 방식이 거부되면 일반 읽기로 넘어가고 경고를 한 줄 남긴다.
  - Linux 호스트 A의 CIFS 마운트에서 `O_DIRECT` 읽기가 받아들여지고, 일반 읽기와 결과가 같은 것을 확인했다.
- **061-R4 체크포인트 push**: 같은 약점(캐시에서 다시 읽기)이 있어서 같은 방식으로 바꿨다(`Manifest::verify_on_server`).
- **manifest 생성**
  - 해시를 만들기 전에 하위 폴더의 shard를 모두 끝까지 디코드해 본다.
  - 실패하면 실패한 shard를 모두 나열하고 manifest를 쓰지 않는다.
- **새 shard에 zstd 내용 체크섬**: 체크섬 없는 이전 shard도 그대로 읽는다.
- **040-R20 (신설): 한 출처 안의 같은 리플레이 파일은 한 게임**
  - 출처 전체의 정렬된 목록(`--part`로 나누기 전)에서 앞선 파일만 처리한다. 나머지는 `duplicate file in this source (040-R20): same file as global index <n>`으로 거른다.
  - `shard-stats`는 하위 폴더까지 읽고, 같은 (리플레이, 시점)이 두 번 나오면 실패한다.
- **`dataset-stats`**: 중복 제거 키에 출처와 040-R20 기록인지 여부를 넣었다.

### 4. 수리 (학습 PC, 7b0879e 바이너리와 브리지)
삭제와 records 수정은 사용자 승인(2026-10-20)을 받았다.

- **지운 shard 7개**는 위 표와 같다. 목록(경로, 크기, 시각, 해시 앞 12자리)은 `<shared>/datasets/replays/build/r5-backup/deleted_shards.tsv`에 있다.
- **지운 records 줄**: written 132줄
  - 대상은 읽히는 shard에 없는 게임이다. (출처, 리플레이 해시, APM, 매치업)으로 대조했다. shard의 `perspective_slot`은 records의 `view.perspective`와 값이 다르기 때문이다.
  - 파일별: `stardata_p00` 36, `stardata_p03` 18, `stardata_p08` 19, `stardata_p27` 16, `scr_ladder_scr_ladder-p0` 43
  - 지운 줄 원문은 `build/r5-backup/dropped-records/`에 보관했다.
  - scr_ladder는 Linux 호스트 B build가 끝난 뒤의 최종 상태에서 다시 계산했다. 그 build가 새로 쓴 shard 18개는 모두 정상이었다.
- **재개 build**(jobs 12): 지운 게임만 다시 처리했다.

  | 파트 | 걸린 시간 | 결과 |
  |---|---|---|
  | stardata p00 | 106초 | written 36 |
  | stardata p03 | 50초 | written 18 |
  | stardata p08 | 63초 | written 19 |
  | stardata p27 | 96초 | written 16 |
  | scr_ladder | 70초 | written 43 |

  - 040-R19로 새로 걸러진 시점은 없다.
  - 다시 만든 시점은 레이블러 2이고 액션 감사 이후 마스크를 쓴다. 그래서 stardata p00·p03·p08에서 73시점이 레이블러 1에서 2로 바뀌었다.
- **personal 전체 재build**
  - 옛 shard 7개(153게임)와 records는 `build/r5-backup/personal/`로 옮겼다(삭제하지 않음).
  - 결과: written 152, 040-R20 중복 2파일. 중복 2파일은 같은 리플레이 사본이고, 하나는 written 게임, 하나는 "1 active players"로 걸러진 리플레이다.
  - shard 게임 수 152가 고유 게임 수와 같다.
- 로그: `build/logs/r5-repair/`

### 5. 검증 결과 (7b0879e)
- **manifest**: 네 출처 모두 다시 만들었고, 만들 때 모든 shard 디코드에 성공했다. `--verify`도 통과했다.

  | 출처 | 파일 | 크기 |
  |---|---|---|
  | stardata | 1,913 | 38.4GB |
  | scr_ladder | 81 | 1.47GB |
  | community | 43 | 0.82GB |
  | personal | 7 | 0.09GB |

- **`shard-stats`**(하위 폴더 포함): 네 출처 모두 다음이 0이다.
  - `is_enemy_violations`, `game_info_bad`, `first_step_not_game_start`, `duplicate_games`
  - shard 게임 수는 records written과 같다: stardata 34,664, scr_ladder 1,233, community 396, personal 152.
- **처음부터 다시 한 디코드 검사**: 손상 shard를 지우고 다시 만든 뒤 전체 shard를 다시 압축 해제했고, 오류가 0이다.

### 6. 숫자
records 기준. 같은 출처·리플레이·시점이 여러 번 있으면 마지막 기록만 센다. 040-R20 기록은 따로 센다.

| 출처 | 리플레이 | written | filtered | desync | unplayable | error |
|---|---|---|---|---|---|---|
| stardata | 65,646 | 34,664 | 32,441 | 644 | 301 | 0 |
| community | 8,219 | 396 | 6,552 | 1,552 | 12 | 1 |
| scr_ladder | 4,000 | 1,233 | 2,926 | 98 | 0 | 4 |
| personal | 574 | 152 | 436 | 5 | 8 | 0 |
| 이전 build 합계(R-3 기록) | | 35,870 | 40,665 | 2,326 | 320 | 4 |

- 이전 build 출처별 값(written / filtered / desync / unplayable / error)
  - stardata: 34,714 / 32,309 / 727 / 300 / 0
  - community: 393 / 6,464 / 1,556 / 12 / 1
  - scr_ladder(리플레이 2,000개였음): 609 / 1,458 / 38 / 0 / 3
  - personal: 154 / 434 / 5 / 8 / 0
- filtered 사유
  - stardata: APM 29,697, 5분 미만 2,364, 인원 244, 040-R19 132, 맵 크기 4
  - community: 다른 출처 중복 5,092, APM 1,306, 인원 124, 5분 미만 30
  - scr_ladder: 제외 맵(040-R16) 1,438, APM 714, 5분 미만 443, 인원 248, 040-R19 83
  - personal: 인원 368, 5분 미만 43, APM 22, 040-R20 중복 2, 040-R19 1
- **040-R19 무효 레이블 필터**(시점 수, 비율 = 걸러짐 ÷ (written + 걸러짐))

  | 출처 | 레이블러 1 | 레이블러 2 |
  |---|---|---|
  | stardata | 50(49게임) 0.35% | 82(79게임) 0.40% |
  | scr_ladder | 36(36게임) 6.39% | 47(45게임) 6.24% |
  | personal | — | 1 0.65% |
  | community | — | 0 |

  - 사유 문자열: `invalid_labels>=20% (<비율>%, <무효>/<전체>) (040-R19)`
  - 레이블러 2 판정 중 오늘 증분 build(f513b92, 마스크 수정 후) 분량이 섞여 있다.
- **대상 맵**(게임 / 시점)
  - 투혼: 10,476 / 11,211. 증분 전에는 시점 11,101로, 예상한 약 11,100과 맞는다.
  - 서킷브레이커: 725 / 781
- **승패 판정 불가**(result 0, value 가중치 0): 7,421 / 36,445 = **20.36%**
  - 출처별: stardata 20.94%, community 18.94%, personal 56.58%, scr_ladder 0.08%
  - 재build 전 약 70%에서 크게 줄었다.
- **privileged 없는 스텝**(E-10, `split-stats`): 네 출처 모두 0게임 / 0스텝(0%)
- **디싱크 폐기율**(040-AC2): 전체 5.9%
  - 출처별: stardata 1.8%, community 79.7%, scr_ladder 7.4%, personal 3.2%
  - 맵 패밀리별: 투혼 1.5%, 서킷브레이커 0.4%, 기타 8.0%
- **분할**(041-R12, 시점 수. 로더는 시점 하나를 게임 하나로 센다)

  | 맵 그룹 | train | val | test |
  |---|---|---|---|
  | 대상 맵 | 9,608 | 1,192 | 1,192 |
  | 기타 맵 | 23,264 | 1,189 | 없음 |

  - 대상 맵 test 셋 출처별(시점): stardata 1,153, scr_ladder 17, personal 17, community 5
- 맵·매치업·출처·stage·분할별 표: `reports/dataset_stats.md`(040-AC2·AC4, 생성 명령은 파일 첫머리)

### 7. 학습 길이 `[steps]` (041-R14, `bc-epoch`, lanes 32, seq_len 64, seed 1)
| 항목 | 이전 | 새 값 |
|---|---|---|
| s1 | 36,985 | **38,584** (S1 32,872시점, 78.0M 스텝) |
| s2 / ablation_a | 7,915 | **11,319** (S2 9,608시점, 22.9M 스텝) |
| s2_warmup / ablation_a_warmup | 396 | **566** (11,319 × 0.05 = 565.95) |
| S1 warmup | 2,000 | 2,000 (그대로) |

- `config/bc_core_ablation.toml`은 s2·ablation_a·두 warmup만 맞췄다. s1 5,300과 warmup 265는 032-R17 고정값이라 그대로 두었다.
- S2가 크게 늘었다(+43%). 투혼 CP949 제목 수정(A-1)으로 대상 맵 게임이 늘었기 때문이다.

### 8. BC 스모크
| 항목 | 학습 PC CUDA | Linux 호스트 A Vulkan |
|---|---|---|
| 데이터 | `--dataset` 네 출처(manifest 검증 2,045파일) | `--shards` 네 출처(CIFS 마운트 직접) |
| update | 6 | 3 |
| 첫 update(autotune 측정 포함) | 237초 | 124초(시작 검사 스캔 1,017초 별도, 스캔 캐시 없음) |
| 이후 s/update | 약 10초 | 약 101초 |
| loss | 유한값. total 10.0 → 15.9 | 유한값. total 10.3 → 13.0 |
| 기타 | B-2 맵 사전계산 검사 통과(1,212맵), `no_result_games` 4 | master 8757c29(7b0879e 포함)를 그 호스트에서 빌드. autotune 후보 `matmul_naive`가 "Cube count too big"으로 작업 스레드 panic을 출력했지만 학습은 계속됐다(R-2와 같은 현상). `meta.json` autotune·fusion·`grad_clip` 1.0 기록 확인 |

- 체크포인트 `meta.json`에 `autotune: true`, `fusion: true`, `grad_clip: 1.0`이 기록됐다. 코드는 전역 L2 norm clip(041-R3)이다.
- 레이블러 1·2(schema 5·6)가 섞인 2,044파일을 시작 검사 스캔이 모두 읽었다.
- 같은 seed인데도 `--dataset`(학습 PC)와 `--shards`(Linux 호스트 A)는 첫 배치가 달랐다(데이터 지문 `2044:1491…` / `2044:46ae…`).
  - 학습 PC에서 `--shards`로 1 update를 다시 돌리자 지문(`46ae…`)과 첫 배치(무효 474, valid 1,574)가 Linux 호스트 A와 같았다. 스캔 캐시 키도 두 장비가 같다.
  - 그래서 장비 차이가 아니라 데이터 지정 방식 차이다. 장비를 오가며 같은 run을 이어 하려면 모든 호스트가 같은 방식(작업자 큐의 `--dataset`)을 써야 한다.
- 처음 6 update 동안 loss가 오른다. 레인이 같은 게임들의 뒷부분으로 진행하면서 선택 수가 큰 구간이 나오기 때문으로 보인다. 학습률은 warmup 초반이다.
- 출력은 학습 PC 작업 tmp의 임시 `--out`이고, 잠금(heartbeat)은 쓰지 않았다.

### 9. 맵 사전계산 동기화
- `build/maps`의 v2 사전계산 1,244개를 학습 PC `URANIUM_MAPS_DIR`로 복사했다. 이 폴더에는 원래 `.scx`만 있었고, 지운 것은 없다.
- 대상 맵 `.scx` 두 개에 `uranium-replay mapgen`을 돌렸다. 자원은 투혼 122, 서킷브레이커 136이다.
- 각 `.resources.toml`의 CHK 해시로 v2 사전계산(`<chk>.toml`, version 2)을 찾을 수 있다. 라이브 에이전트의 ReplayBuild 경로로 투혼·서킷브레이커 사전계산을 얻는다.

### 10. 고정 바이너리 사본 정리
- 사용 중 확인: 네 장비 모두 uranium-replay·run_queue 프로세스가 없었다. Linux 호스트 B의 다른 세션 build는 10:35에 끝났다.
- 지운 것

  | 장비 | 대상 | 크기 |
  |---|---|---|
  | 학습 PC | 로컬 앱 데이터 `uranium-build/aa187b2`, `uranium-build/d452c29` | 4.6MB, 4.5MB |
  | Linux 호스트 A | `~/uranium-build/aa187b2`, `d452c29`, `build_aa187b2.log`, 저장소 `bridge/openbw/build-aa187b2` | 42MB, 45MB, 4KB, 3MB |
  | Linux 호스트 B | 같음 | 42MB, 41MB, 4KB, 3MB |
  | Linux 호스트 C | git worktree `~/uranium-build/src-aa187b2`·`src-d452c29`(`git worktree remove --force` 뒤 `prune`), `aa187b2`, `d452c29`, build 로그 2개 | 598MB, 597MB, 49MB, 48MB, 24KB |

- Linux 호스트 C 저장소는 깨끗한 master라서 `git pull --ff-only`로 aa187b2에서 8757c29로 올렸다.
- 남긴 것
  - 라이브 브리지 `bridge/openbw/build/`, 각 저장소 `target/`, `build/hashes/`, `quarantine/`, `raw/`, shard, records
  - Linux 호스트 B의 `~/uranium-build/f513b92`: 다른 세션의 래더 증분 build용이다.
- 학습 PC 메인 작업 사본도 7b0879e로 맞췄다(`pull --ff-only`).

### 상태
- 재build 데이터 검증이 끝났다. 코어 비교(032-R17)와 BC 본학습은 시작하지 않았다.
- 040-R19 20% 기준은 그대로다. 출처별 걸러짐은 위 6절에 있다.
- 남은 것
  - Linux 호스트 A·B의 CIFS 쓰기 손실 원인(커널·`wsize`)은 확정하지 못했다. 사용자 결정으로 마운트는 그대로 둔다. 이제는 검증과 체크섬이 손상을 잡는다.
  - 레이블러 1·2 혼합과 마스크 버전 혼합은 받아들인 상태다.
