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
