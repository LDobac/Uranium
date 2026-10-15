# 061 — Shared Storage (체크포인트·데이터 공용 드라이브)

- **Status**: Approved (2026-10-15, 개정 2026-10-15: SMB 공유 → 리눅스 서버 NFS + Samba 병행)
- **Related**: [060](060-infra-bc250.md), [040](040-replay-dataset.md), [041](041-imitation-learning.md), [042](042-rl-training.md), [032](032-model-architecture.md)

## Purpose
여러 학습 호스트(BC250, CUDA GPU PC)가 체크포인트와 학습 데이터를 로컬 네트워크 공용 드라이브(리눅스 서버, Linux 호스트는 NFS·Windows 호스트는 Samba)로 공유하도록 규칙을 정한다. 학습할 PC는 공용 드라이브에서 체크포인트를 받아(pull) 학습하고, 결과를 다시 올린다(push).

## 확정 사항 (2026-10-15, 사용자 결정)
| 항목 | 결정 |
|---|---|
| 공용 드라이브 | **별도 리눅스 서버/NAS**(학습 PC와 분리, 상시 가동)의 한 디렉터리를 두 프로토콜로 동시 공유한다. Linux 학습 호스트는 **NFS** 마운트, Windows 학습 호스트는 같은 디렉터리를 **Samba(SMB)** 공유로 드라이브 문자나 UNC 경로로 연결한다. (변경 이력: 2026-10-15 'Windows/NAS SMB 공유'에서 변경) |
| 동시 학습 | **학습 실행(run) 단위로 한 PC만** 학습한다. 서로 다른 run은 다른 PC에서 동시에 진행할 수 있다 |
| 학습 데이터 | 원본은 공용 드라이브에 두고, 학습 PC는 필요한 shard를 로컬 캐시로 복사해 쓴다 |
| 학습 호스트 OS | Linux와 Windows 모두 지원 |

## Scope & Non-goals
- 범위:
  - 체크포인트, 리그 스냅샷, 승격 체크포인트의 저장과 배포
  - BC 데이터셋(shard, `.rep`) 저장
  - 학습 실행 잠금
  - 로컬 캐시
- 비범위:
  - 여러 GPU가 하나의 모델을 함께 학습하는 분산 학습
  - 게임 데이터(MPQ·CASC 추출물·맵). 이 파일은 각 게임 PC가 `config/assets.toml` 해시로 검증한다(011-R13).

## Requirements
| ID | 요구사항 |
|---|---|
| 061-R1 | **MUST** 공용 드라이브 루트는 설정값 `URANIUM_SHARED_DIR`로 받는다(예: Linux `/mnt/uranium`, Windows `Z:\uranium` 또는 `\\nas\uranium`). 코드에 경로를 하드코딩하지 않는다. 로컬 캐시 루트는 `URANIUM_CACHE_DIR`이다 |
| 061-R2 | **MUST** 디렉터리 구조는 아래 "구조"를 따른다 |
| 061-R3 | **MUST** 학습 시작 절차: 잠금 획득(R5) → `pull`(최신 체크포인트와 필요한 shard를 로컬 캐시로 복사하고 SHA-256 검증) → 학습 → 주기적 `push`(R4). 체크포인트가 없으면 새 run으로 시작한다 |
| 061-R4 | **MUST** `push`는 원자적으로 한다. 순서는 다음과 같다. 시점은 N분마다(설정, 기본 30분), 승격 시, 정상 종료 시다. |
| 061-R5 | **MUST** 학습 실행 잠금은 **heartbeat 임대 방식**이다. NFS·SMB 파일 잠금(flock/fcntl, oplock)은 쓰지 않는다. 두 프로토콜 사이에 잠금이 공유되지 않기 때문이다. |
| 061-R6 | **MUST** 보존 정책: run마다 최신 체크포인트 5개, 승격된 체크포인트 전부, 리그 스냅샷 전부. 정리는 잠금을 가진 PC만 한다 (기존 060-R4에서 이동) |
| 061-R7 | **MUST** 로컬 캐시는 공용 드라이브 원본의 사본으로만 취급한다. manifest 해시가 다르면 다시 받는다. 아직 push하지 않은 로컬 체크포인트는 "유실 가능"으로 간주하고, 비정상 종료 후 재시작 시 공용 드라이브의 `LATEST`에서 재개한다 |
| 061-R8 | **MUST** 저장 예산(데이터셋 ≤ 600GB, 체크포인트·스냅샷 ≤ 150GB, 060-R7)은 공용 드라이브 기준으로 적용한다. 학습 시작 시 공용 드라이브와 로컬 캐시의 여유 공간을 확인하고, 부족하면 경고한다 |
| 061-R9 | **MUST** 경로 처리는 OS 중립이어야 한다(`std::path`, 대소문자·구분자 차이 허용). 파일 이름에는 SMB·Windows 금지 문자(`<>:"/\|?*`)를 쓰지 않고, 대소문자만 다른 이름을 같은 디렉터리에 두지 않는다(Linux 서버는 대소문자를 구분하고 Windows 클라이언트는 구분하지 않음) |
| 061-R10 | **SHOULD** 체크포인트 형식은 Burn 레코드(가중치) + 옵티마이저 상태 + 학습 상태(step, RNG, 리그 표, 승률 표) + 메타데이터(032-R6, 042-R1)다. 장치(Vulkan/CUDA) 독립 형식으로 저장해 다른 호스트에서 이어서 학습할 수 있어야 한다 |
| 061-R11 | **MUST** 공용 드라이브 서버 설정은 저장소의 `tools/shared-storage/`에 예시 설정(NFS `exports`, Samba `smb.conf` 공유 섹션)과 클라이언트 마운트 안내로 둔다. 아래 "서버 구성" 조건을 만족해야 한다 |
| 061-R12 | **MUST** 잠금 만료 판정(R5의 10분)은 잠금 파일에 기록된 heartbeat 시각으로 하며, 모든 학습 호스트와 서버는 NTP로 시각을 동기화한다. 학습 시작 시 로컬 시각과 공용 드라이브에 새로 쓴 파일의 mtime 차이가 1분을 넘으면 경고한다 |

### 061-R4 push 순서
1. `ckpt-<step>.tmp-<host>/`에 모든 파일과 `manifest.json`(파일별 SHA-256)을 쓴다.
2. 디렉터리를 `ckpt-<step>/`로 rename한다.
3. 마지막에 `LATEST` 파일을 임시 파일 쓰기 → rename 방식으로 갱신한다.

### 061-R5 잠금 규칙
- 잠금 파일 `locks/<run_id>.json`에는 host, pid, 시작 시각, 마지막 heartbeat를 기록한다. heartbeat는 1분마다 갱신한다.
- 유효한 잠금이 있으면 다른 PC는 같은 run의 학습을 거부한다.
- heartbeat가 10분 넘게 끊긴 잠금은 **사용자가 `--take-over`로 명시할 때만** 인수한다.

### 서버 구성 (061-R11)
- 공유 디렉터리 하나를 NFS와 Samba로 동시에 export한다. 두 프로토콜의 파일 소유권이 같은 uid/gid로 매핑되도록 한다(NFS `all_squash`+`anonuid/anongid` 또는 동일 계정, Samba `force user`/`force group`).
- NFS: NFSv4 이상, `sync` export. 클라이언트는 `hard` 마운트를 쓴다(soft 마운트는 쓰기 중 오류로 체크포인트를 손상시킬 수 있다).
- Samba: 다른 프로토콜에서 바뀐 파일을 오래된 캐시로 읽지 않도록 이 공유에서는 oplock/lease를 끈다(`oplocks = no`, `level2 oplocks = no`, `smb2 leases = no`). 대소문자 처리는 기본값(`case sensitive = auto`)을 유지한다.
- `LATEST`와 잠금 파일은 작아서 자주 바뀐다. 클라이언트는 이 두 파일을 읽을 때 캐시된 값을 믿지 않고 매번 새로 연다(NFS close-to-open 일관성에 의존). NFS 속성 캐시 지연(기본 최대 60초)은 잠금 만료 기준(10분)보다 충분히 짧다.

## 구조
```
$URANIUM_SHARED_DIR/
  checkpoints/<run_id>/
    ckpt-<step>/            # weights, optimizer, train_state, config, manifest.json
    LATEST                  # "ckpt-<step>" 한 줄
    promoted/<name>/        # 050 게이트 통과 체크포인트
    league/<snapshot>/      # 042 리그 스냅샷 (T2 상대)
  datasets/
    rep/                    # 원본 리플레이 (040, 내부 전용)
    shards/                 # 040-R12 저장 shard + manifest
  locks/<run_id>.json
  reports/                  # 평가·학습 리포트 사본
$URANIUM_CACHE_DIR/         # 학습 PC 로컬 캐시 (shards, 현재 run 체크포인트)
```

## Acceptance Criteria
- 061-AC1: PC A에서 pull → 학습 → push → PC B에서 pull → resume을 했을 때, 가중치·옵티마이저·RNG 상태가 비트 단위로 일치하고, 이어서 진행한 학습 손실 곡선이 끊김 없이 이어진다.
- 061-AC2: push 도중 강제 종료해도 `LATEST`는 항상 manifest가 검증되는 완전한 체크포인트를 가리킨다(장애 주입 테스트).
- 061-AC3: 유효한 잠금이 있는 run을 두 번째 PC에서 시작하면 명확한 에러로 거부된다. 만료된 잠금은 `--take-over` 없이는 인수되지 않는다.
- 061-AC4: Linux(NFS 마운트)와 Windows(Samba 드라이브 문자) 경로 모두에서 AC1–AC3이 통과한다.
- 061-AC6: 교차 프로토콜 — NFS 클라이언트(Linux)가 push한 체크포인트를 Samba 클라이언트(Windows)가 pull해 manifest 검증에 통과하고, 반대 방향도 같다. 한쪽이 잡은 잠금을 다른 쪽이 유효한 잠금으로 인식해 거부한다.
- 061-AC5: Vulkan 호스트에서 저장한 체크포인트를 CUDA 호스트에서 불러왔을 때, 같은 입력의 forward 출력 차이가 ≤ 1e-4(f32)다. 반대 방향도 같다.

## Risks & Open Questions
- rename 원자성: 리눅스 서버 로컬 파일시스템에서 같은 디렉터리 안 rename은 원자적이고, NFS·Samba 모두 서버 rename 한 번으로 처리한다. 다만 Windows 클라이언트에서 대상이 이미 있을 때의 덮어쓰기 rename(`LATEST` 갱신) 동작은 M5에서 실제 공용 드라이브로 061-AC2·AC6을 검증한다.
- 교차 프로토콜 캐시: Samba oplock을 끄지 않으면 Windows 클라이언트가 Linux 쪽에서 갱신된 `LATEST`·잠금 파일을 늦게 볼 수 있다. 서버 구성(061-R11)으로 막고 AC6으로 확인한다.
- 시각 차이: heartbeat 만료 판정은 호스트 시각에 의존한다. NTP 미동기화 시 잘못된 만료 판정이 날 수 있어 061-R12로 경고한다(인수는 어차피 `--take-over`가 필요하다).
- 네트워크 대역폭에 따라 shard pull 시간이 길 수 있다. 로컬 캐시를 재사용해 완화한다.
- 공용 드라이브 장애는 학습 중단으로 이어진다. 로컬 캐시로 학습을 계속하고, 드라이브가 복구되면 push를 재시도한다.
