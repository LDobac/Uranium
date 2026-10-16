# STARDATA 전체 build 전 표본 검증 (2026-10-17)

전체 build(약 36시간)를 다시 하지 않도록, shard 내용(관측·레이블·GameInfo)에 영향을 주는 문제를 표본으로 먼저 찾았다.

## 표본
- STARDATA 원본에서 `--parts 300 --part 0`(인덱스 % 300 == 0)으로 고르게 219개 → written 103 · filtered 119 · desync 2 · unplayable 1. 로컬 jobs 12로 약 7분.
- shard 6개(약 111MB), 103게임, 판단 스텝 232,413개. 매치업 TvP 48 · TvZ 43 · TvT 12, 맵 패밀리 투혼 17 · 서킷브레이커 2 · 기타 84.
- 전체 규모로 환산하면 local jobs 12 기준 약 35시간(기존 추정과 같음).

## 확인한 것
| 항목 | 결과 |
|---|---|
| schema 2 / `GameInfo` (AMB-040-6) | 103게임 모두 일관(맵 크기·지형 높이·시작 위치·self_player = perspective) |
| 레이블 커버리지 (031-AC3) | 98.91% (목표 ≥ 95%) |
| 액션 분포 (034) | 100개 중 90개 등장. 미등장 10개는 드문 기술(Lockdown, Restoration, Optical Flare, 관련 연구·업그레이드)과 `Load_Unit`(태우기는 대부분 우클릭 = `Smart_Unit`) |
| BC 끝까지 실행 (041) | CUDA, 레인 2, 600 업데이트 완주·체크포인트 저장. action loss 2.65 → 약 1.2–1.5, delay 2.74 → 약 2.1 |

## 찾아서 고친 것
| 문제 | 영향 | 수정 | shard 재빌드 필요 |
|---|---|---|---|
| 레이블러가 죽은·탑승한 유닛을 선택·부대지정에서 빼지 않음 (040-R3) | 무효 레이블의 대부분(23,639건). 12칸이 차면 새 유닛 추가가 막혀 레이블 정보 손실 | 명령 프레임의 자기 유닛에 없는 태그를 선택·그룹에서 제거 | **예** (본 build 전에 반영) |
| 애드온 발자국 위치를 비대칭 `dimensions`로 계산 (034-R3) | 머신샵이 한 칸 어긋나 시즈탱크·드랍십 생산 마스크가 100% 닫힘 | `top_left`를 BWAPI 규칙(`tile*32 + tile_size*16`)으로 | 아니요 (feature 코드) |
| BC가 게임 전체 feature를 한 번에 만듦 (041-R2) | 게임당 약 4GB → RAM 부족 | BPTT 청크(T=64) 단위 스트리밍 | 아니요 |

무효 레이블 비율: 17.62% → **6.57%**. 남은 무효:
- `type_mask`(7,154건): 미네랄 부족·인구수 막힘 상태의 연타 등 게임에서도 실패하는 명령. 031-R4대로 weight 0.
- `select_cap`(7,276건): 일부만 명령 자격이 있는 혼합 선택(메딕 섞인 스팀·어택땅, 서플라이 디팟 섞인 랠리 등). 학습 시점 판정 정책이라 shard와 무관.

## 결정 (본 build 전, 2026-10-17 사용자 확정)
- 대상 지정 명령(0x15) 중 034 액션에 매핑되지 않았던 order: `HealMove`(177, 메딕 어택땅) 1,666건, `Guard`(2) 129건, `TowerGuard`/`TowerAttack` 6건.
- `HealMove` 위치 대상은 `Attack_Pos`, 유닛 대상은 `Heal_Unit`으로 변환한다. 나머지는 `Unknown`으로 둔다(040-R11 반영).

## 본 build와 무관한 발견 (M5)
- 기본 BC 설정(레인 32 × T64)은 16GB GPU에서 메모리 부족. 실제 게임은 엔티티가 상한 512에 자주 닿아 엔티티 attention(N²)이 행마다 수 MB를 쓴다. 레인 8도 부족, 레인 2는 동작. f16·배치·체크포인팅 조정이 필요하다(032-AC3 측정은 N=256 기준이었다).
- `uranium-replay build`의 재개 기능은 records 폴더에서 이름이 `<src>`로 시작하는 모든 `.jsonl`을 읽는다. 표본 records는 별도 하위 폴더에 둬야 본 build가 표본 리플레이를 건너뛰지 않는다.

## 도구
- `uranium-replay shard-stats <dir>`: 액션 분포, delay bucket, 인자 사용률, GameInfo 검사.
- `uranium-train label-check --shards <dir>`: 마스크에 걸리는 레이블을 액션·사유별로 집계.
- build records의 `coverage.unknown_orders`: 매핑되지 않은 0x15 order id별 건수.
