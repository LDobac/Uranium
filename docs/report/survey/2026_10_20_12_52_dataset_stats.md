# 데이터셋 통계 (040-R6, 040-AC2, 040-AC4)

`uranium-replay dataset-stats`로 build 기록(records 77개 파일)에서 생성. 개수만 기록한다(040-R14). **게임** = 원본 리플레이(`replay_sha256`) 고유 개수, **시점** = shard에 쓴 테란 시점(TvT는 게임당 2, 040-R2), **스텝** = 판단 스텝 수.

생성 명령(2026-10-20, master 7b0879e, R-5 수리 뒤): `uranium-replay dataset-stats <URANIUM_SHARED_DIR>/datasets/replays/records --out reports/dataset_stats.md`. 다시 만들면 이 줄은 지워지므로 손으로 다시 넣는다.

- 저장: 게임 33880 / 시점 36445 / 스텝 86368898
- 레이블 커버리지(031-AC3, 목표 ≥ 95%): 게임 영향 명령 69625685개 중 69528678개, 99.86%
- 중복 기록(같은 리플레이·시점, 마지막 것만 셈): 29개 제외
- 기록 승자(040-R8, 기록 단위): 이름 일치 1172, 불일치 0

## 레이블러 버전 (040 "Shard 포맷")

버전 1(shard schema 5)은 선택을 `max_select`로 자르고 미분류 명령 시간을 NoOp으로 채웠다. 버전 2(schema 6)는 선택 전체·`unknown` 레이블(가중치 0)·`Unload_Unit`. 기록에 버전이 없으면 1이다.

| 레이블러 | 게임 | 시점 | 스텝 | 게임 영향 명령 | 레이블 | unknown |
|---|---|---|---|---|---|---|
| 1 | 13559 | 14591 | 34249944 | 27660383 | 27599008 | 61375 |
| 2 | 20322 | 21854 | 52118954 | 41965302 | 41929670 | 35632 |

## 맵 패밀리 × 매치업 (040-AC4)

| 맵 패밀리 | 매치업 | 게임 | 시점 | 스텝 |
|---|---|---|---|---|
| circuit_breaker | TvP | 389 | 389 | 959623 |
| circuit_breaker | TvT | 56 | 112 | 289550 |
| circuit_breaker | TvZ | 280 | 280 | 651704 |
| fighting_spirit | TvP | 5415 | 5415 | 13061944 |
| fighting_spirit | TvT | 751 | 1486 | 3949802 |
| fighting_spirit | TvZ | 4310 | 4310 | 9589396 |
| other | TvP | 11501 | 11501 | 28299741 |
| other | TvT | 1822 | 3596 | 9041278 |
| other | TvZ | 9356 | 9356 | 20525860 |

## 맵 패밀리 분포 (040-R15)

| 맵 패밀리 | 게임 | 시점 | 스텝 |
|---|---|---|---|
| circuit_breaker | 725 | 781 | 1900877 |
| fighting_spirit | 10476 | 11211 | 26601142 |
| other | 22679 | 24453 | 57866879 |

## 맵 제목별 (040-R6, 게임 / 시점)

| 맵 제목 | 맵 패밀리 | TvT | TvP | TvZ | 합계 |
|---|---|---|---|---|---|
| \| iCCup \| Fighting Spirit  | fighting_spirit | 514 / 1017 | 3803 / 3803 | 3133 / 3133 | 7450 / 7953 |
| \| iCCup \| Python 1.3 | other | 191 / 378 | 1705 / 1705 | 1195 / 1195 | 3091 / 3278 |
| 투혼 1.3 | fighting_spirit | 180 / 359 | 1436 / 1436 | 1061 / 1061 | 2677 / 2856 |
| \| iCCup \| Heartbreak Ridge | other | 120 / 239 | 1031 / 1031 | 1188 / 1188 | 2339 / 2458 |
| \| iCCup \| Destination 1.1 | other | 130 / 259 | 906 / 906 | 1032 / 1032 | 2068 / 2197 |
| Python 1.3 | other | 36 / 72 | 342 / 342 | 245 / 245 | 623 / 659 |
| \| iCCup \| Othello 1.1 | other | 45 / 89 | 238 / 238 | 304 / 304 | 587 / 631 |
| \| iCCup \| Circuit Breakers | circuit_breaker | 38 / 76 | 318 / 318 | 224 / 224 | 580 / 618 |
| \| iCCup \| Tau Cross | other | 44 / 88 | 301 / 301 | 223 / 223 | 568 / 612 |
| \| iCCup \| Andromeda 1.0 | other | 13 / 26 | 342 / 342 | 108 / 108 | 463 / 476 |
| \| iCCup \| ColosseumII 2.0 | other | 21 / 42 | 192 / 192 | 200 / 200 | 413 / 434 |
| \| iCCup \| Destination 1.1  | other | 25 / 50 | 189 / 189 | 192 / 192 | 406 / 431 |
| \| iCCup \| Longinus | other | 23 / 46 | 189 / 189 | 171 / 171 | 383 / 406 |
| \| iCCup \| Blue Storm 1.2 | other | 23 / 45 | 191 / 191 | 166 / 166 | 380 / 402 |
| KnockOut 1.4 | other | 60 / 115 | 109 / 109 | 92 / 92 | 261 / 316 |
| Octagon SE 2.0 | other | 53 / 102 | 110 / 110 | 99 / 99 | 262 / 311 |
| Destination 1.1 | other | 19 / 38 | 169 / 169 | 78 / 78 | 266 / 285 |
| Backrooms 1.1 | other | 54 / 101 | 87 / 87 | 96 / 96 | 237 / 284 |
| \| iCCup \| NeoMedusa 2.1 | other | 21 / 41 | 143 / 143 | 98 / 98 | 262 / 282 |
| \| iCCup \| Aztec 2.1 | other | 15 / 30 | 124 / 124 | 124 / 124 | 263 / 278 |
| \| iCCup \| Icarus | other | 20 / 37 | 117 / 117 | 122 / 122 | 259 / 276 |
| \| iCCup \| MoonGlaive 1.2 | other | 50 / 98 | 97 / 97 | 70 / 70 | 217 / 265 |
| \| iCCup \| Python 1.3 OBs | other | 17 / 34 | 121 / 121 | 95 / 95 | 233 / 250 |
| \| iCCup \| Eye of the Storm | other | 18 / 36 | 90 / 90 | 121 / 121 | 229 / 247 |
| \| iCCup \| Empire of the Su | other | 11 / 22 | 115 / 115 | 99 / 99 | 225 / 236 |
| \| iCCup \| Luna the Final | other | 18 / 36 | 103 / 103 | 86 / 86 | 207 / 225 |
| MatchPoint 1.3 | other | 18 / 36 | 102 / 102 | 85 / 85 | 205 / 223 |
| \| iCCup \| Match Point 1.3 | other | 16 / 32 | 107 / 107 | 75 / 75 | 198 / 214 |
| 투혼 1.4 | fighting_spirit | 41 / 79 | 73 / 73 | 56 / 56 | 170 / 208 |
| \| iCCup \| Lost Temple 2.4 | other | 18 / 36 | 90 / 90 | 81 / 81 | 189 / 207 |
| \| iCCup \| The Fortress 2.0 | other | 10 / 20 | 86 / 86 | 98 / 98 | 194 / 204 |
| \| iCCup \| Outsider 2.2 | other | 21 / 42 | 65 / 65 | 95 / 95 | 181 / 202 |
| \| iCCup \| Gaia | other | 15 / 30 | 83 / 83 | 78 / 78 | 176 / 191 |
| \| iCCup \| Grand Line SE 2. | other | 15 / 30 | 89 / 89 | 56 / 56 | 160 / 175 |
| \| iCCup \| Sniper Ridge 2.0 | other | 18 / 36 | 71 / 71 | 61 / 61 | 150 / 168 |
| \| iCCup \| Bloody Ridge 2.1 | other | 11 / 22 | 65 / 65 | 75 / 75 | 151 / 162 |
| \| iCCup \| MoonGlaive 2.1 | other | 17 / 34 | 70 / 70 | 50 / 50 | 137 / 154 |
| Circuit Breakers 1.0 | circuit_breaker | 17 / 34 | 63 / 63 | 48 / 48 | 128 / 145 |
| \| iCCup \| Jade 2.0 | other | 12 / 24 | 68 / 68 | 53 / 53 | 133 / 145 |
| \| iCCup \| Benzene 1.1 | other | 9 / 17 | 49 / 49 | 75 / 75 | 133 / 141 |
| Luna | other | 10 / 20 | 108 / 108 | 12 / 12 | 130 / 140 |
| \| iCCup \| Electric Circuit | other | 5 / 10 | 56 / 56 | 63 / 63 | 124 / 129 |
| \| iCCup \| Match Point 1.2 | other | 5 / 10 | 53 / 53 | 61 / 61 | 119 / 124 |
| \| iCCup \| Ground Zero 2.0 | other | 8 / 16 | 54 / 54 | 45 / 45 | 107 / 115 |
| \| iCCup \| Ground Zero 1.2 | other | 5 / 10 | 59 / 59 | 38 / 38 | 102 / 107 |
| \| iCCup \| Bloody Ridge | other | 6 / 12 | 38 / 38 | 49 / 49 | 93 / 99 |
| \| iCCup \| Dantes Peak SE 2 | other | 6 / 12 | 49 / 49 | 37 / 37 | 92 / 98 |
| Tau Cross 1.1 | other | 8 / 15 | 60 / 60 | 18 / 18 | 86 / 93 |
| \| iCCup \| Gods Garden | other | 10 / 20 | 42 / 42 | 30 / 30 | 82 / 92 |
| \| iCCup \| La Mancha 1.1 | other | 6 / 12 | 35 / 35 | 37 / 37 | 78 / 84 |
| \| iCCup \| Aztec 1.1 | other | 6 / 12 | 33 / 33 | 38 / 38 | 77 / 83 |
| 신 단장의 능선 2.1 | other | 5 / 10 | 44 / 44 | 27 / 27 | 76 / 81 |
| \| iCCup \| ColosseumII 2.0  | other | 10 / 20 | 31 / 31 | 29 / 29 | 70 / 80 |
| \| iCCup \| Sniper Ridge 1.0 | other | 10 / 20 | 30 / 30 | 30 / 30 | 70 / 80 |
| Blue Storm 1.1 | other | 10 / 20 | 50 / 50 | 9 / 9 | 69 / 79 |
| \| iCCup \| Gladiator | other | 4 / 8 | 38 / 38 | 33 / 33 | 75 / 79 |
| \| iCCup \| Dante's Peak SE  | other | 2 / 4 | 45 / 45 | 26 / 26 | 73 / 75 |
| \| iCCup \| Ride of Valkyrie | other | 8 / 16 | 34 / 34 | 24 / 24 | 66 / 74 |
| Aiolos 1.0b | other | 18 / 32 | 24 / 24 | 17 / 17 | 59 / 73 |
| Blue Storm 1.2 | other | 7 / 14 | 42 / 42 | 14 / 14 | 63 / 70 |
| \| iCCup \| Alternative 1.3 | other | 7 / 14 | 32 / 32 | 24 / 24 | 63 / 70 |
| \| iCCup \| Wind and Cloud 0 | other | 10 / 20 | 31 / 31 | 19 / 19 | 60 / 70 |
| The Lost Temple_Gamei | other | 5 / 9 | 46 / 46 | 13 / 13 | 64 / 68 |
| \| iCCup \| Hunters_KeSPA 1. | other | 7 / 14 | 30 / 30 | 24 / 24 | 61 / 68 |
| \| iCCup \| Beltway 2.0 | other | 6 / 12 | 30 / 30 | 25 / 25 | 61 / 67 |
| \| iCCup \| Sniper Ridge 1.1 | other | 3 / 6 | 34 / 34 | 26 / 26 | 63 / 66 |
| \| iCCup \| Tau Cross 1.0 | other | 5 / 10 | 28 / 28 | 27 / 27 | 60 / 65 |
| 단장의 능선 1.1 | other | 6 / 11 | 30 / 30 | 23 / 23 | 59 / 64 |
| Odyssey:RE 2.0 | other | 12 / 22 | 26 / 26 | 15 / 15 | 53 / 63 |
| [CPL] Fighting Spirit 1.3  | fighting_spirit | 4 / 8 | 32 / 32 | 22 / 22 | 58 / 62 |
| \| iCCup \| Blue Storm 1.2 O | other | 1 / 2 | 28 / 28 | 32 / 32 | 61 / 62 |
| \| iCCup \| Polaris Rhapsody | other | 6 / 12 | 27 / 27 | 21 / 21 | 54 / 60 |
| \| iCCup \| Chain Reaction 2 | other | 1 / 2 | 34 / 34 | 22 / 22 | 57 / 58 |
| \| iCCup \| Pathfinder | other | 3 / 6 | 28 / 28 | 23 / 23 | 54 / 57 |
| \| iCCup \| Longinus 1.0 | other | 7 / 14 | 20 / 20 | 21 / 21 | 48 / 55 |
| \| iCCup \| Tau Cross OBs | other | 3 / 6 | 22 / 22 | 27 / 27 | 52 / 55 |
| \| iCCup \| Tornado | other | 2 / 4 | 25 / 25 | 22 / 22 | 49 / 51 |
| [gm] 투혼 1.3 | fighting_spirit | 3 / 6 | 30 / 30 | 13 / 13 | 46 / 49 |
| \| iCCup \| Gemlong 1.0 | other | 2 / 4 | 30 / 30 | 15 / 15 | 47 / 49 |
| LonginusⅡ | other | 4 / 8 | 34 / 34 | 6 / 6 | 44 / 48 |
| \| iCCup \| Andromeda 1.0 OB | other | 5 / 10 | 23 / 23 | 15 / 15 | 43 / 48 |
| \| iCCup \| Nostalgia 1.3 | other | 3 / 6 | 25 / 25 | 17 / 17 | 45 / 48 |
| Andromeda 1.0 | other | 6 / 12 | 23 / 23 | 11 / 11 | 40 / 46 |
| 신 백두대간 2.4 | other | 7 / 14 | 23 / 23 | 9 / 9 | 39 / 46 |
| 투혼 1.1 | fighting_spirit | 5 / 10 | 23 / 23 | 13 / 13 | 41 / 46 |
| \| iCCup \| Rush Hour 3.1 | other | 4 / 8 | 15 / 15 | 21 / 21 | 40 / 44 |
| Othello 1.1 | other | 10 / 20 | 16 / 16 | 6 / 6 | 32 / 42 |
| \| iCCup \| Jade 1.0 | other | 2 / 4 | 22 / 22 | 16 / 16 | 40 / 42 |
| MatchPoint 1.2 | other | 1 / 2 | 17 / 17 | 22 / 22 | 40 / 41 |
| Neo Jade 2.0 | other | 2 / 4 | 22 / 22 | 15 / 15 | 39 / 41 |
| \| iCCup \| Tornado 2.1 | other | 3 / 6 | 19 / 19 | 16 / 16 | 38 / 41 |
| La Mancha 1.1 | other | 5 / 10 | 19 / 19 | 11 / 11 | 35 / 40 |
| Python 1.1 | other | 5 / 10 | 18 / 18 | 12 / 12 | 35 / 40 |
| \| iCCup \| Jade 1.1 | other | 3 / 6 | 21 / 21 | 13 / 13 | 37 / 40 |
| Gorky Island 1.0 | other | 9 / 18 | 11 / 11 | 10 / 10 | 30 / 39 |
| \| iCCup \| Dantes Peak | other | 3 / 6 | 6 / 6 | 26 / 26 | 35 / 38 |
| \| iCCup \| Outlier 2.2 | other | 4 / 8 | 11 / 11 | 19 / 19 | 34 / 38 |
| \| iCCup \| Outsider 2.0 | other | 1 / 2 | 12 / 12 | 24 / 24 | 37 / 38 |
| \| iCCup \| Match Point 1.1 | other | 4 / 8 | 14 / 14 | 15 / 15 | 33 / 37 |
| \| iCCup \| Outsider | other | 5 / 10 | 14 / 14 | 12 / 12 | 31 / 36 |
| Big Game Hunters | other | 1 / 2 | 28 / 28 | 5 / 5 | 34 / 35 |
| \| iCCup \| Monte Cristo 1.0 | other | 1 / 2 | 19 / 19 | 14 / 14 | 34 / 35 |
| \| iCCup \| Judgement Day 1. | other | 2 / 4 | 13 / 13 | 17 / 17 | 32 / 34 |
| \| iCCup \| Othello 1.1 OBs | other | 2 / 4 | 18 / 18 | 12 / 12 | 32 / 34 |
| \| iCCup \| Sin Chupung Ryeo | other | 2 / 4 | 8 / 8 | 22 / 22 | 32 / 34 |
| \| iCCup \| Tornado 2.2 | other | 4 / 8 | 13 / 13 | 13 / 13 | 30 / 34 |
| GrandLine_SE 2.0 | other | 3 / 6 | 6 / 6 | 20 / 20 | 29 / 32 |
| Rush Hour Ⅲ | other | 1 / 2 | 27 / 27 | 2 / 2 | 30 / 31 |
| \| iCCup \| Chain Reaction 1 | other | 3 / 6 | 14 / 14 | 10 / 10 | 27 / 30 |
| WCG Estrella 1.0 | other | 6 / 12 | 11 / 11 | 6 / 6 | 23 / 29 |
| 신 저격능선 2.0 | other | 2 / 4 | 13 / 13 | 12 / 12 | 27 / 29 |
| Katrina1.3 | other | 3 / 6 | 18 / 18 | 4 / 4 | 25 / 28 |
| Outsider 1.0 | other | 2 / 4 | 12 / 12 | 12 / 12 | 26 / 28 |
| Paradoxxx ll 2.0 | other | 5 / 10 | 10 / 10 | 8 / 8 | 23 / 28 |
| 피의 능선 1.0 | other | 1 / 2 | 16 / 16 | 9 / 9 | 26 / 27 |
| \| iCCup \| Blue Storm 1.1 | other | 2 / 4 | 19 / 19 | 3 / 3 | 24 / 26 |
| \| iCCup \| Medusa 2.1 | other | 3 / 6 | 9 / 9 | 11 / 11 | 23 / 26 |
| ColosseumII2.0 | other | 3 / 6 | 11 / 11 | 8 / 8 | 22 / 25 |
| \| iCCup \| Colosseum II 2.0 | other | 4 / 8 | 10 / 10 | 7 / 7 | 21 / 25 |
| \| iCCup \| Faoi 2.1 | other | 2 / 3 | 9 / 9 | 13 / 13 | 24 / 25 |
| \| iCCup \| Heatbreak Ridge | other | 3 / 6 | 11 / 11 | 8 / 8 | 22 / 25 |
| \| iCCup \| Longinus OBs | other | 3 / 6 | 11 / 11 | 8 / 8 | 22 / 25 |
| \| iCCup \| NeoMedusa 2.1 OB | other | 1 / 2 | 13 / 13 | 10 / 10 | 24 / 25 |
| \| iCCup \| Triathlon 1.1 | other | 2 / 4 | 14 / 14 | 7 / 7 | 23 / 25 |
| \| iCCup \| Desertec 1.0 | other | 2 / 4 | 13 / 13 | 7 / 7 | 22 / 24 |
| \| iCCup \| Gaia 1.0 | other | 2 / 4 | 13 / 13 | 7 / 7 | 22 / 24 |
| Blood Bath | other | 0 / 0 | 8 / 8 | 15 / 15 | 23 / 23 |
| Jade 1.0 | other | 0 / 0 | 19 / 19 | 4 / 4 | 23 / 23 |
| \| iCCup \| Resonance II 1.0 | other | 4 / 7 | 8 / 8 | 8 / 8 | 20 / 23 |
| Benzene1.1 | other | 1 / 2 | 8 / 8 | 12 / 12 | 21 / 22 |
| Neo Ground Zero 2.0 | other | 1 / 2 | 8 / 8 | 12 / 12 | 21 / 22 |
| R - Point | other | 0 / 0 | 19 / 19 | 3 / 3 | 22 / 22 |
| \| iCCup \| Heatbreak Ridge  | other | 0 / 0 | 9 / 9 | 13 / 13 | 22 / 22 |
| \| iCCup \| Lost Temple 2.4  | other | 1 / 2 | 5 / 5 | 15 / 15 | 21 / 22 |
| \| iCCup \| Luna the Final 1 | other | 1 / 2 | 12 / 12 | 8 / 8 | 21 / 22 |
| Neo Forbidden Zone | other | 3 / 6 | 11 / 11 | 3 / 3 | 17 / 20 |
| Neo The Lost Temple | other | 2 / 4 | 13 / 13 | 3 / 3 | 18 / 20 |
| \| iCCup \| Medusa 1.1 | other | 2 / 4 | 8 / 8 | 8 / 8 | 18 / 20 |
| \| iCCup \| Othello 1.1 OBS | other | 0 / 0 | 9 / 9 | 11 / 11 | 20 / 20 |
| \| iCCup \| Outlier 1.0 | other | 2 / 4 | 8 / 8 | 8 / 8 | 18 / 20 |
| Medusa1.0 | other | 3 / 6 | 12 / 12 | 1 / 1 | 16 / 19 |
| PolarisRhapsody 1.0 | other | 2 / 4 | 9 / 9 | 6 / 6 | 17 / 19 |
| Rush Hour Ⅱ | other | 3 / 6 | 11 / 11 | 2 / 2 | 16 / 19 |
| STL_Destination 1.1 | other | 1 / 2 | 10 / 10 | 7 / 7 | 18 / 19 |
| WCG Paranoid Android  | other | 3 / 6 | 8 / 8 | 5 / 5 | 16 / 19 |
| Aztec1.1 | other | 2 / 4 | 7 / 7 | 7 / 7 | 16 / 18 |
| Electric Circ | other | 1 / 2 | 10 / 10 | 6 / 6 | 17 / 18 |
| \| iCCup \| Outsider OBs | other | 1 / 2 | 4 / 4 | 12 / 12 | 17 / 18 |
| Dante's Peak SE 2.0 | other | 1 / 2 | 5 / 5 | 10 / 10 | 16 / 17 |
| Icarus 1.1 | other | 2 / 4 | 4 / 4 | 9 / 9 | 15 / 17 |
| NeoMoonGlaive2.1 | other | 1 / 2 | 11 / 11 | 4 / 4 | 16 / 17 |
| Outsider SE 2.2 | other | 2 / 4 | 9 / 9 | 4 / 4 | 15 / 17 |
| 폭풍의 언덕 1.0 | other | 5 / 10 | 7 / 7 | 0 / 0 | 12 / 17 |
| FS [MapHack] | other | 4 / 8 | 6 / 6 | 2 / 2 | 12 / 16 |
| Neo Bifrost | other | 3 / 6 | 9 / 9 | 1 / 1 | 13 / 16 |
| Ride of Valkyries 1.0 | other | 3 / 6 | 7 / 7 | 3 / 3 | 13 / 16 |
| \| iCCup \| Luna the Final O | other | 1 / 2 | 6 / 6 | 8 / 8 | 15 / 16 |
| Arcadia II | other | 1 / 2 | 11 / 11 | 2 / 2 | 14 / 15 |
| Colosseum1.0 | other | 2 / 4 | 10 / 10 | 1 / 1 | 13 / 15 |
| Luna the Final | other | 2 / 4 | 11 / 11 | 0 / 0 | 13 / 15 |
| Neo Forte 2.1 | other | 1 / 2 | 12 / 12 | 1 / 1 | 14 / 15 |
| PGT - Luna The Final [07] | other | 1 / 2 | 13 / 13 | 0 / 0 | 14 / 15 |
| \| iCCup \| Rush Hour 3.1 OB | other | 1 / 2 | 8 / 8 | 5 / 5 | 14 / 15 |
| \| iCCup \| Troy 1.0 | other | 1 / 2 | 6 / 6 | 7 / 7 | 14 / 15 |
| Neo Hall of Valhalla 2.0 | other | 3 / 6 | 3 / 3 | 5 / 5 | 11 / 14 |
| 단장의 능선 1.2 | other | 0 / 0 | 6 / 6 | 8 / 8 | 14 / 14 |
| Blue Storm 1.16 | other | 2 / 4 | 8 / 8 | 1 / 1 | 11 / 13 |
| NeoMedusa2.1 | other | 2 / 4 | 3 / 3 | 6 / 6 | 11 / 13 |
| PLU_Luna The Final | other | 1 / 2 | 10 / 10 | 1 / 1 | 12 / 13 |
| STM_LunaTheFinal 2.1 | other | 1 / 2 | 8 / 8 | 3 / 3 | 12 / 13 |
| The Lost Temple | other | 3 / 6 | 5 / 5 | 2 / 2 | 10 / 13 |
| WCG Gaia 1.1 | other | 1 / 2 | 7 / 7 | 4 / 4 | 12 / 13 |
| \| iCCup \| God's Garden 1.0 | other | 2 / 4 | 8 / 8 | 1 / 1 | 11 / 13 |
| \| iCCup \| Medusa 1.1 OBs | other | 1 / 2 | 7 / 7 | 4 / 4 | 12 / 13 |
| \| iCCup \| Moon Glaive 2.1 | other | 1 / 2 | 10 / 10 | 1 / 1 | 12 / 13 |
| \| iCCup \| The Fortress 1.1 | other | 0 / 0 | 5 / 5 | 8 / 8 | 13 / 13 |
| \| iCCup \| ZodiAc | other | 1 / 2 | 9 / 9 | 2 / 2 | 12 / 13 |
| 신 단장의 능선 2.0 | other | 0 / 0 | 8 / 8 | 5 / 5 | 13 / 13 |
| Icarus 1.0 | other | 3 / 5 | 3 / 3 | 4 / 4 | 10 / 12 |
| MoonGlaive1.2 | other | 2 / 4 | 5 / 5 | 3 / 3 | 10 / 12 |
| Reverse Temple 1.0 | other | 1 / 2 | 7 / 7 | 3 / 3 | 11 / 12 |
| \| iCCup \| Odd Eye 3.0 | other | 2 / 4 | 6 / 6 | 2 / 2 | 10 / 12 |
| \| iCCup \| Outsider 2.0 OBs | other | 0 / 0 | 5 / 5 | 7 / 7 | 12 / 12 |
| 단장의 능선 1.0 | other | 1 / 2 | 6 / 6 | 4 / 4 | 11 / 12 |
| 신 용오름 2.2 | other | 1 / 2 | 4 / 4 | 6 / 6 | 11 / 12 |
| 신 피의 능선 2.1 | other | 0 / 0 | 8 / 8 | 4 / 4 | 12 / 12 |
| 저격능선 1.0 | other | 0 / 0 | 6 / 6 | 6 / 6 | 12 / 12 |
| Gaema Gowon | other | 1 / 2 | 9 / 9 | 0 / 0 | 10 / 11 |
| Pathfinder 1.0 | other | 1 / 2 | 4 / 4 | 5 / 5 | 10 / 11 |
| The Fortress SE 2.0 | other | 0 / 0 | 3 / 3 | 8 / 8 | 11 / 11 |
| \| iCCup \| Python 1.1 | other | 2 / 4 | 6 / 6 | 1 / 1 | 9 / 11 |
| Blitz X | other | 2 / 4 | 6 / 6 | 0 / 0 | 8 / 10 |
| GrandLine_SE 2.2 | other | 2 / 4 | 1 / 1 | 5 / 5 | 8 / 10 |
| NeoMedusa2.2 | other | 2 / 4 | 2 / 2 | 4 / 4 | 8 / 10 |
| Outsider SE 2.0 | other | 0 / 0 | 6 / 6 | 4 / 4 | 10 / 10 |
| WGT13 - Luna the Final (Ob | other | 3 / 6 | 3 / 3 | 1 / 1 | 7 / 10 |
| \| iCCup \| NeoMedusa2.0 OBs | other | 0 / 0 | 7 / 7 | 3 / 3 | 10 / 10 |
| \| iCCup \| PeaksOfBeakdu | other | 0 / 0 | 4 / 4 | 6 / 6 | 10 / 10 |
| �Fa�te�t Po��ible Map Eve | other | 0 / 0 | 10 / 10 | 0 / 0 | 10 / 10 |
| Gorky Island | other | 3 / 6 | 3 / 3 | 0 / 0 | 6 / 9 |
| Jade 1.1 | other | 0 / 0 | 3 / 3 | 6 / 6 | 9 / 9 |
| Medusa1.1 | other | 0 / 0 | 8 / 8 | 1 / 1 | 9 / 9 |
| Monty Hall_SE 2.1 | other | 1 / 2 | 5 / 5 | 2 / 2 | 8 / 9 |
| NeoMedusa2.0 | other | 0 / 0 | 5 / 5 | 4 / 4 | 9 / 9 |
| Plains to Hill | other | 3 / 5 | 3 / 3 | 1 / 1 | 7 / 9 |
| Sin Gaema Gowon | other | 1 / 2 | 6 / 6 | 1 / 1 | 8 / 9 |
| WCG Azalea 1.0 | other | 1 / 2 | 5 / 5 | 2 / 2 | 8 / 9 |
| WCG_The Lost Temple | other | 0 / 0 | 9 / 9 | 0 / 0 | 9 / 9 |
| \| ICCup \| Troy 1.0 | other | 0 / 0 | 9 / 9 | 0 / 0 | 9 / 9 |
| \| iCCup \| Longinus 2.0 | other | 2 / 4 | 5 / 5 | 0 / 0 | 7 / 9 |
| \| iCCup \| Neo Jade 2.1 | other | 0 / 0 | 7 / 7 | 2 / 2 | 9 / 9 |
| DAHLIA | other | 0 / 0 | 6 / 6 | 2 / 2 | 8 / 8 |
| Defiler Perversion 3 | other | 1 / 2 | 3 / 3 | 3 / 3 | 7 / 8 |
| Neo Aztec2.0 | other | 0 / 0 | 6 / 6 | 2 / 2 | 8 / 8 |
| Neo Aztec2.1 | other | 1 / 2 | 2 / 2 | 4 / 4 | 7 / 8 |
| Neo Beltway 2.0 | other | 1 / 2 | 1 / 1 | 5 / 5 | 7 / 8 |
| Neo Electric Circuit  | other | 0 / 0 | 4 / 4 | 4 / 4 | 8 / 8 |
| PGT - Gaia 1.0 [10] | other | 2 / 4 | 4 / 4 | 0 / 0 | 6 / 8 |
| STL_NeoMedusa 2.2 | other | 2 / 4 | 4 / 4 | 0 / 0 | 6 / 8 |
| WCG_Neo Hall of Valhalla | other | 0 / 0 | 8 / 8 | 0 / 0 | 8 / 8 |
| WGT13 - Tau Cross 1.1 | other | 0 / 0 | 5 / 5 | 3 / 3 | 8 / 8 |
| WGTour (7) Lost Temple 2.3 | other | 1 / 2 | 6 / 6 | 0 / 0 | 7 / 8 |
| [397] 투혼 1.3 | fighting_spirit | 1 / 2 | 5 / 5 | 1 / 1 | 7 / 8 |
| [CPL] Destination 1.1 OBS | other | 0 / 0 | 6 / 6 | 2 / 2 | 8 / 8 |
| [CPL] Match Point 1.2 OBS | other | 1 / 2 | 4 / 4 | 2 / 2 | 7 / 8 |
| \| iCCup \| Longinus 2.0 OB | other | 2 / 4 | 1 / 1 | 3 / 3 | 6 / 8 |
| \| iCCup \| Match Point 1.1  | other | 1 / 2 | 3 / 3 | 3 / 3 | 7 / 8 |
| \| iCCup \| Toad Stone 1.3 | other | 0 / 0 | 5 / 5 | 3 / 3 | 8 / 8 |
| \| svn \| Python 1.3 | other | 2 / 4 | 4 / 4 | 0 / 0 | 6 / 8 |
| 신 용오름 2.1 | other | 1 / 2 | 4 / 4 | 2 / 2 | 7 / 8 |
| 신 청풍명월 2.1 | other | 1 / 2 | 3 / 3 | 3 / 3 | 7 / 8 |
| 신 태양의 제국 2.0 | other | 0 / 0 | 4 / 4 | 4 / 4 | 8 / 8 |
| Bifrost lll | other | 0 / 0 | 2 / 2 | 5 / 5 | 7 / 7 |
| Ground Zero 1.2 | other | 0 / 0 | 3 / 3 | 4 / 4 | 7 / 7 |
| Nostalgia 1.3 | other | 0 / 0 | 5 / 5 | 2 / 2 | 7 / 7 |
| PGT - Neo Forte 2.1 [06] | other | 0 / 0 | 7 / 7 | 0 / 0 | 7 / 7 |
| STL_Python 1.3 | other | 0 / 0 | 6 / 6 | 1 / 1 | 7 / 7 |
| WGT13 - Luna The Final (Ob | other | 0 / 0 | 7 / 7 | 0 / 0 | 7 / 7 |
| WGT13 - Ride of Valkyries  | other | 2 / 4 | 2 / 2 | 1 / 1 | 5 / 7 |
| [CPL] Heartbreak Ridge 2.1 | other | 0 / 0 | 2 / 2 | 5 / 5 | 7 / 7 |
| \| iCCup \| Chupung-Ryeung 1 | other | 0 / 0 | 4 / 4 | 3 / 3 | 7 / 7 |
| \| iCCup \| Circuit Breaker  | circuit_breaker | 0 / 0 | 3 / 3 | 4 / 4 | 7 / 7 |
| \| iCCup \| Fortress SE 2.0 | other | 0 / 0 | 5 / 5 | 2 / 2 | 7 / 7 |
| \| iCCup \| Jade 2.1 | other | 0 / 0 | 3 / 3 | 4 / 4 | 7 / 7 |
| \| iCCup \| Loki II | other | 0 / 0 | 6 / 6 | 1 / 1 | 7 / 7 |
| \| iCCup \| Pathfinder 1.0 | other | 1 / 2 | 3 / 3 | 2 / 2 | 6 / 7 |
| !iCCup Fighting Spirit 1.3 | fighting_spirit | 1 / 1 | 3 / 3 | 2 / 2 | 6 / 6 |
| Chow Chow | other | 1 / 2 | 2 / 2 | 2 / 2 | 5 / 6 |
| MatchPoint 1.1 | other | 0 / 0 | 3 / 3 | 3 / 3 | 6 / 6 |
| Neo Hunters | other | 0 / 0 | 6 / 6 | 0 / 0 | 6 / 6 |
| Neo The Lost Temple 2.0 | other | 1 / 2 | 4 / 4 | 0 / 0 | 5 / 6 |
| PGT - R - Point 1.0 [07] | other | 2 / 4 | 2 / 2 | 0 / 0 | 4 / 6 |
| PLU_LunaTheFinal 2.1 | other | 0 / 0 | 6 / 6 | 0 / 0 | 6 / 6 |
| WGT12 -- Lost Temple 2.4 ( | other | 2 / 4 | 2 / 2 | 0 / 0 | 4 / 6 |
| WGT14 - Python 1.3 | other | 0 / 0 | 5 / 5 | 1 / 1 | 6 / 6 |
| [CPL] Outsider 2.2 OBS | other | 0 / 0 | 1 / 1 | 5 / 5 | 6 / 6 |
| iCCMC Overwatch 1.0 | other | 0 / 0 | 1 / 1 | 5 / 5 | 6 / 6 |
| \| iCCup \| Central District | other | 0 / 0 | 3 / 3 | 3 / 3 | 6 / 6 |
| \| iCCup \| El Nino | other | 0 / 0 | 4 / 4 | 2 / 2 | 6 / 6 |
| \| iCCup \| NeoMedusa2.0 | other | 0 / 0 | 4 / 4 | 2 / 2 | 6 / 6 |
| \| iCCup \| Return of the Ki | other | 0 / 0 | 3 / 3 | 3 / 3 | 6 / 6 |
| Blitz | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| CEG - Luna | other | 1 / 2 | 3 / 3 | 0 / 0 | 4 / 5 |
| Destination 1.0 | other | 1 / 2 | 1 / 1 | 2 / 2 | 4 / 5 |
| Fighting Spirit 1.3 | fighting_spirit | 1 / 2 | 1 / 1 | 2 / 2 | 4 / 5 |
| Martian Cross | other | 1 / 2 | 3 / 3 | 0 / 0 | 4 / 5 |
| Monty Hall 1.1 | other | 2 / 4 | 1 / 1 | 0 / 0 | 3 / 5 |
| New Heartbreak Ridge 2 | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| PGT - R - Point 1.0 [06] | other | 1 / 2 | 3 / 3 | 0 / 0 | 4 / 5 |
| PGT - Rush Hour 2.0 [06] | other | 1 / 2 | 3 / 3 | 0 / 0 | 4 / 5 |
| PGT - Rush Hour III [09] | other | 1 / 2 | 3 / 3 | 0 / 0 | 4 / 5 |
| Perversion Python | other | 0 / 0 | 2 / 2 | 3 / 3 | 5 / 5 |
| Requiem | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| The Lost War 1.3 (Obs) | other | 0 / 0 | 3 / 3 | 2 / 2 | 5 / 5 |
| Troy 1.0 | other | 2 / 4 | 1 / 1 | 0 / 0 | 3 / 5 |
| VGT20 - BW - Space | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| WCG_Neo Hall of Valhalla  | other | 2 / 4 | 1 / 1 | 0 / 0 | 3 / 5 |
| WGT14 - PeaksOfBaekdu 2.4  | other | 1 / 2 | 1 / 1 | 2 / 2 | 4 / 5 |
| WGT14 - Tau Cross 1.1 (Ob) | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| WGTour (5) Lost Temple 2.1 | other | 1 / 1 | 3 / 3 | 1 / 1 | 5 / 5 |
| [S.G] Lost Temple | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| \| iCCup \| Athena II 2.0 OB | other | 1 / 2 | 2 / 2 | 1 / 1 | 4 / 5 |
| \| iCCup \| Eddy 1.02 | other | 1 / 2 | 2 / 2 | 1 / 1 | 4 / 5 |
| \| iCCup \| Gladiator 1.1 | other | 0 / 0 | 4 / 4 | 1 / 1 | 5 / 5 |
| \| iCCup \| God's Garden OBs | other | 0 / 0 | 1 / 1 | 4 / 4 | 5 / 5 |
| \| iCCup \| GreatBarrier Ree | other | 0 / 0 | 3 / 3 | 2 / 2 | 5 / 5 |
| Abyss Tau Cross [03] | other | 1 / 2 | 1 / 1 | 1 / 1 | 3 / 4 |
| Alternative 1.3 | other | 0 / 0 | 1 / 1 | 3 / 3 | 4 / 4 |
| Athena 1.0 | other | 2 / 4 | 0 / 0 | 0 / 0 | 2 / 4 |
| Benzene0.90 | other | 0 / 0 | 3 / 3 | 1 / 1 | 4 / 4 |
| Byzantium II 2.2 | other | 1 / 1 | 2 / 2 | 1 / 1 | 4 / 4 |
| Carthage 1.0 | other | 0 / 0 | 2 / 2 | 2 / 2 | 4 / 4 |
| Circuit Breakers 1.0 (수 | circuit_breaker | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| Dire Straits | other | 0 / 0 | 1 / 1 | 3 / 3 | 4 / 4 |
| HSL 서킷브레이커 1.0 | circuit_breaker | 0 / 0 | 1 / 1 | 3 / 3 | 4 / 4 |
| Indian Lament | other | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| Jim Raynor's Memory | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| Longinus | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| MERCURY | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| Nostalgia | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| PGT - Luna The Final [04] | other | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| PGT - Luna the Final [08] | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| PGT - Luna the Final [09] | other | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| PGT - Ride of Valkyries [0 | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| Perversion Aztec | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| Symmetry of Psy | other | 0 / 0 | 3 / 3 | 1 / 1 | 4 / 4 |
| The Fortress 1.1 | other | 0 / 0 | 0 / 0 | 4 / 4 | 4 / 4 |
| Ultimatum 1.01 | other | 0 / 0 | 1 / 1 | 3 / 3 | 4 / 4 |
| WGT13 - Estrella 1.0 | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| WGT13 - Longinus 2.0 (Ob) | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| WGT13 - R - Point 1.0 (Ob) | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| WGTOUR Temple Gamei | other | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| iCCMC Bird Nest 1.01 | other | 0 / 0 | 4 / 4 | 0 / 0 | 4 / 4 |
| \| iCCup \| Central Plains | other | 0 / 0 | 0 / 0 | 4 / 4 | 4 / 4 |
| \| iCCup \| Dogma | other | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| \| iCCup \| Neo Medusa 2.2 | other | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| \| iCCup \| Tornado OBs | other | 0 / 0 | 3 / 3 | 1 / 1 | 4 / 4 |
| \| svn \| Destination 1.1 OB | other | 0 / 0 | 1 / 1 | 3 / 3 | 4 / 4 |
| 투혼 1.2 | fighting_spirit | 1 / 2 | 2 / 2 | 0 / 0 | 3 / 4 |
| 팔진도 1.1 | other | 2 / 4 | 0 / 0 | 0 / 0 | 2 / 4 |
| 815Ⅲ 3.2 | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| Byzantium II 2.3 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| Defiler Perversion 1 | other | 1 / 2 | 0 / 0 | 1 / 1 | 2 / 3 |
| Defiler Perversion 3 ( | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| Gladiator1.1 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| GrandLine 1.0 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| Hitchhiker1.0 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| Moriya 1.1 | other | 1 / 2 | 0 / 0 | 1 / 1 | 2 / 3 |
| Neo Arkanoid 2.0 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| Neo Heartbreak Ridge 2 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| Neo Requiem 2.0 | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| PGL_Arcadia II | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| PGL_BlitzX 2.0 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| PGT - Gaia 1.0 [08] | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| PGT - Luna The Final [05] | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| PGT - Luna the Final [10] | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| PGT - Neo Requiem 2.0 [03] | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| PGT - Nostalgia [02] | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| PGT - ParanoidAndroid1.0 [ | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| PGT - R - Point 1.0 [09] | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| PGT - Rush Hour III [08] | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| PLU_Rush Hour 3.03 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| Perversion BB | other | 0 / 0 | 0 / 0 | 3 / 3 | 3 / 3 |
| River of Flames | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| Roadrunner0.91 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| STL_LunaTheFinal 2.1 | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| Super★빠른무한(Ob내전용 | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| TLT - Blade Storm 1.5 (Ob) | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| The Lost Temple_GaMax | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| Triathlon 1.1 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| Triathlon 1.11 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| VGT26 - BW - Space | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| WGT12 - Nostalgia 1.3 B (O | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| WGT12 - Sin Gaema Gowon B  | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| WGT12 -- Lost Temple 2.4 | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| WGT13 - Gaia 1.0 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| WGT13 - IntoTheDarkness II | other | 1 / 2 | 0 / 0 | 1 / 1 | 2 / 3 |
| WGT13 - Longinus 2.0 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| WGT13 - Tau Cross 1.1 (Ob) | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| WGT14 - Python 1.3 (Ob) | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| WGTOUR Temple WCG | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| WGTour (6) Neo Lost Temple | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| WGTour (7) Neo Lost Temple | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| ZodiAc | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| \| iCCup \| Byzantium 3.0 OB | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| \| iCCup \| El Nino OBs | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| \| iCCup \| Faoi 2.1 OBs | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| \| iCCup \| Holy World 1.1 | other | 0 / 0 | 0 / 0 | 3 / 3 | 3 / 3 |
| \| iCCup \| Katrina 1.3 | other | 0 / 0 | 3 / 3 | 0 / 0 | 3 / 3 |
| \| iCCup \| Latin Quarter 1. | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| \| iCCup \| Mist 1.0 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| \| iCCup \| Mist 1.0 OB | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| \| iCCup \| Monty Hall_SE 2. | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| \| iCCup \| MoonGlaive 1.2 O | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| \| iCCup \| PeaksOfBeakdu OB | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| \| iCCup \| Roadrunner 1.3 | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| \| iCCup \| Shades Of Twilig | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| \| iCCup \| The Latin Quarte | other | 1 / 2 | 0 / 0 | 1 / 1 | 2 / 3 |
| \| iCCup \| Vampire 1.0 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| \| iCCup \| Yellow 1.1 | other | 0 / 0 | 1 / 1 | 2 / 2 | 3 / 3 |
| ޫ߾ - Dreamliner 1.0 | other | 0 / 0 | 0 / 0 | 3 / 3 | 3 / 3 |
| 신 백두대간 2.41 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
| 신 추풍령2.1 | other | 0 / 0 | 0 / 0 | 3 / 3 | 3 / 3 |
| 신 피의 능선 2.0 | other | 1 / 2 | 1 / 1 | 0 / 0 | 2 / 3 |
| 저격능선 1.1 | other | 0 / 0 | 2 / 2 | 1 / 1 | 3 / 3 |
|  \| iCCup \| Vampire OBs | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| (2)Cross Game | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| ??퍞 1.3 | fighting_spirit | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| AGL PeaksOfBaekdu 2.4 (Ob) | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Abyss Longinus [03] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Andromeda 0.95 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| Arcadia | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Big Game Hunters 6/7 or 1 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Blue Storm 1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Byzantium 1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| CL16 - Python 1.1 (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Conquer Fever II | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| GGL Dahlia of Jungle | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| HSL 신피의능선 2.1 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| HSL 저격능선 1.1 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Hall of Valhalla 1.0 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Hitchhiker1.1 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Hunters_KeSPA 1.3 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Hunters_KeSPA 1.4 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Incubus | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| KBK21-The Lost Temple | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Katrina_SE1.90 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| KiSaDanTemple | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Korhal of Ceres | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Loki II | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| Mist 0.91 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Mist 1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| MoonGlaive0.92 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| MoonGlaive1.1 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Neo Chain Reaction 2.0 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Neo Guillotine | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Neo Vertigo | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| NeoMoonGlaive2.0 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Odd-Eye 1.0 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Odd-Eye III 3.0 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| Othello 0.99 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Othello 0.991 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Outlier 1.0 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| PGL_Luna The Final | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGL_Reverse Temple | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| PGL_Rush Hour III | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGT - Blade Storm [02] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGT - Gaia 1.0 [04] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGT - Gaia 1.0 [07] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGT - Lost Temple 2.35 [01 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| PGT - Lost Temple 2.4 [02] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGT - Neo Forte 2.1 [08] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PGT - Neo Requiem 2.0 [07] | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| PGT - Rush Hour 2.0 [07] | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PLU_Longinus II | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PLU_LunaTheFinal | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PLU_LunaTheFinal 2.1 M | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PLU_Reverse Lost Temple | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| PLU_Ride of Valkyries | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Perversion Outlier | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Plains to Hill Desert | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Python 1.2 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| RN_815 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| Retro Tech | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Rush Hour | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| SCCL_S1_Gaia(o) | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| SCCL_S1_Luna(o) | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| SCCL_S1_R_o_Valks(o) | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| STL_New Peaks of Baekdu | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Sauron | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| Sun and Steel | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| TLT - Isles of Siren (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| TLT - Namja Iyagi (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| TLT - Sin Gaema Gowon (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| Twisted Trail | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| VGT30 - SC - Space | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| WCG Gaia 1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| WCG_Longinus (ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WCG_Neo Legacy of Char | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| WCG_Othello 1.1 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| WCG_Rivalry | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT10 - Xeno Sky 1.0 (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT11 - Sin Gaema Gowon B  | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT11 - Xeno Sky 1.0 (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT12 - Gaia 0.90 (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT13 - Arkanoid 1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| WGT13 - Azalea 1.0 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT13 - Nostalgia 1.3 B (O | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT13 - Rush Hour 2.0 (Ob) | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| WGT13 - Rush Hour III (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT14 - Blitz X 2.0 (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT14 - ParanoidAndroid1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| WGT9 - Gorky Island (Ob) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGT9 - Lost Temple 2.3 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| WGTOUR (4) Neo Jungle Stor | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGTOUR Plains to Hill | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| WGTour (5) Nostalgia B (Ob | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| WSL Lost Temple 2.1 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| [AI] BGH 3.0 BETA | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| [SaSin] Python 1.3 _v.D | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| [SaSin]Python 1.3_(DTR) | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| [pOk] 투혼 1.3 | fighting_spirit | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| iCCMC Demonio Azul 2.02 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| iCCMC Feel My Love 1.02 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| iCCMC Shuriken 2.0 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| \| Sigma \| 0.31b OBs | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Athena II 2.0 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| \| iCCup \| Avalon 0.93 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| \| iCCup \| Byzantium 3.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Devils Crossing  | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| \| iCCup \| Holy World 1.1 O | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| \| iCCup \| Katrina_SE 2.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Katrina_SE 2.0 O | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Medusa 2.2 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Neo Medusa 2.2 O | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| \| iCCup \| Overwatch 2.1 OB | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Queensbridge 1.3 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| \| iCCup \| Reverse Temple | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| \| iCCup \| Roadrunner 1.0 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| \| iCCup \| Shine obs | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| \| iCCup \| Vampire 1.0 Ob | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| \| iCCup \| Wuthering H 1.0 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 神의 정원 1.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| 몽환 1.1 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 몽환 Ⅱ 2.0 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| 신 개척시대 2.1 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 심판의 날 1.1 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 심판의 날 1.2 | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| 아크로 0.92 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| 용오름 1.0 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 추풍령1.1 | other | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 태양의 제국 1.1 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| 투혼 1.3 수동(Ob) | fighting_spirit | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| 투혼(OB) 1.3 (Boni) | fighting_spirit | 0 / 0 | 1 / 1 | 1 / 1 | 2 / 2 |
| 폭풍의 언덕 0.95 | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| 피의 능선 1.2 | other | 0 / 0 | 0 / 0 | 2 / 2 | 2 / 2 |
| �BWCL� Rush Hour | other | 1 / 2 | 0 / 0 | 0 / 0 | 1 / 2 |
| �Fa�te�t Po��ible  Maps   | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
| �Fa�te�t Po��ible Map Fin | other | 0 / 0 | 2 / 2 | 0 / 0 | 2 / 2 |
|  \| iCCup \| Vampire | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| (2)Micro Jail | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| (2)Path of Victory | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| (2)Yaxchilan | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| (4) Lost Temple  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| (4)GrandLine_SE2.0.scx | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 3 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| <WCG>The Lost Temple | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Abyss Azalea [03] | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Abyss Reverse Temple [ | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Ak Guild 투혼 [ob] | fighting_spirit | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Andromeda 1.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Aphrodite 4.7 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Arkanoid 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Asgard0.90 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Athena II 2.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Avant-garde II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Aztec0.90 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| BKC-7 Paranoid Android  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| BWCL Outsider 2.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| BWCL S-20 Gaia1.0_ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| BWCL S-20 Luna_ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| BWL - The Lost Temple | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Beltway 1.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Benzene1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Blade Storm | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Blitz II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Blue Storm 0.97 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Byzantium III 3.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| CEG - Neo Forte 2.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Chain Reaction 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Circuit Breakers 0.90 | circuit_breaker | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Colosseum0.93 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Colosseum0.96 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| ColosseumII1.92 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Crusader | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Dante's Peak 1.01 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Defiler Perversion 1 ( | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Defiler Perversion 2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Defiler Perversion 2 ( | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Defiler Perversion 4 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Dream Tour - Temple Gamei | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| E-SL_Lost Temple | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| ESL - Luna the Final | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Fastest DroT | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Fastest Fatal 2 OBS | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Fastest Space 6.3 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Forbidden Zone | other | 1 / 1 | 0 / 0 | 0 / 0 | 1 / 1 |
| FunFunGame Destinatio | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| GGL Lost Temple | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Gemlong 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Gorky Park | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| HSL 투혼 1.3 | fighting_spirit | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Hall of Valhalla V3.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Heartwood | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Hype | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| I don't know how to make r | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEF Chupung Ryeong 2.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEF_BlueStorm 1.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEF_Tau Cross 1.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEST_Longinus II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEST_Luna The Final | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEST_R-Point | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEST_Ride of Valkyries | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| IEST_Rush Hour III | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Incubus 2004 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Into The Darkness Par | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Jade 0.90 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Jim Raynor's Memory J 1.5 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Jungle story 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Katrina1.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Katrina_SE1.91 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Katrina_SE2.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| King of the Abyss | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| King of the Abyss (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| LRM Destination 1.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Longinus II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| LostTemple RushFam | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Luna - MBCgame | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| MatchPoint 0.91 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Monte Cristo 1.03 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| MoonGlaive0.91 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| NSL_R - Point | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Namja Iyagi | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Neo Forbidden Zone 2.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Neo Hall of Valhalla | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Neo Legacy of Char | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Neo Outlier 2.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Neo Requiem 2.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| New Blood Bath Advance 1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| New Peaks of Beakdu 2.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Nightlight(Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Odd-Eye II 2.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Othello 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Oxide | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| PGL_Longinus II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Azalea 1.0 [08] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Azalea 1.0 [10] | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| PGT - Gaia 1.0 [09] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Longinus [10] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Lost Temple 2.4 [07] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Luna 2.01 [02] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Luna The Final [06] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Neo Forte 2.1 [07] | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| PGT - Neo Requiem 2.0 [05] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Nostalgia [03] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Nostalgia [05] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - R - Point 1.0 [05] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Rush Hour 1.0 [04] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Rush Hour III [10] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Sin Gaema Gowon [01] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PGT - Sin Gaema Gowon [07] | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PLU_Andromeda 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PLU_Arcadia II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PLU_Arcadia II M | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PLU_Bakdu 2.4 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PLU_Sin 815 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PLU_Tau Cross 1.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| PSL - Python 1.3 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Paradoxxx 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Paradoxxx ll 2.0 - EZ | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Parallelism 1.34 observers | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Perversion Outlier (obs) | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Plasma 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Queensbridge 1.1(obs)wip | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Ride of Valkyries1.0  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Rivalry | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Rosario 1.4 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Rush Hour 3.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| S ignal(Ob) | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| SBL] Infinity | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| SCCL_S1_R_-_Point(o) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| SCD  20 -- Space | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| SSL3_RushHour 3.03 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| SSabi | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| STL_Soul of Fighter 1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| STL_Tau Cross 1.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| STM_LunaTheFinal 2.1  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| STM_R - Point | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Shooting Star | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| SiGLost Temple Gam | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Silent Vortex | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Snowbound | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Snowbound_Starc | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Star Gates | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Super ★ 빠른무한 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| T.[FS]_SoulofFighter | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| TLT - Lost Temple 2.4 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| TLT - Sin Gaema Gowon | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Technetium | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| The Bifrost | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| The Eye | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| The Hunters-Gamei | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| The KORHAL | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| The Lost Temple_ | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| The Lost Temple_KTF | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Tiamat 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| UGL 14 - Anti Hack | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| Un'Goro Crater 1.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| VGT22 - BW - Space | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| VGT25 - SC - Space | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| VGT27 - BW - Space | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| Vendetta 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WCG The Lost Temple V2 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WCG_Neo Jungle story | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WCG_Python 1.3 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT10 - Bifrost lll (Ob) | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGT10 - Namja Iyagi (Ob) | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGT10 - Nostalgia 1.3 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT10 -- Lost Temple 2.4 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT11 - Nostalgia 1.3 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT11 -- Lost Temple 2.4 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT11 -- Lost Temple 2.4 ( | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT12 - Dahlia of Jungle ( | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT12 - Enter the D 2004 ( | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT12 - Gaia 0.90 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT12 - IntoTheDarkness II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT12 - Nostalgia 1.3 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT12 - Vertigo Plus 3.1 ( | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGT13 - Arcadia II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Azalea 1.0 (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Blitz X 2.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Estrella 1.0 (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Longinus (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Nostalgia 1.3 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - ParanoidAndroid1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - PeaksOfBaekdu 2.4 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGT13 - R - Point 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGT13 - Reverse Temple 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Ride of Valkyries | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Rush Hour 2.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 - Rush Hour III | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT13 -- Lost Temple 2.4 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGT13 -- Lost Temple 2.4 ( | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT14 - Blue Storm 1.1 (Ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT14 - Fantasy 1.1 (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT14 - Lost Temple 2.4 (O | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT9 - Korhal of Ceres | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT9 - Korhal of Ceres (Ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT9 - Lost Temple 2.3 (Ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGT9 - Xeno Sky | other | 1 / 1 | 0 / 0 | 0 / 0 | 1 / 1 |
| WGTOUR (2) Plains to Hill | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (6) JR Memory Jungl | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (6) Legacy of Char  | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGTour (6) Nostalgia B (Ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (7) Detonation (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (7) Enter the D 200 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (7) Neo Forbidden Z | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (8) Gorky Park | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (8) Gorky Park (Ob) | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| WGTour (8) Lost Temple_Gam | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WGTour (8) Parallel Lines  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WSL Lost Temple 2.1 (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WSL Neo Forbidden Zone | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| WindyTemple 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| [Dream]투혼 1.3 | fighting_spirit | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| [Op Bsw-]투혼 1.3 | fighting_spirit | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| [Op NsPClaN-] 투혼 1.3 | fighting_spirit | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| [gm] Circuit Breakers 1 | circuit_breaker | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| [드림] Circuit Breakers  | circuit_breaker | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| iCCMC Infinity 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| iCCMCob Bird Nest 1.01 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| luna-MBCgame | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| nG-i Lost Temple 2.3 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| tC LoSt TemPle(ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| MSL \| Python 1.3 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| Sigma \| 0.3b OBs | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Adrenaline Rush | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Arcadia 1.03 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Azalea | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Benzene 1.1 OB | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Byzantium II 2.2 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Calypso | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Carthage 2.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Carthage 2.0 OBs | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Carthage1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Colosseum 2.0 OB | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Cross Game 1.04  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Cross Game 1.06  | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Demian 1.02 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Demian 1.02 OB | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Desperado 1.01 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Hannibal 1.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Heartbeat 1.06 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Medusa 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Medusa 2.2 OB | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Mirage II | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Neo Harmony 2.1  | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Neo Heartbreak R | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Neo Overwatch 2. | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Neo Requiem 2.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Paradoxxx 2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Paranoid Android | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Python 1.3  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Ridge to Ridge O | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| River of Flames  | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Roadkill 1.08 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Roadkill 1.08 OB | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Rush Hour III | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Rush Hour III ob | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Sword in the Moo | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Tears of the Moo | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Troy 1.0 (Ob) | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| \| iCCup \| Ultimatum 1.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| UnGoro Crater 1. | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Wuthering H 1.0  | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| \| iCCup \| Yellow 1.1 OBs | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 개척시대 1.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 백마고지 1.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 신 백두대간 2.2 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 신 용오름 2.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 신 풍운 2.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 신 풍운 2.2 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 심판의 날 0.90 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 아크로 0.91 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 왕의 귀환 1.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 왕의 귀환 1.2 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 용오름 0.93 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 은빛날개 0.90 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 은빛날개 0.91 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 제 1회 BSL 파이썬 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 철의 장막 1.01 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 청풍명월 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 태양의 제국 1.0 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 태양의 제국 1.3 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 태양의 제국 1.4 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 투혼 1.3 Fish Op WinG- | fighting_spirit | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| 해설크래프트  데스티네이션 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| 황혼의 그림자 1.1 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| �BWCL� Chupung Ryeong | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| �BWCL� Colosseum 2.0 | other | 0 / 0 | 0 / 0 | 1 / 1 | 1 / 1 |
| �BWCL� Destination 1.1 | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| �BWCL� Heartbreak Ridge | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| �Fa�te�t Po�ble  Maps [N | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| �Fa�te�tPo�bleMapEver Fan | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |
| �[pG]�-=TeMpLe | other | 0 / 0 | 1 / 1 | 0 / 0 | 1 / 1 |

## src별

| src | 게임 | 시점 | 스텝 |
|---|---|---|---|
| community | 361 | 396 | 1473492 |
| personal | 131 | 152 | 290021 |
| scr_ladder | 1021 | 1233 | 3513375 |
| stardata | 32367 | 34664 | 81092010 |

## stage별 (040-R9, 기록용)

| stage | 게임 | 시점 | 스텝 |
|---|---|---|---|
| S1 | 23458 | 25257 | 59508805 |
| S2 | 10504 | 11188 | 26860093 |

## 분할별 (041-R12)

| 맵 패밀리 | 분할 | 게임 | 시점 | 스텝 |
|---|---|---|---|---|
| circuit_breaker | test | 73 | 78 | 184715 |
| circuit_breaker | train | 594 | 639 | 1573785 |
| circuit_breaker | val | 58 | 64 | 142377 |
| fighting_spirit | test | 1042 | 1114 | 2582695 |
| fighting_spirit | train | 8388 | 8969 | 21348122 |
| fighting_spirit | val | 1046 | 1128 | 2670325 |
| other | train | 21566 | 23264 | 55096581 |
| other | val | 1113 | 1189 | 2770298 |

## 승패 판정 근거 (040-R8 `result_source`, 시점 단위)

| result_source | result | 시점 | 비율 |
|---|---|---|---|
| bridge | -1 | 5973 | 16.4% |
| bridge | 1 | 6396 | 17.5% |
| leave | -1 | 8275 | 22.7% |
| leave | 1 | 8371 | 23.0% |
| none | 0 | 7421 | 20.4% |
| recorded | -1 | 5 | 0.0% |
| recorded | 1 | 4 | 0.0% |

## 디싱크 폐기율 (040-AC2, 목표 < 10%)

폐기율 = `desync` ÷ (`written` + `desync`), 시점 단위.

| 구분 | 값 | 재생 시점 | 디싱크 | 폐기율 |
|---|---|---|---|---|
| map_family | circuit_breaker | 784 | 3 | 0.4% |
| map_family | fighting_spirit | 11386 | 175 | 1.5% |
| map_family | other | 26574 | 2121 | 8.0% |
| src | community | 1948 | 1552 | 79.7% |
| src | personal | 157 | 5 | 3.2% |
| src | scr_ladder | 1331 | 98 | 7.4% |
| src | stardata | 35308 | 644 | 1.8% |
| 전체 |  | 38744 | 2299 | 5.9% |

## build 결과 (040-R6, 기록 단위)

시점이 정해진 뒤의 결과는 시점 단위, 그 전에 거른 리플레이(필터·중복·파싱 오류)는 리플레이 하나가 기록 하나다.

| src | 리플레이 | written | desync | filtered | unplayable | error | review | 기타 |
|---|---|---|---|---|---|---|---|---|
| community | 8219 | 396 | 1552 | 6552 | 12 | 1 | 0 | 0 |
| personal | 574 | 152 | 5 | 436 | 8 | 0 | 0 | 0 |
| scr_ladder | 4000 | 1233 | 98 | 2926 | 0 | 4 | 0 | 0 |
| stardata | 65646 | 34664 | 644 | 32441 | 301 | 0 | 0 | 0 |

## 필터 사유 (`filtered`, 기록 단위)

`invalid_labels>=20% (040-R19)`은 시점 단위(040-R19), 나머지는 대부분 리플레이 단위다.

| 사유 | src | 기록 |
|---|---|---|
| 0 active players | personal | 92 |
| 0 active players | scr_ladder | 28 |
| 1 active players | personal | 133 |
| 1 active players | scr_ladder | 220 |
| 3 active players | community | 114 |
| 3 active players | personal | 26 |
| 3 active players | stardata | 235 |
| 4 active players | community | 10 |
| 4 active players | personal | 38 |
| 4 active players | stardata | 9 |
| 5 active players | personal | 43 |
| 6 active players | personal | 27 |
| 7 active players | personal | 9 |
| duplicate file in this source (040-R20) | personal | 2 |
| duplicate of another source | community | 5092 |
| excluded map (040-R16) | scr_ladder | 1438 |
| invalid_labels>=20% (040-R19) | personal | 1 |
| invalid_labels>=20% (040-R19) | scr_ladder | 83 |
| invalid_labels>=20% (040-R19) | stardata | 132 |
| map larger than 128 tiles | stardata | 4 |
| no Terran view passes the APM bar | community | 1306 |
| no Terran view passes the APM bar | personal | 22 |
| no Terran view passes the APM bar | scr_ladder | 714 |
| no Terran view passes the APM bar | stardata | 29697 |
| shorter than 5 minutes | community | 30 |
| shorter than 5 minutes | personal | 43 |
| shorter than 5 minutes | scr_ladder | 443 |
| shorter than 5 minutes | stardata | 2364 |
