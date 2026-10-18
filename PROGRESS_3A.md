# PROGRESS 3A (M6 3단계 구현 A) — 재개 시 이 파일부터 읽는다

브랜치: `worktree-agent-a074350bc376aff78` (origin에 push, master 아님, rebase로 force-push함). 기준: 3단계 계획서(사용자 승인 2026-10-17, Q1~Q6 모두 추천안).

## 완료 (master)
- `f6919d9` 1단계 spec 개정: 042 구현 구조, 042-R3 Aborted 회수(AMB-042-6), AMB-042-5·7·8·9, AMB-050-2, 020-R12, 032-R16, 033-R7, 033-R8.
- `6177d47` 4·5단계 + 020-R13: runner lib(JobSource/QueueSource/Dispatch/InFlight.stop_all/run_source), 추론 서버 swap/state/version, Core host state, ModelPolicy RL hook, PolicySpec::Learner, EndBoard + `game_result`(020-R13, AMB-020-3), runner summary `decisive()`. `cargo test --workspace` 162개 통과.

## 브랜치 WIP (master 전)
- uranium-train `rl/`: config, returns(V-trace·UPGO·GAE), trajectory(PackedObs, 1스텝 겹침 세그먼트, revoke/lag, hold_terminal), learner(V-trace/PPO, BC 재forward KL, 엔트로피, micro_lanes, reset_to_bc), league(PFSP, EMA 승률, 커리큘럼, 스냅샷·리셋, 시분할, 종족·Random, 보상 가중 스케줄), run(오케스트레이터: LeagueSource, 020-R13 판정으로 r_win·승률, 3B storage/ckpt 사용, heartbeat 1분, Ctrl+C, CSV), CLI `rl`·`z-table`, rl.toml `[[t1]]`.
- `cargo test -p uranium-train --lib rl::` 22개 통과. spec 033-R8 명령 이름 `uranium-train z-table`로 수정.

## 받은 결정·주의
- 020-R13: 더 늦게 끝난(GameEnd 프레임 큰) 세션의 승자, 같으면 먼저 받은 보고. 불일치 건수 따로. 리그 승률·r_win·games.csv 모두 이 판정(run.rs `LeagueSource::done`, master e25dce8에 연결 완료). 스모크 불일치 1건은 규칙대로 처리된 정상 기록.
- 3B API: Ctrl+C 정상 종료에서 push가 Pushed가 아니면 `release()` 금지(잠금 유지 → 다음 시작 복구, 061-R11-5·6). heartbeat 1분마다 `run.heartbeat()`, Lost면 학습 중지. (run.rs에 반영)
- 042-AC1 본 실행 금지(준비 상태·예상 시간만 보고).

## 다음 할 일
1. CUDA release 빌드(백그라운드, target `C:/Users/yikol/.claude/jobs/392328e2/tmp/target-a`, CUDA_PATH v13.4).
2. 짧은 BC(Q1): shard `stardata/p00`, `URANIUM_MAPS_DIR`=공용 `datasets/replays/build/maps`, CUDA, 약 1시간.
3. 스모크(C0, 이 PC hostd `127.0.0.1:7750` max_games 3, hosts 파일 `tmp/fork_a/smoke_hosts.toml`, agent `127.0.0.1:7650`, 1시간, 스모크용 lanes 축소 rl.toml 사본) → `reports/m6_rl.md`(처리량, 지연 버림, 추론 p50/p95, 메모리).
4. `cargo test --workspace` 후 rebase → master push. PROGRESS_3A.md는 master에 넣지 않는다.
