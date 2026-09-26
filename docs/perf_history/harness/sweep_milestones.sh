#!/bin/bash
# Long-context milestone sweep (LONG_CONTEXT.md, 2026-09-26).
# Runs bench_ctx.py against each commit checked out in its own worktree under ./wt/.
#   usage: docs/perf_history/harness/sweep_milestones.sh [outdir]
H=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$H" rev-parse --show-toplevel)
OUT=${1:-$H/out}; mkdir -p "$OUT" "$H/wt"
run() {  # run <commit> <context> [--cuda-graphs]
  local c=$1 ctx=$2 flag=$3 tag=$1${3:+-graphs}_$2
  [ -d "$H/wt/$c" ] || git -C "$REPO" worktree add --detach "$H/wt/$c" "$c" -q
  timeout 1800 python "$H/bench_ctx.py" --repo "$H/wt/$c" --seq-len "$ctx" $flag --json-out "$OUT/$tag.json" > "$OUT/$tag.log" 2>&1
  echo "$tag $(tail -1 "$OUT/$tag.log" | cut -c1-160)"
}
HEAD_COMMIT=1f28138
for ctx in 512 8192 32768 65536 114688; do
  for c in 8cea929 9bd7daf $HEAD_COMMIT; do run $c $ctx; done
  run $HEAD_COMMIT $ctx --cuda-graphs
done
# split optimization #8 into its commits
for ctx in 32768 65536 114688; do
  for c in a496b35 a861708 e9610a1; do run $c $ctx; done
done
echo "clean up worktrees with: git worktree remove <path> (see git worktree list)"
