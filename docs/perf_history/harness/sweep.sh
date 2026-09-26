#!/bin/bash
# Re-run the per-commit sweep.  usage: sweep.sh <tag> <on|off> [workdir]
# Creates one detached git worktree per commit in commits.txt under <workdir>/wt,
# writes <workdir>/out/<tag>_<commit>.json. Never touches the main checkout.
set -u
H=$(cd "$(dirname "$0")" && pwd)
REPO=$(git -C "$H" rev-parse --show-toplevel)
W=${3:-/tmp/perf_sweep}
mkdir -p "$W/wt" "$W/out"
for c in $(cat "$H/commits.txt"); do
  [ -d "$W/wt/$c" ] || git -C "$REPO" worktree add --detach "$W/wt/$c" "$c" -q
  timeout 900 python "$H/bench_fixed.py" --repo "$W/wt/$c" --hooks "$2" --json-out "$W/out/$1_$c.json" > "$W/out/$1_$c.log" 2>&1
  echo "$c $(tail -1 "$W/out/$1_$c.log" | cut -c1-200)"
done
echo "clean up with: git -C $REPO worktree remove --force <path>  (or: rm -r $W/wt && git -C $REPO worktree prune)"
