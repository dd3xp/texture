#!/bin/bash
# (M61) readiness probe, meant to run ON the GPU node. Exit code only:
#   0 = all 56 criteria JSONs landed AND all four sentinels present
#   nonzero = not ready (also the safe direction for an ssh failure)
case "$(hostname)" in
  a100-node*) ;;
  *) echo "M61_READY_WRONG_HOST: $(hostname)" >&2; exit 4 ;;
esac
n=$(ls /tmp/m61_ctrl_s*.json /tmp/m61_more_s*.json 2>/dev/null | wc -l)
[ "$n" -eq 56 ] || exit 1
for f in /tmp/m61_ctrlA.txt /tmp/m61_ctrlB.txt /tmp/m61_moreA.txt /tmp/m61_moreB.txt; do
  grep -q 'DONE' "$f" || exit 2
done
exit 0
