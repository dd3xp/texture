#!/usr/bin/env bash
# (M78) 只读自检：①两份脚本语法 ②哨兵 grep 模式带阳性+阴性对照 ③当前真实的两样完成证据
# ⛔ 不读日志里任何指标（只数哨兵行、只看文件在不在）⇒ 携 0 比特。
set -u
case "$(hostname)" in
  a100-node03*) ;;
  *) echo "M75_WRONG_HOST_$(hostname)"; exit 4 ;;
esac
for f in /tmp/m75_wait_and_go.sh /tmp/m75_go.sh /tmp/m75_arm.sh; do
  bash -n "$f" && echo "SYNTAX_OK $f"
done
printf 'junk\nM75_W384_DONE\n' > /tmp/m75_probe_pos.txt
printf 'junk\nM75_W384_DONE_NOT\n' > /tmp/m75_probe_neg.txt
grep -q '^M75_W384_DONE$' /tmp/m75_probe_pos.txt && echo POS_MATCH_OK || echo POS_MATCH_FAIL
grep -q '^M75_W384_DONE$' /tmp/m75_probe_neg.txt && echo NEG_BAD || echo NEG_REJECT_OK
rm -f /tmp/m75_probe_pos.txt /tmp/m75_probe_neg.txt
for a in w384 w512; do
  U=$(echo "$a" | tr 'a-z' 'A-Z')
  R=/tmp/runs/trd_${a}_09191345
  ck=NO; [ -f "$R/step_44000.pt" ] && ck=YES
  st=NO; [ -f "$R/step_22000.pt" ] && st=YES
  echo "$a: step_44000.pt=$ck step_22000.pt=$st sentinel=$(grep -c "^M75_${U}_DONE$" /tmp/m75_${a}.txt)"
done
nvidia-smi --query-gpu=index,memory.free --format=csv,noheader
echo M75_CHECK_GUARD_DONE
