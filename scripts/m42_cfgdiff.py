import json
import os
import sys

a = json.load(open("runs/%s/config.json" % sys.argv[1]))
b = json.load(open("runs/%s/config.json" % sys.argv[2]))
for k in sorted(set(a) | set(b)):
    if a.get(k) != b.get(k):
        print("  %-22s %s=%r | %s=%r" % (k, sys.argv[1], a.get(k), sys.argv[2], b.get(k)))
for r in (sys.argv[1], sys.argv[2]):
    log = json.load(open("runs/%s/log.json" % r))
    print(r, "last_step", log[-1]["step"], "val", round(log[-1]["val"], 4),
          "files", sorted(os.listdir("runs/" + r)))
