#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""(M67) 零 GPU：宽度这条杠杆的先验到底能不能从项目内部拿到？

## 为什么问这个

(M62) 给「尺寸臂 d=512」定了价（~22 GPU 小时）并排除了血统混淆（从零单阶段训），
但写死 **⛔ (C2)「宽度的预期效应量」仍无来源 ⇒ 定价不授权开臂**。
(M64)/(M66) 又把选题门加强成两问（分得开吗 / 量得到交付物吗）。
架构侧现在只剩宽度这一条杠杆，而它卡在 (C2)。

本文件不判任何实验，只回答一个**可以用现成文件回答**的问题：
**(C2) 这个先验，在不跑这条臂的前提下，项目内部还有没有来源？**

两个可能的内部来源，各自只需读 `runs/*/config.json` 与 `runs/*/log.json`：

- **来源 A：历史上有没有训过别的宽度？** 若有同口径读数，就能直接算斜率。
- **来源 B：模型现在是不是容量受限？** 训练/验证损失的走势是标准的容量诊断。

**零 GPU、零 API、零判官、零活件改动。** 不碰 test split、不碰 `final_test.sh`。

## ⚠ 披露

本文件**不是**预注册实验：它没有主判据、没有盲写要求，因为它读的是早已存在的配置与日志，
没有"读数"可供事后挑选。v8 的末行 train/val 在写本文件之前已被看过（用 `cat log.json`），
照实披露；下面的**结论不依赖任何具体数值**，只依赖"有没有第二个宽度"与
"(M37)/(M60) 是否已经判过 val 的方向"这两件事。
"""
import json
import os
import sys

ARCH_KEYS = ["d", "depth", "heads", "codes"]
ALL_KEYS = ARCH_KEYS + [
    "steps", "sizes", "p32", "batch", "batch32", "lr", "warmup",
    "init_from", "extra_file", "coarse", "p_coarse", "bias_hidden",
    "pal_aug", "pal_smooth", "clip_w", "clip_bs",
]
# 生成器 = 本项目的 TRD 架构训练；critic_v7 是判别器、trd_smoke 是 40 步冒烟测试
NOT_GENERATOR = {"critic_v7", "trd_smoke", "b7_ddpm_09120300",
                 "lora_convention", "lora_lr1e5"}


def scan_configs(root):
    out = {}
    for name in sorted(os.listdir(root)):
        p = os.path.join(root, name, "config.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            c = json.load(f)
        a = c.get("args", c)
        out[name] = {k: a[k] for k in ALL_KEYS if k in a}
    return out


def scan_logs(root, names):
    out = {}
    for name in names:
        p = os.path.join(root, name, "log.json")
        if not os.path.isfile(p):
            continue
        with open(p, encoding="utf-8") as f:
            h = json.load(f)
        if not h:
            continue
        keys = ("train", "val", "val32")
        rows = [{k: e.get(k) for k in ("step",) + keys} for e in h]
        best = {}
        for k in ("val", "val32"):
            vals = [(e[k], e["step"]) for e in rows if e.get(k) is not None]
            best[k] = {"min": min(vals)[0], "at_step": min(vals)[1],
                       "last": vals[-1][0], "last_step": vals[-1][1]} if vals else None
        tr = [(e["train"], e["step"]) for e in rows if e.get("train") is not None]
        out[name] = {"n_entries": len(rows), "best": best,
                     "train_first": tr[0][0] if tr else None,
                     "train_last": tr[-1][0] if tr else None}
    return out


def main():
    root = sys.argv[1] if len(sys.argv) > 1 else "runs"
    out_path = None
    if "--out" in sys.argv:
        out_path = sys.argv[sys.argv.index("--out") + 1]

    cfgs = scan_configs(root)
    gens = {k: v for k, v in cfgs.items()
            if k not in NOT_GENERATOR and v.get("d") is not None}
    logs = scan_logs(root, sorted(gens))

    # 来源 A：宽度有没有被变过
    widths = {}
    for name, c in gens.items():
        widths.setdefault(c["d"], []).append(name)
    arch_sigs = {}
    for name, c in gens.items():
        sig = tuple(c.get(k) for k in ARCH_KEYS)
        arch_sigs.setdefault(sig, []).append(name)

    src_a = {
        "n_generator_runs": len(gens),
        "widths": {str(k): sorted(v) for k, v in widths.items()},
        "distinct_arch_signatures": len(arch_sigs),
        "arch_signature_keys": ARCH_KEYS,
        "signatures": {"/".join(str(x) for x in k): sorted(v)
                       for k, v in arch_sigs.items()},
        "width_ever_varied": len(widths) > 1,
        "excluded_as_non_generator": sorted(
            n for n in cfgs if n in NOT_GENERATOR),
    }

    # 来源 B：val 曲线的走势（⚠ 描述性，判决见下方 note）
    src_b = {"per_run": logs}

    verdict = "WIDTH_PRIOR_UNOBTAINABLE" if not src_a["width_ever_varied"] else "WIDTH_PRIOR_AVAILABLE"

    res = {
        "verdict": verdict,
        "source_A_internal_width_contrast": src_a,
        "source_B_val_curve_descriptive": src_b,
        "notes": [
            "来源 A：全部生成器训练的 (d,depth,heads,codes) 若只有一种签名，"
            "则项目内部不存在任何宽度对照 ⇒ (C2) 拿不到同口径锚点。",
            "来源 B ⛔ 不可用：(M37) 实测按 val 挑检查点显著更差、(M41) B 臂复现、"
            "(M60) 在 val 已回升之后继续训 12000 步仍买到 +0.1250 ⇒ "
            "val 回升不能读成容量到顶，因此 val 曲线形不成宽度先验。",
            "⇒ 两个内部来源都不成立时，(C2) 在不跑这条臂的前提下不可获得。",
        ],
    }

    print(json.dumps(res, ensure_ascii=False, indent=1, sort_keys=True))
    print()
    print("=== 摘要 ===")
    print(f"生成器训练次数: {src_a['n_generator_runs']}")
    print(f"不同架构签名 {ARCH_KEYS}: {src_a['distinct_arch_signatures']} 种")
    for sig, names in src_a["signatures"].items():
        print(f"  {sig}: {len(names)} 次 {names}")
    print(f"宽度被变过吗: {src_a['width_ever_varied']}")
    for name, lg in sorted(logs.items()):
        b = lg["best"]
        if b.get("val"):
            print(f"  {name}: train {lg['train_first']:.3f}->{lg['train_last']:.3f} | "
                  f"val min {b['val']['min']:.3f}@{b['val']['at_step']} "
                  f"last {b['val']['last']:.3f}@{b['val']['last_step']}")
    print(f"VERDICT: {verdict}")

    if out_path:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(res, f, ensure_ascii=False, indent=1, sort_keys=True)
        print(f"written: {out_path}")


if __name__ == "__main__":
    main()
