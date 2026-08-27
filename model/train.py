"""D4：训练掩码预测模型。

盯的不是训练损失，是**验证损失与训练损失的差距**——
3395 张 16×16 配 4.95M 参数，过拟合画风是主要风险，
而验证集按包留出，所以验证损失衡量的正是"对没见过的画师的泛化"。

验证时固定掩码率（不随机），否则不同轮次之间不可比。
"""

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import TileSet                                    # noqa: E402
from model import PixelTextureModel, training_step          # noqa: E402


@torch.no_grad()
def evaluate(model, loader, device, rates=(0.25, 0.5, 0.75)):
    """在几个固定掩码率上评估，取平均。固定才可比。"""
    model.eval()
    tot = {"loss": 0.0, "acc": 0.0, "n": 0}
    for batch in loader:
        idx = batch["idx"].to(device)
        pal = batch["palette"].to(device)
        val = batch["pal_valid"].to(device)
        mat = batch["material"].to(device)
        B, N, _ = idx.shape
        for r in rates:
            g = torch.Generator(device="cpu").manual_seed(1234)
            m = (torch.rand(B, N, N, generator=g).to(device) < r)
            m[:, 0, 0] |= ~m.view(B, -1).any(1)
            inp = torch.where(m, torch.full_like(idx, model.MASK), idx)
            logits = model(inp, pal, val, mat)
            invalid = (val < 0.5)[:, None, :].expand(-1, N * N, -1)
            logits = logits.masked_fill(invalid, float("-inf"))
            sel = m.view(B, -1)
            tgt = idx.view(B, -1)
            loss = F.cross_entropy(logits[sel], tgt[sel], reduction="sum")
            acc = (logits[sel].argmax(-1) == tgt[sel]).float().sum()
            tot["loss"] += loss.item()
            tot["acc"] += acc.item()
            tot["n"] += int(sel.sum().item())
    model.train()
    return tot["loss"] / tot["n"], tot["acc"] / tot["n"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("data/tiles/dataset_k16.json"))
    ap.add_argument("--size", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=300)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--depth", type=int, default=6)
    ap.add_argument("--dim", type=int, default=256)
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--max-minutes", type=float, default=12.0,
                    help="墙钟上限。无人值守时单次 GPU 任务不得超过 15 分钟")
    ap.add_argument("--out", type=Path, default=Path("experiments/model"))
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tr = TileSet(args.data, "train", size=args.size, augment=True)
    va = TileSet(args.data, "val", size=args.size, augment=False)
    print(f"train {len(tr)}  val {len(va)}  材质 {tr.n_materials}  K {tr.k}")

    dtr = DataLoader(tr, batch_size=args.batch, shuffle=True, drop_last=True,
                     num_workers=4, persistent_workers=True)
    dva = DataLoader(va, batch_size=args.batch, num_workers=2)

    net = PixelTextureModel(k=tr.k, n_materials=tr.n_materials, size=args.size,
                            d=args.dim, depth=args.depth, drop=args.drop).to(dev)
    print(f"参数量 {sum(p.numel() for p in net.parameters())/1e6:.2f}M")

    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=args.wd)
    total = args.epochs * len(dtr)

    def lr_at(step):
        if step < args.warmup:
            return step / max(args.warmup, 1)
        p = (step - args.warmup) / max(total - args.warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_at)

    hist, best, best_ep, step = [], float("inf"), -1, 0
    t0 = time.time()
    stop = "跑满设定轮数"
    for ep in range(1, args.epochs + 1):
        agg = {"loss": 0.0, "acc": 0.0, "n": 0}
        for batch in dtr:
            loss, st = training_step(net, batch, dev)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            step += 1
            agg["loss"] += st["loss"]
            agg["acc"] += st["acc"]
            agg["n"] += 1
        trl, tra = agg["loss"] / agg["n"], agg["acc"] / agg["n"]

        if ep % 5 == 0 or ep == 1:
            vl, vacc = evaluate(net, dva, dev)
            gap = vl - trl
            hist.append({"epoch": ep, "train_loss": trl, "train_acc": tra,
                         "val_loss": vl, "val_acc": vacc, "gap": gap})
            mark = ""
            if vl < best:
                best, best_ep = vl, ep
                torch.save({"state": net.state_dict(), "args": vars(args),
                            "k": tr.k, "n_materials": tr.n_materials,
                            "mat2id": tr.mat2id, "size": args.size},
                           args.out / "best.pt")
                mark = "  *"
            print(f"ep {ep:>4}  训练 {trl:.4f}/{tra:.3f}   "
                  f"验证 {vl:.4f}/{vacc:.3f}   差距 {gap:+.4f}{mark}")

        if (time.time() - t0) / 60 > args.max_minutes:
            stop = f"到达墙钟上限 {args.max_minutes} 分钟（第 {ep} 轮）"
            break

    mins = (time.time() - t0) / 60
    print(f"\n结束：{stop}，用时 {mins:.1f} 分钟")
    print(f"最佳验证损失 {best:.4f} @ ep{best_ep}")
    if hist:
        h = hist[-1]
        print(f"末轮 训练 {h['train_loss']:.4f} / 验证 {h['val_loss']:.4f} "
              f"→ 差距 {h['gap']:+.4f}")
    (args.out / "history.json").write_text(json.dumps(
        {"history": hist, "best_val": best, "best_epoch": best_ep,
         "minutes": mins, "stop_reason": stop}, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()
