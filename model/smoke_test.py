"""D3 冒烟测试：只验证代码能跑通、形状对、约束成立。不是训练。

检查四件事，每一件都对应技术路线里的一条承诺：
  1. 前向/反向能跑，损失有限
  2. 采样输出**只含调色板内的索引**（调色板合规由构造保证）
  3. 区域外的格子保持 BG（区域约束由构造保证）
  4. 无效档（超出该样本用色数）永远不会被采样到
"""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import TileSet          # noqa: E402
from model import build_model, training_step, generate  # noqa: E402

dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = TileSet(Path("data/tiles/dataset_k16.json"), "train", size=16)
print(f"训练样本 {len(ds)}   材质数 {ds.n_materials}   K={ds.k}")

dl = torch.utils.data.DataLoader(ds, batch_size=8, shuffle=True)
batch = next(iter(dl))
ARCH = sys.argv[1] if len(sys.argv) > 1 else "transformer"
net = build_model(ARCH, k=ds.k, n_materials=ds.n_materials, size=16,
                  d=128, depth=6).to(dev)
n_par = sum(p.numel() for p in net.parameters())
print(f"架构 {ARCH}   参数量 {n_par/1e6:.2f}M")

loss, stats = training_step(net, batch, dev)
loss.backward()
gn = sum((p.grad ** 2).sum().item() for p in net.parameters() if p.grad is not None) ** 0.5
print(f"\n1. 前向+反向: loss={stats['loss']:.4f}  acc={stats['acc']:.3f}  "
      f"掩码率={stats['mask_frac']:.2f}  梯度范数={gn:.3f}")
assert torch.isfinite(loss), "损失非有限"

pal = batch["palette"].to(dev)
val = batch["pal_valid"].to(dev)
mat = batch["material"].to(dev)
out = generate(net, pal, val, mat, size=16, steps=8, device=dev)
in_pal = ((out >= 0) & (out < ds.k)).all().item()
print(f"2. 采样输出全部落在调色板内: {in_pal}  (取值范围 {out.min().item()}..{out.max().item()})")
assert in_pal

region = torch.zeros(pal.shape[0], 16, 16, dtype=torch.bool, device=dev)
region[:, 4:12, 4:12] = True
out2 = generate(net, pal, val, mat, size=16, steps=8, region=region, device=dev)
outside_bg = (out2[~region] == net.BG).all().item()
inside_ok = ((out2[region] >= 0) & (out2[region] < ds.k)).all().item()
print(f"3. 区域外保持 BG: {outside_bg}   区域内合法: {inside_ok}")
assert outside_bg and inside_ok

bad = 0
for b in range(pal.shape[0]):
    n_used = int(val[b].sum().item())
    bad += int((out[b] >= n_used).sum().item())
print(f"4. 采样到无效档的格子数: {bad}  (应为 0)")
assert bad == 0

print("\n冒烟测试全部通过。")
