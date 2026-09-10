# Supplementary code and data (anonymized)

This bundle contains the analysis code, figure code, generation pipeline, and a
snapshot of the primary data behind every quantitative claim in the paper.
Internal research logs and server-ops scripts are omitted; some code comments
are in Chinese, but nothing below requires them.

## Environment

All statistics and figures reproduce on **CPU only** with Python 3.10+ and
`numpy`, `matplotlib`, `Pillow`. **No SciPy is required**: binomial tests use
exact `math.comb` sums and Spearman p-values use permutation tests, both
implemented inline (`analysis/exact.py`). Only the generation pipeline needs a
GPU, and only the judge scripts need network access.

## Data snapshot (included)

| Path | Contents |
| --- | --- |
| `data/tiles/dataset_k16.json` | quantized human tile dataset behind the agreement and unit-convention analyses |
| `experiments/annotate/*_labels.csv` | the three human blind studies: `a4_labels` (72 items), `b2_labels` (27), `study_labels` (171, the largest) |
| `experiments/annotate/vlm_*.json`, `opus5_*.json` | full VLM-judge replays, including the two discarded runs on an unsuitable validation set |
| `experiments/crop_*.json`, `render_*.json` | per-pair records of every crop and render-resolution comparison |
| `experiments/pack/`, `experiments/pack60/` | 78 delivered materials at 16/24/32 with manifests |
| `experiments/prompts_holdout60.json` | the 60 materials used for the generalisation replication |
| `experiments/figqual/` | rendered sources behind the qualitative figure |

## Reproducing specific claims

Run everything from the bundle root.

**Sec. 3.1, inter-artist agreement, and its emoji replication:**

```bash
python analysis/paired/fig_agreement.py     # 0.0977 vs 0.0859 paired null
python analysis/paired/emoji_agreement.py   # foreground gap; downloads three emoji sets on first run
```

**Sec. 3.3, judge admissibility.** `fig_judges.py` regenerates the three-panel
figure; `judge_bootstrap.py` is the stability analysis the section rests on —
it reports that the effect-retention ordering survives resampling in 83% of
draws while the significance threshold between two judges survives in 54%,
which is why the argument is phrased in terms of retention.

```bash
python analysis/paired/fig_judges.py
python analysis/paired/judge_bootstrap.py
```

**Sec. 4, human blind comparisons.** The label CSVs match the paper's counts
directly. `analyze_d6.py` recomputes the largest study, including the 60%
baseline-over-model rate quoted in Sec. 4.1 and the third artist-versus-
baseline measurement in Sec. 4.2.

```bash
python analysis/annotate/analyze_d6.py
```

**Sec. 5.2, the structural-unit convention:**

```bash
python analysis/paired/fig_units.py            # per-tile unit counts
python analysis/paired/resolution_tiers.py     # artist convention under both detector calibers
```

**Sec. 5.3, the gate and the crop's win rates.** `aniso_gate.py` reproduces the
gate's hit and false-trigger rates on the tuning classes and on a held-out
split assigned by material name. `crop_res5_eval.py` takes any of the crop
series and prints per-tier win rates with exact binomial p-values.

```bash
python analysis/paired/aniso_gate.py
python analysis/paired/crop_res5_eval.py experiments/crop_holdout60.json   # the 60 unseen materials
```

**Sec. 5.4, the pre-registered falsification.** Both protocols, with an
explicit check of the criteria fixed before the runs (which the paper reports
as falsified):

```bash
python analysis/paired/crop_res5_eval.py experiments/crop_res5.json
python analysis/paired/crop_res5_eval.py experiments/crop_res5b.json
python analysis/paired/fig_gradient.py
```

**Sec. 6, the materials the gate declines:**

```bash
python analysis/paired/decline_spread.py    # 0.128 vs 0.240 across 78 materials
python analysis/paired/isotropic_scale.py   # why a crop has nothing to align to
```

**The three results whose scripts need a GPU** (the second generator, the
ControlNet attempt and point sampling) ship their raw per-material JSON, so the
reported statistics can be rechecked without one:

```bash
python analysis/paired/recheck_gpu_claims.py   # recomputes each figure and compares it to the text
```

**Figures.** The five committed PNGs in `figures/` regenerate byte-identically
from `fig_agreement.py`, `fig_judges.py`, `fig_qualitative.py`,
`fig_gradient.py` and `fig_units.py`.

## Requiring a GPU or network

| Script | Needs |
| --- | --- |
| `tools/paint_region.py` | GPU to generate; `--tile` runs on CPU against the included tiles |
| `tools/from_prompt.py`, `tools/make_texture.py` | GPU; standalone stages of the same pipeline |
| `scripts/batch_pack.py` | GPU; regenerates a delivery pack |
| `analysis/paired/second_generator.py` | GPU and the SD 1.5 weights (`scripts/fetch_sd15.sh` fetches them) |
| `analysis/paired/controlnet_units.py`, `point_sample_iso.py`, `render_small_iso.py` | GPU |
| `analysis/paired/spread_rescale.py`, `analysis/annotate/vlm_judge.py` | a VLM endpoint via `VLM_BASE_URL` / `VLM_API_KEY` |

All results reported in the paper use SDXL **base without an adapter**. No
credentials are stored in this bundle.

## A note on `analysis/structure_grain/`

Those sixteen scripts implement the structural-descriptor phase that the paper
reports as withdrawn (Appendix A, and the ruler discussion in Sec. 4.1). They
are included so the retraction can be inspected, and they support none of the
paper's claims; that directory's README says which results were withdrawn and
why.
