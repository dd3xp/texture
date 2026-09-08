# Supplementary code and data (anonymized)

This bundle contains the analysis code, figure code, generation pipeline, and a
snapshot of the primary data behind every quantitative claim in the paper.
Internal research logs and server-ops scripts are omitted; some code comments
are in Chinese, but nothing below requires them.

## Environment

All statistics and figures reproduce on **CPU only** with Python 3.10+ and
`numpy`, `matplotlib`, `Pillow`. **No SciPy is required**: binomial tests use
exact `math.comb` sums and Spearman p-values use permutation tests, both
implemented inline. Only the generation pipeline (SDXL) needs a GPU.

## Data snapshot (included)

| Path | Contents |
| --- | --- |
| `data/tiles/dataset_k16.json` | quantized human tile dataset used by the agreement analysis |
| `experiments/*.json` | raw per-pair records of every reported comparison run (crop/render/spread series, manifest) |
| `experiments/annotate/` | human blind-comparison labels (`a4_labels.csv`, `b2_labels.csv`) and full VLM-judge replays |
| `experiments/figqual/` | rendered tiles behind the qualitative figure |

## Reproducing the paper's numbers

Run everything from the bundle root.

**Pre-registered resolution-gradient protocols (Sec. 4.2):**

```bash
python analysis/paired/crop_res5_eval.py experiments/crop_res5.json
python analysis/paired/crop_res5_eval.py experiments/crop_res5b.json
```

Each prints per-tier win rates with exact binomial p-values, the Spearman
trend with its permutation p, and an explicit check of the pre-registered
pass/fail criteria (which the paper reports as falsified).

**Anisotropy gate (the reproducible metric behind the gating numbers):**

```bash
python analysis/paired/aniso_gate.py
```

**Figures** — regenerates the committed PNGs in `figures/` (verified
byte-identical on a clean clone):

```bash
python analysis/paired/fig_agreement.py    # inter-artist agreement
python analysis/paired/fig_judges.py       # judge-vs-human panels
python analysis/paired/fig_qualitative.py  # qualitative grid
python analysis/paired/fig_gradient.py     # gradient falsification
python analysis/paired/fig_units.py        # structural-unit counts
```

**Cross-domain generalization (emoji redraws):**

```bash
python analysis/paired/emoji_agreement.py
```

This downloads three independently drawn emoji sets (Noto, Twemoji, OpenMoji;
network required on first run, cached under `data/emoji/`) and reports the
foreground per-cell agreement vs. the paired null.

**Human-study accounting:** the raw label CSVs in `experiments/annotate/`
match the counts in the paper directly (60 model-vs-baseline pairs plus 7
model-vs-artist pairs plus 5 attention checks in `a4_labels.csv`; 24 real
pairs plus 3 checks in `b2_labels.csv`).

## Generation pipeline (GPU)

`tools/paint_region.py` (flat-colour image + material name → textured region),
with `tools/from_prompt.py` and `tools/make_texture.py` as standalone stages.
All results reported in the paper use SDXL **base without LoRA**. VLM-judge
scripts read credentials from the `VLM_BASE_URL` / `VLM_API_KEY` environment
variables; no keys are stored in this bundle.
