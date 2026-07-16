# MembraneQuant

Quantification of membrane localization of EGFP-tagged proteins using DiI membrane staining.

## Design principle

**Do not let DiI define cell boundaries.** DiI only answers “is the membrane stained here?”

1. Segment whole cells from **EGFP** (or EGFP+DiI).
2. Build a fixed-width **membrane ring** from the cell boundary (geometry).
3. Use **DiI red coverage** only as QC for the ring.

## Install

```bash
# from repo root (parent of membranequant/)
pip install -r membranequant/requirements.txt
```

Python 3.11+ recommended (tested on 3.13).

### Optional: Cellpose

```bash
pip install cellpose
```

Cellpose is **not** required for the default Otsu path.

### Optional: Web UI

`gradio` is listed in `requirements.txt`. Launch:

```bash
# Preferred: from repo root (parent of membranequant/)
cd D:\杂物\grok
python -m membranequant --webui --port 7860

# Or from inside the membranequant folder:
cd D:\杂物\grok\membranequant
python main.py --webui --port 7860
# double-click / run:
run_webui.cmd
```

Open `http://127.0.0.1:7860` — **Select folder paths → Run**.

> **Common mistake:** running `python -m membranequant.main` *while already inside*
> the `membranequant` folder causes `ModuleNotFoundError: No module named 'membranequant'`.
> Either `cd` up one level, or use `python main.py --webui`.

## Input layout

```
Experiment/
  GroupA/
    C2_104d1-1.tif      # optional Merge
    C2_104d1-1-1.tif    # Red (DiI)
    C2_104d1-1-2.tif    # Green (EGFP)
  GroupB/
    ...
```

Filename pattern: `{prefix?}{experiment}{drug}-{field}[-{channel}].tif`

| suffix | channel |
|--------|---------|
| (none) | Merge (optional) |
| `-1`   | Red / DiI |
| `-2`   | Green / EGFP |

## Run (CLI)

```bash
# default Otsu segmentation
python -m membranequant.main --input Experiment --output Results

# optional Cellpose path
python -m membranequant.main -i Experiment -o Results --cellpose
python -m membranequant.main -i Experiment --seg cellpose --cellpose-model cyto2 --cellpose-diameter 0

# Web UI
python -m membranequant.main --webui --port 7860
```

Useful flags:

```text
--seg otsu|cellpose
--cellpose                 # alias for --seg cellpose
--cellpose-model cyto2
--cellpose-diameter 0      # 0 = auto
--cellpose-gpu
--ring-width 3
--no-overlay
--no-mask
--pearson
--webui --host 127.0.0.1 --port 7860 --share
```

## Web UI features

- Input / output folder paths (local experiment directory)
- **Preview pairs** before running
- Segmentation: **otsu** or **cellpose**
- Ring width, area filters, Red Coverage QC, preprocess options
- Live-ish log + `summary.csv` / `results.csv` tables + overlay gallery

## Output

```
Results/
  csv/
    results.csv
    summary.csv
    graphpad_MC.csv
  masks/
  overlays/
  qc/
  logs/
```

Main metric: **M/C** = Membrane Green Mean / Cytoplasm Green Mean.

Also: MembraneFraction, RedCoverage (QC).

## Config

See `config.yaml`:

```yaml
segmentation_method: otsu   # or cellpose
cellpose_model: cyto2
cellpose_diameter: 0
cellpose_gpu: false
ring_width: 3
...
```

## Tests

```bash
pip install pytest
pytest membranequant/tests -q
```

## Package layout

```
membranequant/
  main.py            # CLI (+ --webui)
  webui.py           # Gradio UI
  config.py
  io.py
  preprocess.py
  segmentation.py    # otsu + optional cellpose
  measurements.py
  qc.py
  export.py
  visualization.py
  utils.py
  config.yaml
  tests/
```
