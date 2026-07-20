# MembraneQuant

Quantification of membrane localization of EGFP-tagged proteins using DualCellQuant + DiI membrane staining.

## Design principle & Architecture

- **Image Analysis Backend**: Powered by [DualCellQuant](https://github.com/fuji3to4/DualCellQuant) (Cellpose segmentation, EDT radial membrane extraction, background correction, Target/Reference masks).
- **Experiment Management & Post-Analysis**: MembraneQuant provides experiment folder scanning, pairing, quality control filtering, GraphPad CSV exporting, Gradio Web UI, and 300 dpi PPT-ready statistical charts.

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

# 🆕 NEW: Multiple watershed methods for adhered cells
python -m membranequant.main -i Experiment -o Results --seg watershed_distance
python -m membranequant.main -i Experiment -o Results --seg hminima_watershed
python -m membranequant.main -i Experiment -o Results --seg combined_markers

# optional Cellpose path
python -m membranequant.main -i Experiment -o Results --cellpose
python -m membranequant.main -i Experiment --seg cellpose --cellpose-model cyto2 --cellpose-diameter 0

# Web UI (recommended)
python -m membranequant.main --webui --port 7860
```

### 🆕 Segmentation Methods (2026-07 Update)

For **severely adhered cells**, we now provide **6 classical image processing methods** (no ML required):

| Method | Speed | Adhesion | Best For | Install |
|--------|-------|----------|----------|---------|
| `otsu` | ⚡⚡⚡ | ❌ | No adhesion (default) | ❌ |
| `watershed_distance` | ⚡⚡ | ⭐⭐⭐ | Round/oval adhered cells (**recommended first**) | ❌ |
| `watershed_gradient` | ⚡⚡ | ⭐⭐ | Clear boundaries | ❌ |
| `hminima_watershed` | ⚡ | ⭐⭐⭐ | Dense cells, over-segmentation | ❌ |
| `morphological_opening` | ⚡⚡⚡ | ⭐ | Mild adhesion | ❌ |
| `combined_markers` | ⚡ | ⭐⭐⭐ | Fallback/robust | ❌ |
| `cellpose` | 🐌 | ⭐⭐⭐⭐ | Last resort | ✅ `pip install cellpose` |

**Recommended workflow for adhesion:**
1. Try `watershed_distance` (ImageJ classic watershed)
2. If over-segmented → `hminima_watershed`
3. If still not good → `combined_markers`

See detailed guide: `粘连细胞分割方法说明.md` (Chinese) or `分割方法快速参考.txt`

Useful flags:

```text
--seg otsu|watershed_distance|watershed_gradient|hminima_watershed|morphological_opening|combined_markers|cellpose
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
- Segmentation: **otsu**, **6 watershed variants for adhesion**, or **cellpose**
  - 🆕 Dropdown menu with detailed descriptions for each method
  - 🆕 Built-in guidance for choosing the right method
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

### Metrics (what to report)

| Metric | Meaning | Use |
|--------|---------|-----|
| **Ratio_T_over_R** | T/R pixel-wise intensity ratio (Dual membrane ROI) | **Primary** DualCellQuant membrane localization |
| **RatioOfMeans_T_R** | Target mean / Reference mean on Dual membrane ROI | Mean-level T/R ratio |
| **Enrichment_Membrane_vs_Whole** | Membrane EGFP mean / Whole-cell EGFP mean | EGFP membrane enrichment |
| **MembraneFraction** | Integrated EGFP on membrane / Whole-cell total | Fraction of signal on membrane |
| **RedCoverage** | Reference (DiI) coverage of membrane ROI | **QC only**, not biology |

Pipeline: background correction → cell segmentation (EGFP) → geometric membrane ring + **DiI-guided membrane mask** → enrichment + Manders/Pearson → QC → CSV + `plots/` (300 dpi).

## Config

See `config.yaml`:

```yaml
segmentation_method: otsu   # otsu, watershed_distance, watershed_gradient,
                            # hminima_watershed, morphological_opening, 
                            # combined_markers, or cellpose
cellpose_model: cyto2
cellpose_diameter: 0
cellpose_gpu: false
ring_width: 3
...
```

**For adhered cells**, change `segmentation_method` to:
- `watershed_distance` (recommended first)
- `hminima_watershed` (if over-segmented)
- `combined_markers` (robust fallback)

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
