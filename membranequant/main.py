"""MembraneQuant CLI entry point.

Architecture
------------
- **I/O**: scan experiment folders, pair Red(DiI)/Green(EGFP) TIFs
- **Image analysis**: DualCellQuant (background → Cellpose → EDT radial membrane
  → masks → per-cell intensity / T/R)
- **Post-analysis**: QC, CSV/GraphPad, statistical plots, Web UI

Recommended ways to start Web UI:

  # From repo root (D:\\杂物\\grok):
  python -m membranequant --webui --port 7860

  # From inside this folder:
  python main.py --webui --port 7860
  run_webui.cmd
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

_REPO = Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from membranequant.config import Config, load_config
from membranequant.dual_backend import analyze_field_dual, dualcellquant_status
from membranequant.export import (
    rows_to_dataframe,
    save_label_mask,
    write_all_graphpad,
    write_multi_metric_summary,
    write_qc_log,
    write_results_csv,
    write_summary_csv,
)
from membranequant.io import FieldPair, scan_pairs
from membranequant.qc import apply_qc
from membranequant.utils import ensure_dir, setup_logging
from membranequant.visualization import draw_overlay, save_coloc_scatter

ProgressCallback = Callable[[str, float], None]


def process_field(
    pair: FieldPair,
    cfg: Config,
    out_dirs: dict[str, Path],
    logger,
) -> list[dict[str, Any]]:
    """Run DualCellQuant analysis on one Red/Green field pair."""
    logger.info("Processing %s (backend=dualcellquant)", pair.image_id)

    meta = {
        "image_id": pair.image_id,
        "field": pair.field,
        "group": pair.group,
        "experiment": pair.experiment,
        "drug": pair.drug,
        "condition_id": pair.condition_id,
    }
    result = analyze_field_dual(pair, cfg, meta=meta)
    rows = apply_qc(result.rows, cfg)

    safe_name = pair.image_id.replace("/", "_").replace("\\", "_")

    if cfg.save_mask:
        save_label_mask(result.labels, out_dirs["masks"] / f"{safe_name}_labels.tif")
        save_label_mask(result.membrane, out_dirs["masks"] / f"{safe_name}_membrane.tif")
        save_label_mask(result.cytoplasm, out_dirs["masks"] / f"{safe_name}_cytoplasm.tif")
        # Dual AND mask as 0/1
        save_label_mask(
            result.and_mask.astype(np.uint16),
            out_dirs["masks"] / f"{safe_name}_and_mask.tif",
        )

    if cfg.save_overlay:
        draw_overlay(
            result.green_vis,
            result.red_vis,
            result.labels,
            result.membrane,
            result.cytoplasm,
            out_dirs["overlays"] / f"{safe_name}_overlay.png",
            title=f"{pair.image_id} (DualCellQuant)",
        )
        save_coloc_scatter(
            result.green_vis,
            result.red_vis,
            result.labels,
            out_dirs["overlays"] / f"{safe_name}_coloc_scatter.png",
            title=f"{pair.image_id} EGFP vs DiI",
        )

    write_qc_log(
        result.rejected or [],
        rows,
        out_dirs["qc"] / f"{safe_name}_qc.txt",
    )

    n_pass = sum(1 for r in rows if r.get("QC") == "pass")
    n_cells = len(rows)
    logger.info(
        "  %s: %d cells from DualCellQuant, %d pass QC",
        pair.image_id,
        n_cells,
        n_pass,
    )
    return rows


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    cfg: Config,
    progress: ProgressCallback | None = None,
) -> Path:
    """Scan experiment folder, Dual-analyze all pairs, write aggregate CSVs/plots."""

    def _progress(msg: str, frac: float) -> None:
        if progress is not None:
            progress(msg, max(0.0, min(1.0, float(frac))))

    out_dirs = {
        "csv": ensure_dir(output_dir / "csv"),
        "masks": ensure_dir(output_dir / "masks"),
        "overlays": ensure_dir(output_dir / "overlays"),
        "qc": ensure_dir(output_dir / "qc"),
        "logs": ensure_dir(output_dir / "logs"),
    }
    logger = setup_logging(out_dirs["logs"] / "pipeline.log")
    logger.info("MembraneQuant start (DualCellQuant backend)")
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Config: %s", cfg.to_dict())
    st = dualcellquant_status()
    logger.info("DualCellQuant status: %s", st["message"])

    _progress("Scanning input folder…", 0.02)
    pairs = scan_pairs(input_dir)
    if not pairs:
        logger.warning("No Red/Green pairs found under %s", input_dir)
        empty = rows_to_dataframe([])
        write_results_csv(empty, out_dirs["csv"] / "results.csv")
        write_summary_csv(empty, out_dirs["csv"] / "summary.csv")
        _progress("No pairs found.", 1.0)
        return out_dirs["csv"] / "results.csv"

    logger.info("Found %d field pair(s)", len(pairs))
    all_rows: list[dict[str, Any]] = []
    n = len(pairs)
    for i, pair in enumerate(pairs):
        _progress(f"Processing {pair.image_id} ({i + 1}/{n})", 0.05 + 0.85 * (i / max(n, 1)))
        try:
            rows = process_field(pair, cfg, out_dirs, logger)
            all_rows.extend(rows)
        except Exception as exc:
            logger.error("Failed on %s: %s", pair.image_id, exc)
            logger.debug(traceback.format_exc())

    _progress("Writing CSV summaries…", 0.95)
    df = rows_to_dataframe(all_rows)
    results_path = out_dirs["csv"] / "results.csv"
    write_results_csv(df, results_path)

    summary_df = write_summary_csv(df, out_dirs["csv"] / "summary.csv", metric="Ratio_T_over_R")
    write_summary_csv(df, out_dirs["csv"] / "summary_RatioOfMeans.csv", metric="RatioOfMeans_T_R")
    write_summary_csv(
        df, out_dirs["csv"] / "summary_Enrichment.csv", metric="Enrichment_Membrane_vs_Whole"
    )
    write_multi_metric_summary(df, out_dirs["csv"] / "summary_all_metrics.csv")

    if cfg.save_graphpad:
        write_all_graphpad(df, out_dirs["csv"])

    _progress("Generating plots…", 0.97)
    try:
        from .plots import generate_all_plots

        generate_all_plots(df, summary_df, output_dir, metric="Ratio_T_over_R")
        logger.info("Statistical plots generated in plots/")
    except Exception as e:
        logger.warning("Failed to generate plots: %s", e)
        logger.debug(traceback.format_exc())

    n_pass = int((df["QC"] == "pass").sum()) if not df.empty and "QC" in df.columns else 0
    logger.info("Done. %d cell rows (%d pass QC). Results: %s", len(df), n_pass, results_path)
    _progress(f"Done. {n_pass} cells passed QC.", 1.0)
    return results_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="membranequant",
        description=(
            "Membrane localization quantification: DualCellQuant for image analysis, "
            "MembraneQuant for experiment I/O and result summaries."
        ),
    )
    p.add_argument("--webui", action="store_true", help="Launch the Gradio Web UI.")
    p.add_argument(
        "--input", "-i", type=Path, default=None,
        help="Experiment root folder (contains Group subfolders with TIFs).",
    )
    p.add_argument(
        "--output", "-o", type=Path, default=None,
        help="Output Results directory (default: <input>/Results).",
    )
    p.add_argument(
        "--config", "-c", type=Path, default=None,
        help="Path to config.yaml (default: package config.yaml).",
    )
    # DualCellQuant knobs
    p.add_argument("--bg-radius", type=int, default=None, help="Rolling-ball radius (px).")
    p.add_argument("--no-bg", action="store_true", help="Disable Dual background correction.")
    p.add_argument(
        "--radial-inner", type=float, default=None,
        help="Radial membrane inner %% (0=center).",
    )
    p.add_argument(
        "--radial-outer", type=float, default=None,
        help="Radial membrane outer %% (100=boundary).",
    )
    p.add_argument(
        "--ref-mask", type=str, default=None,
        choices=["none", "global_otsu", "global_percentile", "per_cell_otsu", "per_cell_percentile"],
        help="Dual reference (DiI) mask mode.",
    )
    p.add_argument(
        "--target-mask", type=str, default=None,
        choices=["none", "global_otsu", "global_percentile", "per_cell_otsu", "per_cell_percentile"],
        help="Dual target (EGFP) mask mode.",
    )
    p.add_argument("--diameter", type=float, default=None, help="Cellpose diameter (0=auto).")
    p.add_argument("--gpu", action="store_true", help="Use GPU for Dual Cellpose.")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay export.")
    p.add_argument("--no-mask", action="store_true", help="Disable mask export.")
    p.add_argument("--host", type=str, default="127.0.0.1", help="Web UI host.")
    p.add_argument("--port", type=int, default=7860, help="Web UI port.")
    p.add_argument("--share", action="store_true", help="Public Gradio share link.")
    # Legacy aliases (still accepted)
    p.add_argument("--cellpose-gpu", action="store_true", help="Alias for --gpu.")
    p.add_argument("--cellpose-diameter", type=float, default=None, help="Alias for --diameter.")
    p.add_argument("--ring-width", type=int, default=None, help="Ignored (legacy); use radial %%.")
    p.add_argument("--seg", type=str, default=None, help="Ignored (always DualCellQuant).")
    p.add_argument("--cellpose", action="store_true", help="Ignored (always DualCellQuant).")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.webui:
        from membranequant.webui import launch_webui

        launch_webui(host=args.host, port=args.port, share=args.share, config_path=args.config)
        return 0

    if args.input is None:
        print("Error: --input is required unless --webui is set.", file=sys.stderr)
        return 2

    overrides: dict[str, Any] = {}
    if args.bg_radius is not None:
        overrides["dual_bg_radius"] = args.bg_radius
    if args.no_bg:
        overrides["dual_bg_enable"] = False
    if args.radial_inner is not None:
        overrides["dual_radial_inner_pct"] = args.radial_inner
    if args.radial_outer is not None:
        overrides["dual_radial_outer_pct"] = args.radial_outer
    if args.ref_mask is not None:
        overrides["dual_ref_mask_mode"] = args.ref_mask
    if args.target_mask is not None:
        overrides["dual_target_mask_mode"] = args.target_mask
    if args.diameter is not None:
        overrides["dual_diameter"] = args.diameter
    if args.cellpose_diameter is not None:
        overrides["dual_diameter"] = args.cellpose_diameter
    if args.gpu or args.cellpose_gpu:
        overrides["dual_use_gpu"] = True
    if args.no_overlay:
        overrides["save_overlay"] = False
    if args.no_mask:
        overrides["save_mask"] = False

    cfg = load_config(args.config, overrides=overrides or None)

    st = dualcellquant_status()
    if not st["available"] or not st["cellpose_available"]:
        print(
            "Error: DualCellQuant + Cellpose are required for image analysis.\n"
            '  pip install "git+https://github.com/fuji3to4/DualCellQuant.git"\n'
            "  pip install cellpose torch\n"
            f"Status: {st['message']}",
            file=sys.stderr,
        )
        return 1

    input_dir = args.input.resolve()
    if args.output is not None:
        output_dir = args.output.resolve()
    else:
        output_dir = (input_dir / cfg.output_dir).resolve()

    if not input_dir.is_dir():
        print(f"Error: input directory not found: {input_dir}", file=sys.stderr)
        return 1

    run_pipeline(input_dir, output_dir, cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
