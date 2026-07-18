"""MembraneQuant CLI entry point.

Recommended ways to start Web UI:

  # From repo root (D:\\杂物\\grok):
  python -m membranequant --webui --port 7860
  python -m membranequant.main --webui --port 7860

  # From inside this folder (D:\\杂物\\grok\\membranequant):
  python main.py --webui --port 7860
  run_webui.cmd

Do NOT run `python -m membranequant.main` from *inside* the membranequant
folder — that looks for a nested package and fails with ModuleNotFoundError.
"""

from __future__ import annotations

import argparse
import sys
import traceback
from collections.abc import Callable
from pathlib import Path
from typing import Any

# Always put repo root on sys.path so `python main.py` works inside this folder.
from pathlib import Path as _Path

_REPO = _Path(__file__).resolve().parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from membranequant.config import load_config, Config
from membranequant.export import (
    rows_to_dataframe,
    save_label_mask,
    write_all_graphpad,
    write_multi_metric_summary,
    write_qc_log,
    write_results_csv,
    write_summary_csv,
)
from membranequant.io import FieldPair, load_image, scan_pairs
from membranequant.measurements import measure_cells
from membranequant.preprocess import preprocess_pair
from membranequant.qc import apply_qc
from membranequant.segmentation import build_cell_masks, cellpose_status
from membranequant.utils import ensure_dir, setup_logging
from membranequant.visualization import draw_overlay, save_coloc_scatter

ProgressCallback = Callable[[str, float], None]


def process_field(
    pair: FieldPair,
    cfg: Config,
    out_dirs: dict[str, Path],
    logger,
) -> list[dict[str, Any]]:
    """Run full pipeline on one Red/Green field pair."""
    logger.info("Processing %s (seg=%s)", pair.image_id, cfg.segmentation_method)

    red = load_image(pair.red_path)
    green = load_image(pair.green_path)
    red_p, green_p = preprocess_pair(red, green, cfg)

    masks = build_cell_masks(green_p, red_p, cfg)
    meta = {
        "image_id": pair.image_id,
        "field": pair.field,
        "group": pair.group,
        "experiment": pair.experiment,
        "drug": pair.drug,
        "condition_id": pair.condition_id,
    }
    rows = measure_cells(green_p, red_p, masks, meta, cfg)
    rows = apply_qc(rows, cfg)

    safe_name = pair.image_id.replace("/", "_").replace("\\", "_")

    if cfg.save_mask:
        save_label_mask(masks.labels, out_dirs["masks"] / f"{safe_name}_labels.tif")
        save_label_mask(masks.membrane, out_dirs["masks"] / f"{safe_name}_membrane.tif")
        save_label_mask(masks.cytoplasm, out_dirs["masks"] / f"{safe_name}_cytoplasm.tif")

    if cfg.save_overlay:
        draw_overlay(
            green_p,
            red_p,
            masks.labels,
            masks.membrane,
            masks.cytoplasm,
            out_dirs["overlays"] / f"{safe_name}_overlay.png",
            title=pair.image_id,
        )
        save_coloc_scatter(
            green_p,
            red_p,
            masks.labels,
            out_dirs["overlays"] / f"{safe_name}_coloc_scatter.png",
            title=f"{pair.image_id} EGFP vs DiI",
        )

    write_qc_log(
        masks.rejected,
        rows,
        out_dirs["qc"] / f"{safe_name}_qc.txt",
    )

    n_pass = sum(1 for r in rows if r.get("QC") == "pass")
    logger.info(
        "  %s: %d cells kept after segmentation (%s), %d pass QC",
        pair.image_id,
        len(masks.kept_ids),
        masks.method,
        n_pass,
    )
    return rows


def run_pipeline(
    input_dir: Path,
    output_dir: Path,
    cfg: Config,
    progress: ProgressCallback | None = None,
) -> Path:
    """Scan experiment folder, process all pairs, write aggregate CSVs.

    progress(message, fraction) is optional; fraction is in [0, 1].
    """
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
    logger.info("MembraneQuant start")
    logger.info("Input:  %s", input_dir)
    logger.info("Output: %s", output_dir)
    logger.info("Config: %s", cfg.to_dict())
    if cfg.segmentation_method == "cellpose":
        logger.info("Cellpose status: %s", cellpose_status())

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
    # Primary summary uses DiI-guided M/C (recommended for membrane localization)
    summary_df = write_summary_csv(df, out_dirs["csv"] / "summary.csv", metric="M/C_DiI")
    write_summary_csv(df, out_dirs["csv"] / "summary_Manders_M1.csv", metric="Manders_M1")
    write_summary_csv(df, out_dirs["csv"] / "summary_MEI.csv", metric="MEI")
    write_multi_metric_summary(df, out_dirs["csv"] / "summary_all_metrics.csv")

    if cfg.save_graphpad:
        write_all_graphpad(df, out_dirs["csv"])

    # 生成统计图表（PPT 级 300 dpi PNG）
    _progress("Generating plots…", 0.97)
    try:
        from .plots import generate_all_plots
        generate_all_plots(df, summary_df, output_dir)
        logger.info("Statistical plots generated in plots/")
    except Exception as e:
        logger.warning("Failed to generate plots: %s", e)
        logger.debug(traceback.format_exc())

    n_pass = int((df["QC"] == "pass").sum()) if not df.empty else 0
    logger.info("Done. %d cell rows (%d pass QC). Results: %s", len(df), n_pass, results_path)
    _progress(f"Done. {n_pass} cells passed QC.", 1.0)
    return results_path


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="membranequant",
        description="Quantify membrane localization of EGFP-tagged proteins (DiI membrane QC).",
    )
    p.add_argument(
        "--webui",
        action="store_true",
        help="Launch the Gradio Web UI (Select Folder + Run).",
    )
    p.add_argument(
        "--input",
        "-i",
        type=Path,
        default=None,
        help="Experiment root folder (contains Group subfolders with TIFs).",
    )
    p.add_argument(
        "--output",
        "-o",
        type=Path,
        default=None,
        help="Output Results directory (default: <input>/Results or config.output_dir).",
    )
    p.add_argument(
        "--config",
        "-c",
        type=Path,
        default=None,
        help="Path to config.yaml (default: package config.yaml).",
    )
    p.add_argument("--ring-width", type=int, default=None, help="Membrane ring width in pixels.")
    p.add_argument(
        "--seg",
        choices=["otsu", "cellpose"],
        default=None,
        help="Whole-cell segmentation backend (default: config / otsu).",
    )
    p.add_argument(
        "--cellpose",
        action="store_true",
        help="Shortcut for --seg cellpose (requires: pip install cellpose).",
    )
    p.add_argument(
        "--cellpose-model",
        type=str,
        default=None,
        help="Cellpose model name (e.g. cyto2, cyto3, nuclei).",
    )
    p.add_argument(
        "--cellpose-diameter",
        type=float,
        default=None,
        help="Cellpose diameter in pixels (0 = auto).",
    )
    p.add_argument("--cellpose-gpu", action="store_true", help="Run Cellpose on GPU if available.")
    p.add_argument("--no-overlay", action="store_true", help="Disable overlay export.")
    p.add_argument("--no-mask", action="store_true", help="Disable mask export.")
    p.add_argument("--pearson", action="store_true", help="Compute Pearson on membrane pixels.")
    p.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Web UI host (with --webui). Default 127.0.0.1",
    )
    p.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Web UI port (with --webui). Default 7860",
    )
    p.add_argument(
        "--share",
        action="store_true",
        help="Create a public Gradio share link (with --webui).",
    )
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
    if args.ring_width is not None:
        overrides["ring_width"] = args.ring_width
    if args.cellpose:
        overrides["segmentation_method"] = "cellpose"
    if args.seg is not None:
        overrides["segmentation_method"] = args.seg
    if args.cellpose_model is not None:
        overrides["cellpose_model"] = args.cellpose_model
    if args.cellpose_diameter is not None:
        overrides["cellpose_diameter"] = args.cellpose_diameter
    if args.cellpose_gpu:
        overrides["cellpose_gpu"] = True
    if args.no_overlay:
        overrides["save_overlay"] = False
    if args.no_mask:
        overrides["save_mask"] = False
    if args.pearson:
        overrides["compute_pearson"] = True

    cfg = load_config(args.config, overrides=overrides or None)

    if cfg.segmentation_method == "cellpose" and not cellpose_status()["available"]:
        print(
            "Error: Cellpose is not installed. Install with: pip install cellpose\n"
            "Or use --seg otsu for the built-in pipeline.",
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
