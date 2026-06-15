"""CLI entry point for kicad-image-gen."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from kicad_image_gen.render_2d import LAYER_PRESETS, render_2d
from kicad_image_gen.render_2d_composite import render_2d_composite
from kicad_image_gen.render_3d import _VIEW_PRESETS, render_3d


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="kicad-image-gen",
        description="Generate PNG screenshots of KiCad PCBs — 2D editor view and 3D renders.",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    # --- 2d subcommand ---
    p2d = sub.add_parser("2d", help="2D PCB editor-style screenshot (SVG→PNG)")
    p2d.add_argument("pcb", type=Path, help="Path to .kicad_pcb file")
    p2d.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    p2d.add_argument(
        "-l",
        "--layers",
        default=None,
        help=f"Layer list or preset ({', '.join(LAYER_PRESETS)})",
    )
    p2d.add_argument("-w", "--width", type=int, default=2400, help="Image width (default: 2400)")
    p2d.add_argument("--dpi", type=int, default=None, help="Pixels per mm of board width (overrides --width)")
    p2d.add_argument("--hires", action="store_true", help="Shortcut for --dpi 100")
    p2d.add_argument("-t", "--theme", default=None, help="KiCad color theme name")
    p2d.add_argument("-m", "--mirror", action="store_true", help="Mirror the board")
    p2d.add_argument("--bw", action="store_true", help="Black and white")
    p2d.add_argument(
        "--composite",
        action="store_true",
        help="Use composite mode (3D top-down base + PIL overlays)",
    )
    p2d.add_argument(
        "--pad-labels",
        action="store_true",
        help="Show pad labels (composite mode only)",
    )
    p2d.add_argument(
        "--no-ratsnest",
        action="store_true",
        help="Disable ratsnest overlay",
    )
    p2d.add_argument(
        "--no-keepout",
        action="store_true",
        help="Disable keepout zone overlays",
    )
    p2d.add_argument(
        "--keep-svg",
        action="store_true",
        help="Save the intermediate SVG file alongside the PNG",
    )
    p2d.add_argument(
        "--fill-opacity",
        type=float,
        default=None,
        help="Opacity for copper fills (0.0-1.0, default: no modification)",
    )
    p2d.add_argument(
        "--crop",
        default=None,
        help="Reference designator to zoom into (e.g. U1)",
    )
    p2d.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Padding in mm around crop target (default: 5.0)",
    )

    # --- 3d subcommand ---
    p3d = sub.add_parser("3d", help="3D viewer-style screenshot")
    p3d.add_argument("pcb", type=Path, help="Path to .kicad_pcb file")
    p3d.add_argument("-o", "--output", type=Path, default=None, help="Output PNG path")
    p3d.add_argument(
        "--view",
        default="top",
        help=f"View preset ({', '.join(_VIEW_PRESETS)}) or 'custom' (default: top)",
    )
    p3d.add_argument("-w", "--width", type=int, default=3200, help="Image width (default: 3200)")
    p3d.add_argument("--height", type=int, default=1800, help="Image height (default: 1800)")
    p3d.add_argument("--dpi", type=int, default=None, help="Pixels per mm of board dimension (overrides --width/--height)")
    p3d.add_argument("--hires", action="store_true", help="Shortcut for --dpi 100")
    p3d.add_argument("-q", "--quality", default="basic", choices=["basic", "high"])
    p3d.add_argument("--perspective", action="store_true", help="Perspective projection")
    p3d.add_argument("--floor", action="store_true", help="Enable floor/shadows")
    p3d.add_argument("--zoom", type=float, default=1.0)
    p3d.add_argument("--rotate", default=None, help="Custom rotation X,Y,Z degrees")
    p3d.add_argument("--pan", default=None, help="Camera pan X,Y,Z")
    p3d.add_argument("--pivot", default=None, help="Pivot point X,Y,Z cm")
    p3d.add_argument("--background", default="transparent", choices=["transparent", "opaque"])
    p3d.add_argument("--preset", default=None, help="Appearance preset name")
    p3d.add_argument("--pad-overlay", action="store_true", help="Draw pad position overlays on 3D render")
    p3d.add_argument(
        "--crop",
        default=None,
        help="Reference designator to zoom into (e.g. U1)",
    )
    p3d.add_argument(
        "--padding",
        type=float,
        default=5.0,
        help="Padding in mm around crop target (default: 5.0)",
    )

    # --- both subcommand ---
    pboth = sub.add_parser("both", help="Generate standard 2D + 3D image set")
    pboth.add_argument("pcb", type=Path, help="Path to .kicad_pcb file")
    pboth.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory (default: next to PCB file)",
    )
    pboth.add_argument("-w", "--width", type=int, default=2400, help="2D image width")
    pboth.add_argument("-q", "--quality", default="basic", choices=["basic", "high"])

    return parser


def _default_output(pcb: Path, suffix: str) -> Path:
    return pcb.parent / f"{pcb.stem}_{suffix}.png"


def _cmd_2d(args: argparse.Namespace) -> None:
    output = args.output or _default_output(args.pcb, "2d")
    dpi = args.dpi or (100 if args.hires else None)
    if args.composite:
        result = render_2d_composite(
            args.pcb,
            output,
            width=args.width,
            ratsnest=not args.no_ratsnest,
            pad_labels=args.pad_labels,
        )
        print(f"2D (composite): {result}")
    else:
        result = render_2d(
            args.pcb,
            output,
            layers=args.layers,
            width=args.width,
            dpi=dpi,
            theme=args.theme,
            mirror=args.mirror,
            black_and_white=args.bw,
            ratsnest=not args.no_ratsnest,
            keepout=not args.no_keepout,
            keep_svg=args.keep_svg,
            fill_opacity=args.fill_opacity,
            crop=args.crop,
            padding_mm=args.padding,
        )
        print(f"2D: {result}")


def _cmd_3d(args: argparse.Namespace) -> None:
    output = args.output or _default_output(args.pcb, f"3d_{args.view}")
    dpi = args.dpi or (100 if args.hires else None)
    result = render_3d(
        args.pcb,
        output,
        view=args.view,
        width=args.width,
        height=args.height,
        dpi=dpi,
        quality=args.quality,
        perspective=args.perspective,
        floor=args.floor,
        zoom=args.zoom,
        rotate=args.rotate,
        pan=args.pan,
        pivot=args.pivot,
        background=args.background,
        preset=args.preset,
        pad_overlay=args.pad_overlay,
        crop=args.crop,
        padding_mm=args.padding,
    )
    print(f"3D: {result}")


def _cmd_both(args: argparse.Namespace) -> None:
    out_dir = args.output_dir or args.pcb.parent
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = args.pcb.stem

    renders: list[tuple[str, Path]] = []

    # 2D renders (composite mode — 3D top-down base + overlays)
    out_top = out_dir / f"{stem}_2d_top.png"
    render_2d_composite(args.pcb, out_top, width=args.width, quality=args.quality)
    renders.append(("2D top (composite)", out_top))

    # Bottom view: standard SVG-based render (composite only supports top)
    out_bottom = out_dir / f"{stem}_2d_bottom.png"
    render_2d(args.pcb, out_bottom, layers="bottom", width=args.width, mirror=True)
    renders.append(("2D bottom", out_bottom))

    # 3D renders
    for view in ("top", "bottom", "iso"):
        out = out_dir / f"{stem}_3d_{view}.png"
        render_3d(args.pcb, out, view=view, quality=args.quality)
        renders.append((f"3D {view}", out))

    for label, path in renders:
        print(f"{label}: {path}")


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    if args.verbose:
        logging.basicConfig(level=logging.DEBUG, format="%(name)s %(levelname)s: %(message)s")
    else:
        logging.basicConfig(level=logging.INFO, format="%(message)s")

    handlers = {"2d": _cmd_2d, "3d": _cmd_3d, "both": _cmd_both}
    try:
        handlers[args.command](args)
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        sys.exit(130)
