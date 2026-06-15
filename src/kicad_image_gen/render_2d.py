"""2D PCB editor-style image export via SVG → PNG conversion."""

from __future__ import annotations

import logging
import math
import os
import shutil
import subprocess
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

from kicad_image_gen.core import find_kicad_cli
from kicad_image_gen.ratsnest import (
    nearest_neighbor_ratsnest,
    parse_board_bounds,
    parse_footprint_bounds,
    parse_keepout_zones,
    parse_mounting_holes,
    parse_net_pad_map,
    parse_pad_labels,
    parse_tht_pads,
    parse_vias,
)

logger = logging.getLogger(__name__)

# Layer order matters: later layers render on top. F.Cu last so pads show as red.
_DEFAULT_LAYERS_TOP = "B.Cu,B.SilkS,B.Mask,B.Fab,F.Mask,F.Fab,Edge.Cuts,F.Cu,F.SilkS"
_DEFAULT_LAYERS_BOTTOM = "F.Cu,F.SilkS,F.Mask,F.Fab,B.Mask,B.Fab,B.CrtYd,Edge.Cuts,B.Cu,B.SilkS"
_DEFAULT_WIDTH = 4800

# Layer presets for convenience (ordered: back layers first, front on top)
LAYER_PRESETS: dict[str, str] = {
    "all": "B.Cu,B.SilkS,B.Mask,B.Fab,B.CrtYd,F.Mask,F.Fab,F.CrtYd,Edge.Cuts,F.Cu,F.SilkS",
    "top": "F.Mask,F.Fab,F.CrtYd,Edge.Cuts,F.Cu,F.SilkS",
    "bottom": "B.Mask,B.Fab,B.CrtYd,Edge.Cuts,B.Cu,B.SilkS",
    "copper": "B.Cu,Edge.Cuts,F.Cu",
    "silkscreen": "B.SilkS,Edge.Cuts,F.SilkS",
    "fab": "B.Fab,B.CrtYd,Edge.Cuts,F.Fab,F.CrtYd",
}


def render_2d(
    pcb_path: str | Path,
    output_path: str | Path,
    *,
    layers: str | None = None,
    width: int = _DEFAULT_WIDTH,
    dpi: int | None = None,
    theme: str | None = None,
    background: str | None = None,
    mirror: bool = False,
    black_and_white: bool = False,
    ratsnest: bool = True,
    pad_labels: bool = True,
    grid_dots: bool = False,
    keepout: bool = True,
    keep_svg: bool = False,
    fill_opacity: int | None = None,
    crop: str | None = None,
    padding_mm: float = 5.0,
) -> Path:
    """Export a 2D editor-style PNG image of a KiCad PCB.

    Pipeline: kicad-cli pcb export svg → inject overlays → convert to PNG.

    Args:
        pcb_path: Path to the ``.kicad_pcb`` file.
        output_path: Output PNG file path.
        layers: Comma-separated layer list, or a preset name (top, bottom, all,
                copper, silkscreen, fab). Defaults to all visible layers.
        width: Output image width in pixels.
        dpi: Pixels per mm of board dimension. When set, overrides width by
             computing it from board bounds.
        theme: KiCad color theme name.
        background: Not used for SVG export (SVG background is theme-controlled).
        mirror: Mirror the board (useful for bottom layer views).
        black_and_white: Render in black and white.
        ratsnest: Inject ratsnest lines showing signal-net connectivity (default True).
        pad_labels: Inject pad net-name labels at each pad location (default True).
        grid_dots: Inject subtle grid dot pattern in background (default True).
        keepout: Inject keepout zone overlays (default True).
        keep_svg: Save the intermediate SVG file alongside the PNG (default False).
        fill_opacity: Opacity for copper fills (0.0-1.0). None for no modification.
        crop: Reference designator to zoom into (e.g. "U1"). None for full board.
        padding_mm: Context padding in mm around crop target (default 5.0).

    Returns:
        Resolved output path.
    """
    pcb_path = Path(pcb_path)
    output_path = Path(output_path)
    if not pcb_path.is_file():
        msg = f"PCB file not found: {pcb_path}"
        raise FileNotFoundError(msg)

    # When dpi is specified, compute width from board bounds
    if dpi is not None:
        bx0, _by0, bx1, _by1 = parse_board_bounds(pcb_path)
        board_w_mm = bx1 - bx0
        width = int(board_w_mm * dpi)
        logger.info("DPI %d → width=%d (board %.1f mm wide)", dpi, width, board_w_mm)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Resolve layer preset
    resolved_layers = LAYER_PRESETS.get(layers or "", layers or _DEFAULT_LAYERS_TOP)

    # Step 1: Export SVG (with fill opacity handling if needed)
    if fill_opacity is not None:
        svg_path = _export_svg_with_fill_opacity(
            pcb_path,
            resolved_layers,
            fill_opacity,
            theme=theme,
            mirror=mirror,
            black_and_white=black_and_white,
        )
    else:
        svg_path = _export_svg(
            pcb_path,
            resolved_layers,
            theme=theme,
            mirror=mirror,
            black_and_white=black_and_white,
        )
    if svg_path is None:
        msg = "SVG export failed — ensure kicad-cli supports 'pcb export svg'"
        raise RuntimeError(msg)

    # Step 2: Inject overlays into SVG (always runs for background/grid)
    _inject_overlays(
        svg_path,
        pcb_path,
        ratsnest=ratsnest,
        pad_labels=pad_labels,
        grid_dots=grid_dots,
        keepout=keepout,
        crop=crop,
        padding_mm=padding_mm,
    )

    # Step 3: Convert SVG → PNG
    try:
        png_path = _convert_svg_to_png(svg_path, output_path, width)
    finally:
        if not keep_svg:
            _cleanup(svg_path)
        else:
            svg_output = output_path.with_suffix(".svg")
            shutil.copy2(svg_path, svg_output)
            logger.info("SVG saved: %s", svg_output)
            _cleanup(svg_path)

    if png_path is None:
        msg = (
            "No SVG→PNG converter found. Install one of: "
            "rsvg-convert (librsvg), cairosvg (pip), or use macOS sips."
        )
        raise RuntimeError(msg)

    logger.info("2D export (%s) → %s", resolved_layers, output_path)
    return png_path


def _export_svg(
    pcb_path: Path,
    layers: str,
    *,
    theme: str | None = None,
    mirror: bool = False,
    black_and_white: bool = False,
) -> Path | None:
    """Run kicad-cli pcb export svg, return path to temp SVG or None."""
    kicad_cli = find_kicad_cli()

    with tempfile.TemporaryDirectory(prefix="kicad_svg_") as tmpdir:
        svg_out = Path(tmpdir) / f"{pcb_path.stem}.svg"
        cmd = [
            kicad_cli,
            "pcb",
            "export",
            "svg",
            "--mode-single",
            "-l",
            layers,
            "--exclude-drawing-sheet",
            "--fit-page-to-board",
            "--drill-shape-opt",
            "0",
            "-o",
            str(svg_out),
        ]

        # Default to custom theme for editor-matching pad colors
        cmd.extend(["--theme", theme or "kicad_image_gen"])
        if mirror:
            cmd.append("--mirror")
        if black_and_white:
            cmd.append("--black-and-white")

        cmd.append(str(pcb_path))

        logger.debug("SVG export: %s", " ".join(cmd))
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=60,
            )
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("SVG export failed: %s", exc)
            return None

        if result.returncode != 0:
            logger.warning("SVG export exited %d: %s", result.returncode, result.stderr[:500])
            return None

        if not svg_out.is_file():
            return None

        # Persist outside tmpdir
        fd, tmp_path = tempfile.mkstemp(suffix=".svg", prefix="kicad_2d_")
        os.close(fd)
        shutil.copy2(svg_out, tmp_path)
        return Path(tmp_path)


def _separate_copper_layers(layer_list: str) -> tuple[list[str], list[str]]:
    """Separate copper layers from non-copper layers.
    
    Args:
        layer_list: Comma-separated layer names (e.g., "F.Cu,B.SilkS,B.Cu")
    
    Returns:
        Tuple of (copper_layers, non_copper_layers) as lists
    """
    layers = [l.strip() for l in layer_list.split(",")]
    copper_layers = [l for l in layers if "Cu" in l]
    non_copper_layers = [l for l in layers if "Cu" not in l]
    return copper_layers, non_copper_layers


def _export_svg_with_fill_opacity(
    pcb_path: Path,
    layers: str,
    fill_opacity: float,
    *,
    theme: str | None = None,
    mirror: bool = False,
    black_and_white: bool = False,
) -> Path | None:
    """Export SVG with copper layers handled separately for fill opacity.
    
    Exports non-copper layers together, then exports each copper layer
    separately, applies fill opacity, and merges them back together.
    
    Args:
        pcb_path: Path to .kicad_pcb file
        layers: Comma-separated layer list
        fill_opacity: Opacity value (0.0-1.0) to apply to copper fills
        theme: KiCad color theme name
        mirror: Mirror the board
        black_and_white: Black and white rendering
    
    Returns:
        Path to merged SVG or None on failure
    """
    copper_layers, non_copper_layers = _separate_copper_layers(layers)
    
    svg_paths: list[Path] = []
    
    # Export non-copper layers if any
    if non_copper_layers:
        non_copper_str = ",".join(non_copper_layers)
        svg_path = _export_svg(
            pcb_path,
            non_copper_str,
            theme=theme,
            mirror=mirror,
            black_and_white=black_and_white,
        )
        if svg_path:
            svg_paths.append(svg_path)
    
    # Export each copper layer separately with fill opacity applied
    for copper_layer in copper_layers:
        svg_path = _export_svg(
            pcb_path,
            copper_layer,
            theme=theme,
            mirror=mirror,
            black_and_white=black_and_white,
        )
        if svg_path:
            _apply_fill_opacity(svg_path, fill_opacity)
            svg_paths.append(svg_path)
    
    if not svg_paths:
        return None
    
    # If only one SVG, return it directly
    if len(svg_paths) == 1:
        return svg_paths[0]
    
    # Merge multiple SVGs
    merged_path = _merge_svgs(svg_paths)
    # Clean up individual SVGs (except the base one that was merged and is being returned)
    for svg_path in svg_paths[1:]:
        #print(f"Cleaning up intermediate SVG: {svg_path}")
        _cleanup(svg_path)
    return merged_path


def _apply_fill_opacity(svg_path: Path, fill_opacity: float) -> None:
    """Apply opacity to filled zones (last group) in a single-layer SVG.
    
    Args:
        svg_path: Path to the SVG file.
        fill_opacity: Opacity value (0.0-1.0).
    """
    if not (0.0 <= fill_opacity <= 1.0):
        logger.warning("fill_opacity out of range (0.0-1.0): %.3f — ignoring", fill_opacity)
        return
    
    ET.register_namespace("", _SVG_NS)
    try:
        tree = ET.parse(str(svg_path))
    except ET.ParseError:
        logger.warning("Failed to parse SVG for fill opacity adjustment")
        return
    
    root = tree.getroot()
    opacity_str = f"{fill_opacity:.3f}"
    modified = False
    
    # Find all groups in the SVG
    all_groups = root.findall(f"{{{_SVG_NS}}}g")
    if all_groups:
        # The last group contains the filled zones for this layer
        last_group = all_groups[-1]
        # check if the style attribute has a opacity property, and if so, modify it. Otherwise, set opacity on the group.
        style = last_group.get("style", "")
        #print(f"Original style: '{style}'")
        style_parts = [part.strip() for part in style.split(";") if part.strip()]
        for i, part in enumerate(style_parts):
            if part.startswith("opacity:"):
                style_parts[i] = f"opacity:{opacity_str}"
                break
        else:
            style_parts.append(f"opacity:{opacity_str}")
        
        new_style = "; ".join(style_parts)
        last_group.set("style", new_style)

        #print(f"Modified style: '{new_style}'")
        modified = True
    
    if modified:
        tree.write(str(svg_path), xml_declaration=True, encoding="unicode")
        logger.debug("Applied fill-opacity %.3f to layer fills", fill_opacity)


def _merge_svgs(svg_paths: list[Path]) -> Path:
    """Merge multiple SVG files into a single SVG.
    
    All content (groups and elements) from source SVGs is merged into the first SVG's root.
    
    Args:
        svg_paths: List of SVG file paths to merge
    
    Returns:
        Path to the merged SVG file (first input file is modified and returned)
    """
    if not svg_paths:
        msg = "No SVG paths provided for merging"
        raise ValueError(msg)
    
    ET.register_namespace("", _SVG_NS)
    
    # Parse the first SVG (this will be the base)
    base_path = svg_paths[0]
    base_tree = ET.parse(str(base_path))
    base_root = base_tree.getroot()
    
    # Merge all other SVGs into the base
    for svg_path in svg_paths[1:]:
        try:
            other_tree = ET.parse(str(svg_path))
            other_root = other_tree.getroot()
            
            # Copy all child elements from other SVG to base SVG
            for child in other_root:
                base_root.append(child)
            
            logger.debug("Merged %s into base SVG", svg_path)
        except ET.ParseError:
            logger.warning("Failed to parse SVG for merging: %s", svg_path)
            continue
    
    # Write the merged SVG back to the base path
    base_tree.write(str(base_path), xml_declaration=True, encoding="unicode")
    logger.info("Merged %d SVGs into %s", len(svg_paths), base_path)
    return base_path


def _convert_svg_to_png(
    svg_path: Path,
    output_path: Path,
    width: int,
) -> Path | None:
    """Convert SVG to PNG using the first available backend."""
    for converter in (_convert_rsvg, _convert_cairosvg, _convert_sips):
        result = converter(svg_path, output_path, width)
        if result is not None:
            return result
    return None


def _convert_rsvg(svg_path: Path, output_path: Path, width: int) -> Path | None:
    """Convert via rsvg-convert (librsvg)."""
    rsvg = shutil.which("rsvg-convert")
    if not rsvg:
        return None
    cmd = [rsvg, "-w", str(width), "-o", str(output_path), str(svg_path)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and output_path.is_file():
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def _convert_cairosvg(svg_path: Path, output_path: Path, width: int) -> Path | None:
    """Convert via cairosvg Python library."""
    try:
        import cairosvg  # type: ignore[import-not-found]
    except ImportError:
        return None
    try:
        cairosvg.svg2png(
            url=str(svg_path),
            write_to=str(output_path),
            output_width=width,
        )
        if output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
    except Exception as exc:
        logger.warning("cairosvg failed: %s", exc)
    return None


def _convert_sips(svg_path: Path, output_path: Path, width: int) -> Path | None:
    """Convert via macOS sips (last resort — limited SVG support)."""
    sips = shutil.which("sips")
    if not sips:
        return None
    # sips can convert some SVGs but has limited support.
    # It works better with PDF intermediary, but try direct first.
    cmd = [
        sips,
        "-s",
        "format",
        "png",
        "-z",
        str(width),
        str(width),
        str(svg_path),
        "--out",
        str(output_path),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        if result.returncode == 0 and output_path.is_file() and output_path.stat().st_size > 0:
            return output_path
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


# ---------------------------------------------------------------------------
# SVG overlay injection (ratsnest + pad labels)
# ---------------------------------------------------------------------------

_SVG_NS = "http://www.w3.org/2000/svg"

# KiCad editor dark navy background color
_BG_COLOR = "#001023"

# Board substrate fill (inside Edge.Cuts, behind all other layers)
# Slightly lighter than background to show board area, but not too gray
_BOARD_FILL_COLOR = "#0d1a2e"

# Mounting holes: bright cyan like KiCad editor
_MOUNTING_HOLE_COLOR = "#1ac4d2"

# THT drill holes: dark center dot
_DRILL_HOLE_COLOR = "#001023"

# Ratsnest: slightly more visible than KiCad's thin gray, but not overwhelming
# Ratsnest: visible but not overwhelming — slightly bolder than KiCad's faint lines
_RATSNEST_COLOR = "#6699bb"
_RATSNEST_OPACITY = "0.35"
_RATSNEST_STROKE_WIDTH = "0.12"

# Pad labels: pin number inside pad (cyan), net name offset (smaller, muted)
_PAD_NUM_COLOR = "#00cccc"
_PAD_NUM_SIZE_FACTOR = 0.55  # font size = min(pad_w, pad_h) * factor
_PAD_NUM_MIN_FONT = 0.25
_PAD_NUM_MAX_FONT = 1.5
_NET_NAME_SIZE_FACTOR = 0.30
_NET_NAME_MIN_FONT = 0.18
_NET_NAME_MAX_FONT = 0.60
_NET_NAME_COLOR = "#9988aa"

# Keepout zones: semi-transparent red with dashed outline
_KEEPOUT_FILL = "#ff2222"
_KEEPOUT_FILL_OPACITY = "0.18"
_KEEPOUT_STROKE = "#ff4444"
_KEEPOUT_STROKE_WIDTH = "0.1"
_KEEPOUT_STROKE_OPACITY = "0.6"

# Grid dots
_GRID_DOT_COLOR = "#1a2a3a"
_GRID_DOT_RADIUS = 0.04  # mm
_GRID_SPACING = 1.27  # mm (50mil grid, matching KiCad default)


def _draw_oval_tht(parent_group, pad, offset_x: float, offset_y: float) -> None:
    """Draw a through-hole oval (slot) as a capsule shape.
    
    An oval slot consists of two circles at the ends and connecting lines.
    For visualization, we render the DRILL size, not the full pad size.
    The offset between centers is (major_length - minor_diameter) / 2.
    """
    pad_minor = pad.size
    pad_major = pad.size_major
    pad_offset = (pad_major - pad_minor) / 2

    drill_minor = pad.drill
    drill_major = pad.drill_major
    drill_offset = (drill_major - drill_minor) / 2

    # Rotation in radians
    rot_rad = math.radians(pad.rotation)
    cos_rot = math.cos(rot_rad)
    sin_rot = math.sin(rot_rad)
    
    # Calculate positions of the two end circles for the drill
    pad_c1_x = pad.x - offset_x + pad_offset * cos_rot
    pad_c1_y = pad.y - offset_y + pad_offset * sin_rot
    pad_c2_x = pad.x - offset_x - pad_offset * cos_rot
    pad_c2_y = pad.y - offset_y - pad_offset * sin_rot

    drill_c1_x = pad.x - offset_x + drill_offset * cos_rot
    drill_c1_y = pad.y - offset_y + drill_offset * sin_rot
    drill_c2_x = pad.x - offset_x - drill_offset * cos_rot
    drill_c2_y = pad.y - offset_y - drill_offset * sin_rot

    line = ET.SubElement(parent_group, f"{{{_SVG_NS}}}line")
    line.set("x1", f"{pad_c1_x:.4f}")
    line.set("y1", f"{pad_c1_y:.4f}")
    line.set("x2", f"{pad_c2_x:.4f}")
    line.set("y2", f"{pad_c2_y:.4f}")
    line.set("stroke", "#b8860b")
    line.set("stroke-width", f"{pad_minor:.4f}")
    line.set("stroke-linecap", "round")


    line = ET.SubElement(parent_group, f"{{{_SVG_NS}}}line")
    line.set("x1", f"{drill_c1_x:.4f}")
    line.set("y1", f"{drill_c1_y:.4f}")
    line.set("x2", f"{drill_c2_x:.4f}")
    line.set("y2", f"{drill_c2_y:.4f}")
    line.set("stroke", _BG_COLOR)
    line.set("stroke-width", f"{drill_minor:.4f}")
    line.set("stroke-linecap", "round")


def _inject_overlays(
    svg_path: Path,
    pcb_path: Path,
    *,
    ratsnest: bool = True,
    pad_labels: bool = True,
    grid_dots: bool = True,
    keepout: bool = True,
    crop: str | None = None,
    padding_mm: float = 5.0,
) -> None:
    """Inject ratsnest lines, vias, and/or pad labels into the exported SVG."""
    ET.register_namespace("", _SVG_NS)
    try:
        tree = ET.parse(str(svg_path))
    except ET.ParseError:
        logger.warning("Failed to parse SVG for overlay injection")
        return

    root = tree.getroot()
    viewbox = root.get("viewBox")
    if not viewbox:
        logger.debug("SVG has no viewBox — skipping overlays")
        return

    modified = False

    # --- Expand viewBox for margin, then add background fill ---
    vb_parts = viewbox.split()
    
    # --- Calculate coordinate offset between SVG viewBox and board bounds ---
    # KiCad's SVG export may normalize the viewBox to (0,0) even though .kicad_pcb
    # coordinates have a different origin. We need to offset all overlays accordingly.
    offset_x = 0.0
    offset_y = 0.0
    try:
        bx0, by0, _bx1, _by1 = parse_board_bounds(pcb_path)
        vb_x_start, vb_y_start = float(vb_parts[0]), float(vb_parts[1])
        offset_x = bx0 - vb_x_start
        offset_y = by0 - vb_y_start
        logger.debug("Coordinate offset: (%.2f, %.2f)", offset_x, offset_y)
    except (ValueError, OSError, IndexError):
        logger.debug("Could not calculate coordinate offset — using (0, 0)")
    if len(vb_parts) == 4:
        vb_x, vb_y = float(vb_parts[0]), float(vb_parts[1])
        vb_w, vb_h = float(vb_parts[2]), float(vb_parts[3])
        # Add padding on each side
        pad_x = vb_w * 0.04
        pad_y = vb_h * 0.04
        new_x = vb_x - pad_x
        new_y = vb_y - pad_y
        new_w = vb_w + 2 * pad_x
        new_h = vb_h + 2 * pad_y
        root.set("viewBox", f"{new_x:.4f} {new_y:.4f} {new_w:.4f} {new_h:.4f}")

        # Background fill covering expanded area
        bg_rect = ET.Element(f"{{{_SVG_NS}}}rect")
        bg_rect.set("x", f"{new_x:.4f}")
        bg_rect.set("y", f"{new_y:.4f}")
        bg_rect.set("width", f"{new_w:.4f}")
        bg_rect.set("height", f"{new_h:.4f}")
        bg_rect.set("fill", _BG_COLOR)
        root.insert(0, bg_rect)

        # Board substrate fill (lighter than background, inside Edge.Cuts)
        try:
            bx0, by0, bx1, by1 = parse_board_bounds(pcb_path)
            board_rect = ET.SubElement(root, f"{{{_SVG_NS}}}rect")
            board_rect.set("x", f"{bx0:.4f}")
            board_rect.set("y", f"{by0:.4f}")
            board_rect.set("width", f"{bx1 - bx0:.4f}")
            board_rect.set("height", f"{by1 - by0:.4f}")
            board_rect.set("fill", _BOARD_FILL_COLOR)
            # Insert after background but before everything else
            # Move board rect right after bg_rect (index 1)
            root.remove(board_rect)
            root.insert(1, board_rect)
        except (ValueError, OSError):
            pass  # No Edge.Cuts found, skip board fill

        # Grid dot pattern (1mm spacing)
        if grid_dots:
            defs = ET.SubElement(root, f"{{{_SVG_NS}}}defs")
            pattern = ET.SubElement(defs, f"{{{_SVG_NS}}}pattern")
            pattern.set("id", "grid-dots")
            pattern.set("width", f"{_GRID_SPACING}")
            pattern.set("height", f"{_GRID_SPACING}")
            pattern.set("patternUnits", "userSpaceOnUse")
            dot = ET.SubElement(pattern, f"{{{_SVG_NS}}}circle")
            dot.set("cx", f"{_GRID_SPACING / 2:.4f}")
            dot.set("cy", f"{_GRID_SPACING / 2:.4f}")
            dot.set("r", f"{_GRID_DOT_RADIUS}")
            dot.set("fill", _GRID_DOT_COLOR)
            grid_rect = ET.SubElement(root, f"{{{_SVG_NS}}}rect")
            grid_rect.set("x", f"{new_x:.4f}")
            grid_rect.set("y", f"{new_y:.4f}")
            grid_rect.set("width", f"{new_w:.4f}")
            grid_rect.set("height", f"{new_h:.4f}")
            grid_rect.set("fill", "url(#grid-dots)")

        modified = True

    # --- Mounting holes (cyan filled circles, on top of everything) ---
    mounting_holes = parse_mounting_holes(pcb_path)
    if mounting_holes:
        mh_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
        mh_group.set("id", "mounting-holes")
        for hole in mounting_holes:
            circle = ET.SubElement(mh_group, f"{{{_SVG_NS}}}circle")
            circle.set("cx", f"{hole.x - offset_x:.4f}")
            circle.set("cy", f"{hole.y - offset_y:.4f}")
            circle.set("r", f"{hole.diameter / 2 * 1.25:.4f}")
            circle.set("fill", _MOUNTING_HOLE_COLOR)
        modified = True
        logger.info("Injected %d mounting hole circles", len(mounting_holes))

    # --- Vias (gold circles with dark drill hole) ---
    vias = parse_vias(pcb_path)
    if vias:
        via_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
        via_group.set("id", "vias")
        for v in vias:
            # Outer circle (copper color)
            outer = ET.SubElement(via_group, f"{{{_SVG_NS}}}circle")
            outer.set("cx", f"{v.x - offset_x:.4f}")
            outer.set("cy", f"{v.y - offset_y:.4f}")
            outer.set("r", f"{v.size / 2:.4f}")
            outer.set("fill", "#b8860b")  # dark goldenrod
            # Inner circle (drill hole)
            inner = ET.SubElement(via_group, f"{{{_SVG_NS}}}circle")
            inner.set("cx", f"{v.x - offset_x:.4f}")
            inner.set("cy", f"{v.y - offset_y:.4f}")
            inner.set("r", f"{v.drill / 2:.4f}")
            inner.set("fill", _BG_COLOR)
        modified = True
        logger.info("Injected %d via markers into SVG", len(vias))

    # --- THT drill holes (plated through holes from footprints) ---
    tht_pads = parse_tht_pads(pcb_path)
    if tht_pads:
        tht_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
        tht_group.set("id", "tht-pads")
        for pad in tht_pads:
            if pad.is_oval:
                # Draw oval/slot as a capsule shape
                _draw_oval_tht(tht_group, pad, offset_x, offset_y)
            else:
                # Draw circular pad
                outer = ET.SubElement(tht_group, f"{{{_SVG_NS}}}circle")
                outer.set("cx", f"{pad.x - offset_x:.4f}")
                outer.set("cy", f"{pad.y - offset_y:.4f}")
                outer.set("r", f"{pad.size / 2:.4f}")
                outer.set("fill", "#b8860b")  # dark goldenrod
                # Inner circle (drill hole)
                inner = ET.SubElement(tht_group, f"{{{_SVG_NS}}}circle")
                inner.set("cx", f"{pad.x - offset_x:.4f}")
                inner.set("cy", f"{pad.y - offset_y:.4f}")
                inner.set("r", f"{pad.drill / 2:.4f}")
                inner.set("fill", _BG_COLOR)
        modified = True
        logger.info("Injected %d through-hole pad markers into SVG", len(tht_pads))

    # --- Keepout zones (semi-transparent red polygons with dashed outline) ---
    if keepout:
        keepout_zones = parse_keepout_zones(pcb_path)
    else:
        keepout_zones = []
    if keepout_zones:
        keepout_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
        keepout_group.set("id", "keepout-zones")
        for zone in keepout_zones:
            pts_str = " ".join(f"{x - offset_x:.4f},{y - offset_y:.4f}" for x, y in zone.points)
            polygon = ET.SubElement(keepout_group, f"{{{_SVG_NS}}}polygon")
            polygon.set("points", pts_str)
            polygon.set("fill", _KEEPOUT_FILL)
            polygon.set("fill-opacity", _KEEPOUT_FILL_OPACITY)
            polygon.set("stroke", _KEEPOUT_STROKE)
            polygon.set("stroke-width", _KEEPOUT_STROKE_WIDTH)
            polygon.set("stroke-opacity", _KEEPOUT_STROKE_OPACITY)
            polygon.set("stroke-dasharray", "0.5,0.3")
        modified = True
        logger.info("Injected %d keepout zone overlays into SVG", len(keepout_zones))

    # --- Ratsnest lines (all nets, including power) ---
    if ratsnest:
        net_pads = parse_net_pad_map(pcb_path, include_power=True)
        if net_pads:
            ratsnest_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
            ratsnest_group.set("id", "ratsnest")
            ratsnest_group.set("opacity", _RATSNEST_OPACITY)

            line_count = 0
            for _net_name, pads in net_pads.items():
                if len(pads) < 2:
                    continue
                edges = nearest_neighbor_ratsnest(pads)
                for i, j in edges:
                    line = ET.SubElement(ratsnest_group, f"{{{_SVG_NS}}}line")
                    line.set("x1", f"{pads[i][0] - offset_x:.4f}")
                    line.set("y1", f"{pads[i][1] - offset_y:.4f}")
                    line.set("x2", f"{pads[j][0] - offset_x:.4f}")
                    line.set("y2", f"{pads[j][1] - offset_y:.4f}")
                    line.set("stroke", _RATSNEST_COLOR)
                    line.set("stroke-width", _RATSNEST_STROKE_WIDTH)
                    line_count += 1

            if line_count > 0:
                modified = True
                logger.info("Injected %d ratsnest lines into SVG", line_count)

    # --- Pad labels: pin number (sized to pad, centered) + net name (offset) ---
    if pad_labels:
        all_pads = parse_pad_labels(pcb_path)
        if all_pads:
            labels_group = ET.SubElement(root, f"{{{_SVG_NS}}}g")
            labels_group.set("id", "pad-labels")

            label_count = 0
            for pad in all_pads:
                pad_min = min(pad.pad_width, pad.pad_height)

                # Pin number — sized proportional to pad, centered
                if pad.pad_number:
                    num_font = max(
                        _PAD_NUM_MIN_FONT,
                        min(_PAD_NUM_MAX_FONT, pad_min * _PAD_NUM_SIZE_FACTOR),
                    )
                    # Vertical centering: shift down by ~0.35 * font_size
                    y_offset = num_font * 0.35
                    num_el = ET.SubElement(labels_group, f"{{{_SVG_NS}}}text")
                    num_el.set("x", f"{pad.x - offset_x:.4f}")
                    num_el.set("y", f"{pad.y + y_offset - offset_y:.4f}")
                    num_el.set("font-size", f"{num_font:.3f}")
                    num_el.set("font-family", "sans-serif")
                    num_el.set("font-weight", "bold")
                    num_el.set("fill", _PAD_NUM_COLOR)
                    num_el.set("text-anchor", "middle")
                    num_el.text = pad.pad_number
                    label_count += 1

                # Net name — offset below pad, smaller, muted
                if pad.net_name:
                    net_font = max(
                        _NET_NAME_MIN_FONT,
                        min(_NET_NAME_MAX_FONT, pad_min * _NET_NAME_SIZE_FACTOR),
                    )
                    net_y_offset = pad.pad_height / 2 + net_font * 1.2
                    net_el = ET.SubElement(labels_group, f"{{{_SVG_NS}}}text")
                    net_el.set("x", f"{pad.x - offset_x:.4f}")
                    net_el.set("y", f"{pad.y + net_y_offset - offset_y:.4f}")
                    net_el.set("font-size", f"{net_font:.3f}")
                    net_el.set("font-family", "sans-serif")
                    net_el.set("fill", _NET_NAME_COLOR)
                    net_el.set("text-anchor", "middle")
                    net_el.text = pad.net_name
                    label_count += 1

            if label_count > 0:
                modified = True
                logger.info("Injected %d pad labels into SVG", label_count)

    # --- Crop viewBox to target component ---
    if crop:
        try:
            cx0, cy0, cx1, cy1 = parse_footprint_bounds(pcb_path, crop)
            crop_x = cx0 - padding_mm
            crop_y = cy0 - padding_mm
            crop_w = (cx1 - cx0) + 2 * padding_mm
            crop_h = (cy1 - cy0) + 2 * padding_mm
            root.set("viewBox", f"{crop_x:.4f} {crop_y:.4f} {crop_w:.4f} {crop_h:.4f}")
            modified = True
            logger.info("Cropped SVG viewBox to %s with %.1fmm padding", crop, padding_mm)
        except ValueError:
            logger.warning("Could not find footprint '%s' for crop — rendering full board", crop)

    if modified:
        tree.write(str(svg_path), xml_declaration=True, encoding="unicode")


def _cleanup(path: Path) -> None:
    """Remove temp file, ignoring errors."""
    try:
        path.unlink(missing_ok=True)
    except OSError:
        pass
