"""Ratsnest computation from .kicad_pcb files.

Parses S-expression PCB files directly to extract net-pad connectivity,
then computes minimum spanning trees for signal nets. The resulting edges
can be injected into SVG exports as visual ratsnest lines.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

# Power/ground nets excluded from ratsnest
POWER_NETS: frozenset[str] = frozenset(
    {
        "GND",
        "VIN",
        "VCC",
        "+3V3",
        "+5V",
        "+12V",
        "+24V",
        "+3.3V",
        "+1.8V",
        "VBUS",
        "VBAT",
        "EARTH",
        "GNDREF",
        "GNDA",
        "GNDD",
        "VDDA",
        "VDDD",
        "VDD",
        "AVCC",
        "DVCC",
    }
)

# Regex patterns for parsing .kicad_pcb S-expressions
_RE_FOOTPRINT_AT = re.compile(
    r'\(footprint\s+"[^"]*"\s*\n'
    r"(?:.*?\n)*?"
    r"\s*\(at\s+([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?\)",
    re.MULTILINE,
)

_RE_AT = re.compile(r"\(at\s+([-\d.]+)\s+([-\d.]+)(?:\s+([-\d.]+))?\)")
# Matches both KiCad 9 format: (net 1 "GND") and KiCad 10 format: (net "GND")
_RE_NET = re.compile(r'\(net\s+(?:\d+\s+)?"([^"]*?)"\)')


def _rotate_point(px: float, py: float, angle_deg: float) -> tuple[float, float]:
    """Rotate point around origin using KiCad's CW rotation convention.

    KiCad stores positive angles as clockwise rotation in screen space
    (Y-down coordinate system). The CW rotation matrix is:
        x' = x*cos + y*sin
        y' = -x*sin + y*cos
    """
    rad = math.radians(angle_deg)
    cos_a, sin_a = math.cos(rad), math.sin(rad)
    return px * cos_a + py * sin_a, -px * sin_a + py * cos_a


_RE_PAD_NUM = re.compile(r'\(pad\s+"([^"]*)"')
_RE_REF = re.compile(r'\(property\s+"Reference"\s+"([^"]*)"')


def parse_board_bounds(pcb_path: str | Path) -> tuple[float, float, float, float]:
    """Parse the board outline bounding box from Edge.Cuts layer.

    Scans top-level graphic primitives (gr_line, gr_rect, gr_arc) on the
    Edge.Cuts layer and returns the overall bounding box.

    Uses a block-based approach: finds each gr_* block, checks if it
    references Edge.Cuts, then extracts start/end/mid coordinates.

    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in mm.

    Raises:
        ValueError: If no Edge.Cuts geometry is found.
    """
    text = Path(pcb_path).read_text(encoding="utf-8")

    xs: list[float] = []
    ys: list[float] = []

    coord_re = re.compile(r"\((start|end|mid)\s+([-\d.]+)\s+([-\d.]+)\)")

    # First try top-level (gr_*), then fall back to footprint (fp_*) if needed
    pattern = r"\(gr_(?:line|rect|arc|poly)\b"
    matches = list(re.finditer(pattern, text))
    if not matches:
        # Fall back: also include fp_* (but note these coords may be relative)
        pattern = r"\((?:gr_|fp_)(?:line|rect|arc|poly)\b"
        matches = list(re.finditer(pattern, text))

    for m in matches:
        block_start = m.start()
        # Find the closing paren by counting nesting
        depth = 0
        block_end = block_start
        for i in range(block_start, min(block_start + 2000, len(text))):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    block_end = i + 1
                    break
        block = text[block_start:block_end]

        # Check if this block is on Edge.Cuts
        if '"Edge.Cuts"' not in block:
            continue

        # Extract all start/end/mid coordinates
        for cm in coord_re.finditer(block):
            xs.append(float(cm.group(2)))
            ys.append(float(cm.group(3)))

    if not xs:
        msg = f"No Edge.Cuts geometry found in {pcb_path}"
        raise ValueError(msg)

    return (min(xs), min(ys), max(xs), max(ys))


def _parse_footprint_blocks(
    pcb_path: str | Path,
) -> list[tuple[float, float, float, str, str]]:
    """Parse footprint blocks, yielding (fp_x, fp_y, fp_rot, block_text, refdes)."""
    text = Path(pcb_path).read_text(encoding="utf-8")
    fp_starts = [m.start() for m in re.finditer(r"^\s+\(footprint\s", text, re.MULTILINE)]
    results = []

    for i, start in enumerate(fp_starts):
        end = fp_starts[i + 1] if i + 1 < len(fp_starts) else len(text)
        block = text[start:end]

        at_match = _RE_AT.search(block[:500])
        if not at_match:
            continue

        fp_x = float(at_match.group(1))
        fp_y = float(at_match.group(2))
        fp_rot = float(at_match.group(3)) if at_match.group(3) else 0.0

        ref_match = _RE_REF.search(block[:2000])
        refdes = ref_match.group(1) if ref_match else ""

        results.append((fp_x, fp_y, fp_rot, block, refdes))
    return results


def parse_net_pad_map(
    pcb_path: str | Path,
    *,
    include_power: bool = False,
) -> dict[str, list[tuple[float, float]]]:
    """Parse a .kicad_pcb file and build net → absolute pad positions map.

    Args:
        pcb_path: Path to .kicad_pcb file.
        include_power: If True, include power/ground nets. Default False.
    """
    net_pads: dict[str, list[tuple[float, float]]] = {}

    for fp_x, fp_y, fp_rot, block, _refdes in _parse_footprint_blocks(pcb_path):
        pad_starts = [m.start() for m in re.finditer(r"\(pad\s", block)]
        for ps in pad_starts:
            pad_block = block[ps : ps + 500]

            net_match = _RE_NET.search(pad_block)
            if not net_match:
                continue
            net_name = net_match.group(1)
            if not net_name:
                continue
            if not include_power and (net_name.upper() in POWER_NETS or net_name in POWER_NETS):
                continue

            pad_at = _RE_AT.search(pad_block)
            if not pad_at:
                continue

            pad_x = float(pad_at.group(1))
            pad_y = float(pad_at.group(2))
            rx, ry = _rotate_point(pad_x, pad_y, fp_rot)
            abs_x, abs_y = fp_x + rx, fp_y + ry
            net_pads.setdefault(net_name, []).append((abs_x, abs_y))

    return net_pads


@dataclass(frozen=True)
class PadLabel:
    """A pad with its absolute position, net name, pad number, and component refdes."""

    x: float
    y: float
    net_name: str
    pad_number: str
    refdes: str
    pad_width: float = 1.0
    pad_height: float = 1.0

    @property
    def label(self) -> str:
        """Short label: net name (or pad number if no net)."""
        return self.net_name or self.pad_number


_RE_PAD_SIZE = re.compile(r"\(size\s+([-\d.]+)\s+([-\d.]+)\)")
_RE_PAD_DRILL = re.compile(r"\(drill\s+(?:oval\s+)?([\-\d.]+)(?:\s+([\-\d.]+))?")


@dataclass(frozen=True)
class MountingHole:
    """A mounting hole with absolute position and size."""

    x: float
    y: float
    diameter: float  # pad size in mm


@dataclass(frozen=True)
class THTpad:
    """A through-hole pad with position, size, and drill.
    
    For circular pads: size and drill are diameters.
    For oval pads: size and drill are (minor_diameter, major_length).
    """

    x: float
    y: float
    size: float  # pad diameter (or minor diameter for ovals) in mm
    drill: float  # drill diameter (or minor diameter for ovals) in mm
    is_oval: bool = False  # True if pad is oval shaped
    size_major: float = 0.0  # major length for ovals; for circles this equals size
    drill_major: float = 0.0  # major length for oval drill; for circles this equals drill
    rotation: float = 0.0  # pad rotation in degrees


def parse_mounting_holes(pcb_path: str | Path) -> list[MountingHole]:
    """Parse mounting hole (NPTH) pad positions from a .kicad_pcb file."""
    holes: list[MountingHole] = []
    for fp_x, fp_y, fp_rot, block, _refdes in _parse_footprint_blocks(pcb_path):
        for m in re.finditer(r"\(pad\s", block):
            pad_block = block[m.start() : m.start() + 500]
            if "np_thru_hole" not in pad_block:
                continue
            pad_at = _RE_AT.search(pad_block)
            if not pad_at:
                continue
            pad_x, pad_y = float(pad_at.group(1)), float(pad_at.group(2))
            rx, ry = _rotate_point(pad_x, pad_y, fp_rot)
            size_match = _RE_PAD_SIZE.search(pad_block)
            diameter = float(size_match.group(1)) if size_match else 3.0
            holes.append(MountingHole(x=fp_x + rx, y=fp_y + ry, diameter=diameter))
    return holes


def parse_tht_pads(pcb_path: str | Path) -> list[THTpad]:
    """Parse through-hole (plated) pad positions with size and drill from a .kicad_pcb file.
    
    Handles both circular and oval (slot) pads.
    """
    pads: list[THTpad] = []
    for fp_x, fp_y, fp_rot, block, _refdes in _parse_footprint_blocks(pcb_path):
        for m in re.finditer(r"\(pad\s", block):
            pad_block = block[m.start() : m.start() + 800]  # increased size to capture oval drill
            if "thru_hole" not in pad_block:# or "np_thru_hole" in pad_block:
                continue
            pad_at = _RE_AT.search(pad_block)
            if not pad_at:
                continue
            pad_x, pad_y = float(pad_at.group(1)), float(pad_at.group(2))
            pad_rot = float(pad_at.group(3)) if pad_at.group(3) else 0.0
            rx, ry = _rotate_point(pad_x, pad_y, fp_rot)
            
            # Check if this is an oval pad
            is_oval = "oval" in pad_block
            
            size_match = _RE_PAD_SIZE.search(pad_block)
            drill_match = _RE_PAD_DRILL.search(pad_block)
            
            # For size: if oval, extract both dimensions and sort so minor <= major
            if size_match:
                dim1 = float(size_match.group(1))
                dim2 = float(size_match.group(2))
                size_minor = min(dim1, dim2)
                size_major = max(dim1, dim2)
            else:
                size_minor = 0
                size_major = 0
            
            # For drill: if oval drill, extract both dimensions and sort so minor <= major
            if drill_match:
                drill1 = float(drill_match.group(1))
                drill2 = float(drill_match.group(2)) if drill_match.group(2) else drill1
                drill_minor = min(drill1, drill2)
                drill_major = max(drill1, drill2)
            else:
                drill_minor = 0
                drill_major = 0
            
            pads.append(
                THTpad(
                    x=fp_x + rx,
                    y=fp_y + ry,
                    size=size_minor,
                    drill=drill_minor,
                    is_oval=is_oval,
                    size_major=size_major,
                    drill_major=drill_major,
                    rotation=fp_rot + pad_rot,
                )
            )
    return pads


def parse_pad_labels(pcb_path: str | Path) -> list[PadLabel]:
    """Parse all pads from a .kicad_pcb file with positions and labels.

    Returns every pad (including power nets) with its absolute position,
    net name, pad number, and parent component reference designator.
    """
    labels: list[PadLabel] = []

    for fp_x, fp_y, fp_rot, block, refdes in _parse_footprint_blocks(pcb_path):
        pad_starts = [m.start() for m in re.finditer(r"\(pad\s", block)]
        for ps in pad_starts:
            pad_block = block[ps : ps + 500]

            pad_num_match = _RE_PAD_NUM.search(pad_block)
            pad_number = pad_num_match.group(1) if pad_num_match else ""

            net_match = _RE_NET.search(pad_block)
            net_name = net_match.group(1) if net_match else ""

            pad_at = _RE_AT.search(pad_block)
            if not pad_at:
                continue

            pad_x = float(pad_at.group(1))
            pad_y = float(pad_at.group(2))
            rx, ry = _rotate_point(pad_x, pad_y, fp_rot)

            size_match = _RE_PAD_SIZE.search(pad_block)
            pw = float(size_match.group(1)) if size_match else 1.0
            ph = float(size_match.group(2)) if size_match else 1.0

            labels.append(
                PadLabel(
                    x=fp_x + rx,
                    y=fp_y + ry,
                    net_name=net_name,
                    pad_number=pad_number,
                    refdes=refdes,
                    pad_width=pw,
                    pad_height=ph,
                )
            )

    return labels


@dataclass(frozen=True)
class KeepoutZone:
    """A keepout zone polygon with absolute coordinates."""

    points: tuple[tuple[float, float], ...]


def parse_keepout_zones(pcb_path: str | Path) -> list[KeepoutZone]:
    """Parse keepout zone polygons from a .kicad_pcb file.

    Finds all ``(zone ...)`` blocks containing a ``(keepout ...)`` child and
    extracts their polygon vertex coordinates.

    Returns:
        List of KeepoutZone objects, each with a tuple of (x, y) points.
    """
    text = Path(pcb_path).read_text(encoding="utf-8")
    zones: list[KeepoutZone] = []

    zone_starts = [m.start() for m in re.finditer(r"^\s+\(zone\b", text, re.MULTILINE)]
    for start in zone_starts:
        # Balance parentheses to find the full zone block
        depth = 0
        end = start
        for i in range(start, min(start + 5000, len(text))):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        block = text[start:end]

        if "(keepout" not in block:
            continue

        # Find (pts ...) using balanced-paren search (regex can't handle nested parens)
        pts_start = block.find("(pts")
        if pts_start < 0:
            continue
        pts_depth = 0
        pts_end = pts_start
        for pi in range(pts_start, len(block)):
            if block[pi] == "(":
                pts_depth += 1
            elif block[pi] == ")":
                pts_depth -= 1
                if pts_depth == 0:
                    pts_end = pi + 1
                    break
        pts_block = block[pts_start:pts_end]

        xy_matches = re.findall(r"\(xy\s+([-\d.]+)\s+([-\d.]+)\)", pts_block)
        if len(xy_matches) >= 3:
            points = tuple((float(x), float(y)) for x, y in xy_matches)
            zones.append(KeepoutZone(points=points))

    return zones


@dataclass(frozen=True)
class ViaInfo:
    """A via with absolute position and dimensions."""

    x: float
    y: float
    size: float  # pad diameter in mm
    drill: float  # drill diameter in mm


def parse_vias(pcb_path: str | Path) -> list[ViaInfo]:
    """Parse top-level via entries from a .kicad_pcb file.

    Parses ``(via (at X Y) (size S) (drill D) ...)`` elements that appear
    at the top level of the PCB file (not inside footprints).

    Returns:
        List of ViaInfo with absolute coordinates and dimensions.
    """
    text = Path(pcb_path).read_text(encoding="utf-8")
    vias: list[ViaInfo] = []

    # Match top-level (via ...) blocks — indented by one tab (top-level element)
    for m in re.finditer(r"^\t\(via\b", text, re.MULTILINE):
        # Extract the full block by balancing parens
        depth = 0
        block_end = m.start()
        for i in range(m.start(), min(m.start() + 1000, len(text))):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    block_end = i + 1
                    break
        block = text[m.start() : block_end]

        at_match = _RE_AT.search(block)
        if not at_match:
            continue

        size_match = re.search(r"\(size\s+([-\d.]+)\)", block)
        drill_match = re.search(r"\(drill\s+([-\d.]+)\)", block)
        if not size_match or not drill_match:
            continue

        vias.append(
            ViaInfo(
                x=float(at_match.group(1)),
                y=float(at_match.group(2)),
                size=float(size_match.group(1)),
                drill=float(drill_match.group(1)),
            )
        )

    return vias


def parse_footprint_bounds(
    pcb_path: str | Path,
    refdes: str,
) -> tuple[float, float, float, float]:
    """Parse the bounding box of a specific footprint by reference designator.

    Finds the footprint block with matching Reference property, computes
    the bounding box of all pads and graphic elements within it, and
    returns absolute coordinates.

    Args:
        pcb_path: Path to .kicad_pcb file.
        refdes: Reference designator (e.g. "U1", "C1").

    Returns:
        Tuple of (min_x, min_y, max_x, max_y) in mm (absolute coordinates).

    Raises:
        ValueError: If the footprint is not found.
    """
    for fp_x, fp_y, fp_rot, block, ref in _parse_footprint_blocks(pcb_path):
        if ref != refdes:
            continue

        xs: list[float] = []
        ys: list[float] = []

        # Collect pad positions
        for pm in re.finditer(r"\(pad\s", block):
            pad_block = block[pm.start() : pm.start() + 500]
            pad_at = _RE_AT.search(pad_block)
            if not pad_at:
                continue
            pad_x = float(pad_at.group(1))
            pad_y = float(pad_at.group(2))
            rx, ry = _rotate_point(pad_x, pad_y, fp_rot)

            # Also consider pad size for the bounding box
            size_match = _RE_PAD_SIZE.search(pad_block)
            pw = float(size_match.group(1)) / 2 if size_match else 0.5
            ph = float(size_match.group(2)) / 2 if size_match else 0.5

            abs_x, abs_y = fp_x + rx, fp_y + ry
            xs.extend([abs_x - pw, abs_x + pw])
            ys.extend([abs_y - ph, abs_y + ph])

        # Collect graphic element positions (fp_line, fp_rect, fp_arc, fp_poly)
        coord_re = re.compile(r"\((start|end|mid)\s+([-\d.]+)\s+([-\d.]+)\)")
        for gm in re.finditer(r"\(fp_(?:line|rect|arc|poly)\b", block):
            # Find closing paren
            depth = 0
            gblock_end = gm.start()
            for i in range(gm.start(), min(gm.start() + 2000, len(block))):
                if block[i] == "(":
                    depth += 1
                elif block[i] == ")":
                    depth -= 1
                    if depth == 0:
                        gblock_end = i + 1
                        break
            gblock = block[gm.start() : gblock_end]
            for cm in coord_re.finditer(gblock):
                gx = float(cm.group(2))
                gy = float(cm.group(3))
                rx, ry = _rotate_point(gx, gy, fp_rot)
                xs.append(fp_x + rx)
                ys.append(fp_y + ry)

        if not xs:
            # Fallback: use footprint position as a point
            return (fp_x - 1, fp_y - 1, fp_x + 1, fp_y + 1)

        return (min(xs), min(ys), max(xs), max(ys))

    msg = f"Footprint with reference '{refdes}' not found in {pcb_path}"
    raise ValueError(msg)


def nearest_neighbor_ratsnest(
    points: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Compute ratsnest edges matching KiCad's approach.

    Each pad connects to its nearest same-net pad. This produces the same
    line pattern as KiCad's ratsnest — direct shortest connections without
    the crossing artifacts that MST can create.
    """
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]

    edges: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i in range(n):
        best_j = -1
        best_dist = float("inf")
        for j in range(n):
            if i == j:
                continue
            dx = points[i][0] - points[j][0]
            dy = points[i][1] - points[j][1]
            dist = dx * dx + dy * dy
            if dist < best_dist:
                best_dist = dist
                best_j = j
        if best_j >= 0:
            edge = (min(i, best_j), max(i, best_j))
            if edge not in seen:
                seen.add(edge)
                edges.append(edge)
    return edges


def minimum_spanning_tree(
    points: list[tuple[float, float]],
) -> list[tuple[int, int]]:
    """Compute MST edges for 2D points via Prim's algorithm."""
    n = len(points)
    if n < 2:
        return []
    if n == 2:
        return [(0, 1)]

    in_tree = [False] * n
    min_cost = [float("inf")] * n
    min_edge: list[int] = [-1] * n
    edges: list[tuple[int, int]] = []

    in_tree[0] = True
    for j in range(1, n):
        dx = points[0][0] - points[j][0]
        dy = points[0][1] - points[j][1]
        min_cost[j] = dx * dx + dy * dy
        min_edge[j] = 0

    for _ in range(n - 1):
        best = -1
        best_cost = float("inf")
        for j in range(n):
            if not in_tree[j] and min_cost[j] < best_cost:
                best_cost = min_cost[j]
                best = j
        if best < 0:
            break
        in_tree[best] = True
        edges.append((min_edge[best], best))
        for j in range(n):
            if not in_tree[j]:
                dx = points[best][0] - points[j][0]
                dy = points[best][1] - points[j][1]
                dist2 = dx * dx + dy * dy
                if dist2 < min_cost[j]:
                    min_cost[j] = dist2
                    min_edge[j] = best

    return edges
