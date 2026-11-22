#!/usr/bin/env python3
"""Turn any ROS occupancy map into a simulator track.

The simulators drive F1TENTH Gym, which needs three files per track:

    <name>_map.yaml          resolution / origin / thresholds (ROS map_server format)
    <name>.png               occupancy image, white = free
    <name>_centerline.csv    x_m,y_m -- the loop, used for the start pose and
                             for counting laps

A SLAM map (slam_toolbox's map.pgm + map.yaml) or a map from another team's
repo has the first two, in its own conventions, and never the third. This
module writes all three: it decodes the image using its map_server YAML
before making Gym's binary image, and it finds the loop's centreline itself.
Unknown cells remain blocked; a gray pixel can be free under the map's own
thresholds and must not automatically become a wall. One exception keeps
SLAM's unexplored space out of the track: map_saver writes unknown as 205
(p = 0.196), which a YAML free_thresh above 0.196 (our rebuilt maps use 0.25)
decodes as free. Such cells only become Gym floor in small holes enclosed by
other free cells; larger or border-touching areas stay walls.

How the centreline is found. A driveable loop is a ring of free space around
an *island* -- the block of wall the car drives round. The centreline is the
set of points equally far from the island and from every other wall, which is
the boundary of the region where the island is the nearer of the two. Dead-end
side corridors drop out on their own (both their walls are "other walls"), and
an obstacle inside the corridor bends the line round it, which is what a car
would do. The island is the largest wall region that does not touch the
image border.

    python3 tools/f1tenth_sim/track_import.py ~/racerbot-ws/data/<car>/racerbot_auto/<time>/map.yaml \\
        --name hallway_2026_10_20 --source "SLAM map, run <ts>"

writes the triplet into tools/f1tenth_sim/maps/ and adds the track to
maps/catalog.yaml, after which every simulator tool accepts --track
hallway_2026_10_20. Pure numpy + OpenCV + Pillow + PyYAML; no gym, no ROS.
"""

from __future__ import annotations

import argparse
import math
import sys
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

MAPS_DIR = Path(__file__).resolve().parent / "maps"
CATALOG = MAPS_DIR / "catalog.yaml"
FREE = 255
WALL = 0


@dataclass
class MapSpec:
    resolution: float
    origin_x: float
    origin_y: float
    free: np.ndarray          # bool, row 0 = top of the image (largest y)
    occupancy: np.ndarray | None = None  # map_server values: -1 unknown, 0..100
    unexplored: np.ndarray | None = None  # free only via map_saver's unknown code


def track_free(spec: MapSpec) -> np.ndarray:
    """``spec.free`` without unexplored space: the cells a Gym track may use."""
    unexplored = getattr(spec, "unexplored", None)
    return spec.free if unexplored is None else spec.free & ~unexplored


# Enclosed wall blobs smaller than this are noise for centreline-finding.
SPECK_AREA_M2 = 0.01
# Nav2 map_saver's unknown pixel before applying negate.
SAVER_UNKNOWN_PIXEL = 205
# Unknown-coded holes up to this size, enclosed by free space, are free.
UNKNOWN_HOLE_AREA_M2 = 0.05


def _unexplored(coded: np.ndarray, free: np.ndarray, resolution: float) -> np.ndarray:
    """Cells of ``coded`` outside small holes enclosed by other free cells."""
    import cv2

    count, labels, stats, _ = cv2.connectedComponentsWithStats(coded.astype(np.uint8), connectivity=8)
    known_free = free & ~coded
    blocked = np.zeros_like(coded)
    rows, cols = coded.shape
    limit = math.floor(UNKNOWN_HOLE_AREA_M2 / resolution ** 2 + 1e-9)
    for label in range(1, count):
        x, y, w, h, area = stats[label]
        mask = labels == label
        if x == 0 or y == 0 or x + w == cols or y + h == rows or area > limit:
            blocked |= mask
            continue
        window = np.s_[y - 1:y + h + 1, x - 1:x + w + 1]
        local = mask[window].astype(np.uint8)
        ring = cv2.dilate(local, np.ones((3, 3), np.uint8)).astype(bool) & ~local.astype(bool)
        if not known_free[window][ring].all():
            blocked |= mask
    return blocked


def load_map(yaml_path: Path) -> MapSpec:
    """Decode Nav2 Jazzy map_server pixels, then binarize for Gym.

    Unknown stays blocked, as in pure_pursuit.occupancy_map.OccupancyMap.
    Jazzy includes threshold equality in the free/occupied classes.
    Scale/raw costs use that stack's live-grid occupied threshold (50).
    ``free`` is the map_server decoding. ``unexplored`` marks trinary
    map_saver-unknown-coded cells decoded free, except small enclosed holes
    (see _unexplored); track extraction excludes them.
    """
    from PIL import Image  # local: keeps the module importable without Pillow

    yaml_path = Path(yaml_path)
    spec = yaml.safe_load(yaml_path.read_text())
    with Image.open(yaml_path.parent / spec["image"]) as handle:
        image = np.asarray(handle.convert("L"), dtype=np.float64)
        alpha = np.asarray(handle.convert("RGBA"))[:, :, 3]
    if image.ndim != 2 or image.size == 0:
        raise ValueError(f"{yaml_path}: map image is empty or not 2-D")
    mode = spec.get("mode", "trinary")
    if mode not in ("trinary", "scale", "raw"):
        raise ValueError(f"{yaml_path}: unsupported map mode {mode!r}")
    negate = spec.get("negate", 0)
    if negate not in (0, 1):
        raise ValueError(f"{yaml_path}: negate must be 0 or 1")
    free_thresh = float(spec["free_thresh"])
    occupied_thresh = float(spec["occupied_thresh"])
    if not 0 <= free_thresh < occupied_thresh <= 1:
        raise ValueError(f"{yaml_path}: require 0 <= free_thresh < occupied_thresh <= 1")
    if mode == "raw":
        # Raw map_server pixels already encode 0..100; negate is not applied.
        occupancy = np.where(image <= 100, image, -1).astype(np.int8)
    else:
        p = image / 255.0 if negate else (255.0 - image) / 255.0
        occupancy = np.full(image.shape, -1, dtype=np.int8)
        occupancy[p >= occupied_thresh] = 100
        occupancy[p <= free_thresh] = 0
        if mode == "scale":
            middle = (p > free_thresh) & (p < occupied_thresh)
            occupancy[middle] = np.floor(
                100 * (p[middle] - free_thresh) / (occupied_thresh - free_thresh) + 0.5
            ).astype(np.int8)
        occupancy[alpha < 255] = -1
    free = (occupancy >= 0) & (occupancy < 50)
    resolution = float(spec["resolution"])
    unexplored = None
    if mode == "trinary":
        coded = free & (image == (255 - SAVER_UNKNOWN_PIXEL if negate else SAVER_UNKNOWN_PIXEL))
        if coded.any():
            unexplored = _unexplored(coded, free, resolution)
    origin = spec["origin"]
    return MapSpec(resolution, float(origin[0]), float(origin[1]), free, occupancy, unexplored)


def pixel_to_world(spec: MapSpec, rows, cols):
    height = spec.free.shape[0]
    x = spec.origin_x + (np.asarray(cols, dtype=np.float64) + 0.5) * spec.resolution
    y = spec.origin_y + (height - np.asarray(rows, dtype=np.float64) - 0.5) * spec.resolution
    return x, y


def world_to_pixel(spec: MapSpec, x, y):
    height = spec.free.shape[0]
    col = np.floor((np.asarray(x) - spec.origin_x) / spec.resolution).astype(int)
    row = height - 1 - np.floor((np.asarray(y) - spec.origin_y) / spec.resolution).astype(int)
    return row, col


def _free_regions(free: np.ndarray, seed_rc=None, reference_rc=None):
    """Free components ranked by reference votes, enclosure, then size."""
    import cv2

    count, labels = cv2.connectedComponents(free.astype(np.uint8), connectivity=4)
    if count < 2:
        raise ValueError("map has no free space")
    if seed_rc is not None:
        label = labels[seed_rc]
        if label == 0:
            raise ValueError(f"seed pixel {seed_rc} is not free space")
        yield labels == label
        return
    sizes = np.bincount(labels.ravel())
    votes = np.zeros(count, int) if reference_rc is None else np.bincount(
        labels[reference_rc], minlength=count)
    border = np.unique(np.concatenate((labels[0], labels[-1], labels[:, 0], labels[:, -1])))
    for label in sorted(range(1, count), key=lambda i: (-votes[i], i in border, -sizes[i], i)):
        yield labels == label


def driveable_region(free: np.ndarray, seed_rc=None, reference_rc=None) -> np.ndarray:
    """Prefer a free loop, or the component containing most reference points.

    Reference rows/columns must be inside the map. An explicit seed wins.
    A border loop never displaces a larger enclosed component.
    """
    largest, largest_size = None, 0
    enclosed, enclosed_size = None, 0
    best_reference, best_score = None, 0
    for region in _free_regions(free, seed_rc, reference_rc):
        size = int(region.sum())
        if size > largest_size:
            largest, largest_size = region, size
        if seed_rc is not None:
            if reference_rc is not None and not region[reference_rc].any():
                raise ValueError("seed component does not intersect the reference centreline")
            return region
        if reference_rc is not None:
            score = int(np.count_nonzero(region[reference_rc]))
            if score > best_score:
                best_reference, best_score = region, score
            continue
        border = region[0].any() or region[-1].any() or region[:, 0].any() or region[:, -1].any()
        if not border and size > enclosed_size:
            enclosed, enclosed_size = region, size
        elif border and enclosed_size > size:
            return enclosed
        try:
            find_island(region)
        except ValueError:
            continue
        return region
    if reference_rc is not None:
        if best_reference is None:
            raise ValueError("reference centreline does not intersect free space")
        return best_reference
    return largest


def drop_narrow_gaps(region: np.ndarray, resolution: float, half_width: float) -> np.ndarray:
    """``region`` with every free gap narrower than ``2*half_width`` closed.

    A morphological opening: free space the car's body cannot get into
    disappears, so SLAM speckle beside a wall merges into that wall instead of
    leaving slivers the centreline could thread. Only used to *find* the
    centreline -- the map the simulated car drives keeps every pixel.
    """
    import cv2

    radius = int(math.ceil(half_width / resolution))
    if radius < 1:
        return region
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.morphologyEx(region.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    region = opened.astype(bool)
    # Wall specks under SPECK_AREA_M2 inside free space (single-pixel SLAM
    # noise) would bend the line round themselves; fill them for this
    # purpose. Real obstacles are bigger (circularTrack's smallest box is
    # ~0.04 m^2) and keep bending it.
    walls = (~region).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(walls, connectivity=8)
    speck = SPECK_AREA_M2 / resolution ** 2
    for label in range(1, count):
        if stats[label, cv2.CC_STAT_AREA] < speck:
            region = region | (labels == label)
    return region


def find_island(region: np.ndarray) -> np.ndarray:
    """The largest wall region fully enclosed by ``region`` -- what the loop
    goes round. Raises if the free region is not a loop at all."""
    import cv2

    walls = (~region).astype(np.uint8)
    count, labels = cv2.connectedComponents(walls, connectivity=8)
    border = set(np.unique(np.concatenate(
        (labels[0], labels[-1], labels[:, 0], labels[:, -1]))))
    best, best_size = None, 0
    for label in range(1, count):
        if label in border:
            continue
        size = int(np.count_nonzero(labels == label))
        if size > best_size:
            best, best_size = label, size
    if best is None:
        raise ValueError("free space is not a loop: no wall region is enclosed by it")
    return labels == best


def centerline_pixels(region: np.ndarray, island: np.ndarray) -> np.ndarray:
    """(N, 2) float (row, col) points along the loop, in contour order."""
    import cv2

    others = ~region & ~island
    d_island = cv2.distanceTransform((~island).astype(np.uint8), cv2.DIST_L2, 5)
    d_other = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 5)
    nearer_island = (island | (region & (d_island < d_other))).astype(np.uint8)
    contours, _ = cv2.findContours(nearer_island, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not contours:
        raise ValueError("could not trace a centreline")
    contour = max(contours, key=len)[:, 0, :]          # (N, 2) as (col, row)
    return np.column_stack((contour[:, 1], contour[:, 0])).astype(np.float64)


def _smooth_closed(points: np.ndarray, window: int) -> np.ndarray:
    if window < 2:
        return points
    kernel = np.ones(window) / window
    pad = window
    wrapped = np.concatenate((points[-pad:], points, points[:pad]))
    out = np.column_stack([np.convolve(wrapped[:, k], kernel, mode="same")
                           for k in range(points.shape[1])])
    return out[pad:-pad]


def resample_closed(points: np.ndarray, spacing: float) -> np.ndarray:
    """Evenly spaced points along a closed polyline (last point != first)."""
    loop = np.vstack((points, points[:1]))
    seg = np.hypot(*np.diff(loop, axis=0).T)
    s = np.concatenate(([0.0], np.cumsum(seg)))
    total = s[-1]
    if total <= 0.0:
        raise ValueError("degenerate loop")
    n = max(8, int(round(total / spacing)))
    t = np.linspace(0.0, total, n, endpoint=False)
    return np.column_stack((np.interp(t, s, loop[:, 0]), np.interp(t, s, loop[:, 1])))


def signed_area(points: np.ndarray) -> float:
    x, y = points[:, 0], points[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))


def straightest_index(points: np.ndarray, window_m: float, spacing: float) -> int:
    """Index whose surrounding ``window_m`` of track turns least -- where a
    person would put the car down to start."""
    heading = np.unwrap(np.arctan2(*np.diff(np.vstack((points, points[:1])), axis=0).T[::-1]))
    turn = np.abs(np.diff(np.concatenate((heading, heading[:1] + (heading[-1] - heading[0])))))
    half = max(1, int(round(window_m / spacing / 2)))
    n = len(points)
    cost = np.array([turn[np.arange(i - half, i + half) % n].sum() for i in range(n)])
    return int(np.argmin(cost))


@dataclass
class ImportedTrack:
    centerline: np.ndarray      # (N, 2) world x, y, counter-clockwise, start at index 0
    widths: np.ndarray          # corridor width at each centreline point, m
    spec: MapSpec
    region: np.ndarray          # bool driveable free space


def _project_seed(region, narrowed, seed_rc, radius):
    """Nearest survivor in four-neighbour free-space steps, rejecting competing fronts."""
    import cv2

    count, labels = cv2.connectedComponents((narrowed & region).astype(np.uint8), connectivity=4)
    if count < 2:
        raise ValueError("seed component has no free space after narrowing")
    visited = np.zeros(region.shape, bool)
    visited[seed_rc] = True
    queue = deque([(seed_rc[0], seed_rc[1], 0)])
    height, width = region.shape
    nearest, nearest_label, reach = None, 0, None
    while queue:
        row, col, distance = queue.popleft()
        if reach is not None and distance > reach:
            break
        label = labels[row, col]
        if label:
            if nearest is None:
                nearest, nearest_label = (row, col), label
                reach = distance + radius + 1
            elif label != nearest_label:
                raise ValueError("seed lies on a pinch between connected regions; "
                                 "move it into the intended loop")
        for nr, nc in ((row - 1, col), (row, col - 1), (row, col + 1), (row + 1, col)):
            if (0 <= nr < height and 0 <= nc < width
                    and region[nr, nc] and not visited[nr, nc]):
                visited[nr, nc] = True
                queue.append((nr, nc, distance + 1))
    if nearest is None:
        raise ValueError("seed component has no reachable free space after narrowing")
    return nearest


def _race_loop(spec: MapSpec, seed_rc, car_half_width: float):
    last_error = None
    for region in _free_regions(track_free(spec), seed_rc):
        try:
            narrowed = drop_narrow_gaps(region, spec.resolution, car_half_width)
            narrowed_seed = seed_rc
            if seed_rc is not None and not narrowed[seed_rc]:
                narrowed_seed = _project_seed(
                    region, narrowed, seed_rc, int(math.ceil(car_half_width / spec.resolution)))
            for candidate in _free_regions(narrowed, narrowed_seed):
                try:
                    island = find_island(candidate)
                    pixels = centerline_pixels(candidate, island)
                except ValueError as exc:
                    last_error = exc
                    continue
                return region, pixels
        except ValueError as exc:
            last_error = exc
    source = "free space" if seed_rc is None else f"seed component at {seed_rc}"
    raise ValueError(
        f"{source} is not a loop: no usable race loop after narrowing: {last_error}"
    ) from last_error


def extract_track(spec: MapSpec, seed_xy=None, spacing: float = 0.10,
                  smooth_m: float = 0.5, centerline_from: MapSpec | None = None,
                  car_half_width: float = 0.155) -> ImportedTrack:
    """Centreline + corridor widths for the loop in ``spec``.

    ``centerline_from`` takes the centreline from a different map of the same
    place (e.g. the obstacle-free version of a map with obstacles added), so a
    lap is scored along the corridor rather than round the obstacles. The
    driveable region always comes from ``spec`` itself.
    """
    import cv2

    seed_rc = None
    if seed_xy is not None:
        r, c = world_to_pixel(spec, seed_xy[0], seed_xy[1])
        seed_rc = (int(r), int(c))
    line_spec = centerline_from or spec
    line_seed = None if seed_xy is None else tuple(int(v) for v in world_to_pixel(
        line_spec, seed_xy[0], seed_xy[1]))
    line_region, pix = _race_loop(line_spec, line_seed, car_half_width)
    x, y = pixel_to_world(line_spec, pix[:, 0], pix[:, 1])
    points = np.column_stack((x, y))
    points = resample_closed(points, line_spec.resolution)
    window = max(1, int(round(smooth_m / line_spec.resolution)))
    points = _smooth_closed(points, window)
    points = resample_closed(points, spacing)
    if signed_area(points) < 0.0:
        points = points[::-1]
    start = straightest_index(points, 2.0, spacing)
    points = np.roll(points, -start, axis=0)

    # Corridor width at each centreline point, from the map the car drives.
    rows, cols = world_to_pixel(spec, points[:, 0], points[:, 1])
    inside = (rows >= 0) & (rows < spec.free.shape[0]) & (cols >= 0) & (cols < spec.free.shape[1])
    region = line_region if centerline_from is None else driveable_region(
        track_free(spec), seed_rc, reference_rc=(rows[inside], cols[inside]))
    d_wall = cv2.distanceTransform(region.astype(np.uint8), cv2.DIST_L2, 5) * spec.resolution
    widths = np.zeros(len(points))
    widths[inside] = 2.0 * d_wall[rows[inside], cols[inside]]
    return ImportedTrack(points, widths, spec, region)


def write_gym_track(track: ImportedTrack, directory: Path, name: str) -> Path:
    """Write the gym triplet; return the path gym wants (``<dir>/<name>.yaml``,
    whose stem it uses to find ``<name>_map.yaml`` -- the file itself is never
    read, so it is not created, same as racerbot_sim.tracks.write_track)."""
    from PIL import Image

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    image = np.where(track.region, FREE, WALL).astype(np.uint8)
    Image.fromarray(image, mode="L").save(directory / f"{name}.png")
    spec = track.spec
    (directory / f"{name}_map.yaml").write_text(
        f"image: {name}.png\n"
        f"resolution: {spec.resolution}\n"
        f"origin: [{spec.origin_x}, {spec.origin_y}, 0.0]\n"
        "negate: 0\n"
        "occupied_thresh: 0.65\n"
        "free_thresh: 0.196\n")
    lines = ["x_m,y_m"] + [f"{x:.4f},{y:.4f}" for x, y in track.centerline]
    (directory / f"{name}_centerline.csv").write_text("\n".join(lines) + "\n")
    return directory / f"{name}.yaml"


def track_summary(track: ImportedTrack) -> dict:
    loop = np.vstack((track.centerline, track.centerline[:1]))
    w = track.widths[track.widths > 0]
    return {
        "lap_length_m": round(float(np.sum(np.hypot(*np.diff(loop, axis=0).T))), 2),
        "width_min_m": round(float(w.min()), 2) if w.size else None,
        "width_p10_m": round(float(np.percentile(w, 10)), 2) if w.size else None,
        "width_median_m": round(float(np.median(w)), 2) if w.size else None,
    }


def load_catalog(path: Path = CATALOG) -> dict:
    if not Path(path).exists():
        return {}
    return yaml.safe_load(Path(path).read_text()) or {}


def save_catalog(catalog: dict, path: Path = CATALOG) -> None:
    header = ("# Simulator tracks imported from occupancy maps by track_import.py.\n"
              "# One entry per track; files live beside this one. Regenerate an\n"
              "# entry by re-running the import command recorded in it.\n")
    Path(path).write_text(header + yaml.safe_dump(catalog, sort_keys=True))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("map_yaml", type=Path, help="map_server yaml of the map to import")
    ap.add_argument("--name", required=True, help="track name, as passed to --track later")
    ap.add_argument("--source", required=True, help="where the map came from (recorded in the catalog)")
    ap.add_argument("--seed", type=float, nargs=2, metavar=("X", "Y"),
                    help="a world point inside the loop's corridor, if the map has several free regions")
    ap.add_argument("--centerline-from", type=Path,
                    help="take the centreline from this other map of the same place (e.g. without obstacles)")
    ap.add_argument("--out", type=Path, default=MAPS_DIR)
    ap.add_argument("--plot", type=Path, help="also write a PNG of the map and centreline")
    args = ap.parse_args(argv)

    spec = load_map(args.map_yaml)
    other = load_map(args.centerline_from) if args.centerline_from else None
    track = extract_track(spec, args.seed, centerline_from=other)
    write_gym_track(track, args.out, args.name)
    summary = track_summary(track)
    if args.out.resolve() == MAPS_DIR.resolve():
        catalog = load_catalog()
        command = ["python3 tools/f1tenth_sim/track_import.py", str(args.map_yaml),
                   "--name", args.name, "--source", repr(args.source)]
        if args.seed:
            command += ["--seed", *map(str, args.seed)]
        if args.centerline_from:
            command += ["--centerline-from", str(args.centerline_from)]
        catalog[args.name] = {"source": args.source, "import_command": " ".join(command), **summary}
        save_catalog(catalog)
    if args.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        h, w = spec.free.shape
        extent = [spec.origin_x, spec.origin_x + w * spec.resolution,
                  spec.origin_y, spec.origin_y + h * spec.resolution]
        plt.figure(figsize=(10, 10 * h / max(w, 1)))
        plt.imshow(track.region, cmap="gray", extent=extent)
        plt.plot(track.centerline[:, 0], track.centerline[:, 1], "r-", lw=1)
        plt.plot(*track.centerline[0], "go")
        plt.title(f"{args.name}: {summary}")
        plt.savefig(args.plot, dpi=80)
    print(f"{args.name}: {summary}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
