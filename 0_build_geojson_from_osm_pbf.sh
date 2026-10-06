#!/usr/bin/env bash
#
# Build a balanced ads.geojson (advertising signs) from the Geofabrik France extract.
#
# Signs are classified as city / town / village / countryside from their distance to
# OSM `place=*` nodes, then sampled with equal quotas per class and spread evenly
# over the territory (one sign per grid cell, round-robin).
#
# Usage:   ./make_ads_geojson.sh
#          TOTAL=10000 CELL_DEG=0.05 ./make_ads_geojson.sh
# Needs:   osmium-tool (brew install osmium-tool), curl, and uv (or python3 + numpy + scipy)

set -euo pipefail

PBF_URL="${PBF_URL:-https://download.geofabrik.de/europe/france-latest.osm.pbf}"
PBF="${PBF:-france-latest.osm.pbf}"
OUT="${OUT:-ads.geojson}"
TOTAL="${TOTAL:-6000}"          # signs in the output (many will have no Panoramax picture)
CELL_DEG="${CELL_DEG:-0.1}"     # spatial grid size in degrees (~8-11 km)
SEED="${SEED:-42}"

# Advertising types visible from the street
AD_TYPES="billboard,board,totem,column,poster_box"

command -v osmium >/dev/null 2>&1 || { echo "osmium not found: brew install osmium-tool" >&2; exit 1; }

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

# 1. Download the extract (resumes if interrupted, skipped if already there)
if [ ! -f "$PBF" ]; then
  echo "==> Downloading $PBF_URL"
  curl -L -C - -o "$PBF" "$PBF_URL"
else
  echo "==> Using existing $PBF"
fi

# 2. One pass: keep advertising nodes AND place nodes (used to classify density)
echo "==> Filtering ads and places"
osmium tags-filter "$PBF" \
  "n/advertising=$AD_TYPES" \
  "n/place=city,town,village,hamlet" \
  -o "$TMP/filtered.osm.pbf" --overwrite

# 3. Convert to GeoJSON
echo "==> Exporting to GeoJSON"
osmium export "$TMP/filtered.osm.pbf" -o "$TMP/filtered.geojson" --overwrite

# 4. Classify and sample
echo "==> Classifying and sampling"
if command -v uv >/dev/null 2>&1; then
  PYRUN=(uv run --with numpy --with scipy python -)
else
  PYRUN=(python3 -)
fi

export TMP OUT TOTAL CELL_DEG SEED

"${PYRUN[@]}" <<'PY'
import json, math, os, random
from collections import Counter

import numpy as np
from scipy.spatial import cKDTree

TMP = os.environ["TMP"]
OUT = os.environ["OUT"]
TOTAL = int(os.environ["TOTAL"])
CELL = float(os.environ["CELL_DEG"])
rng = random.Random(int(os.environ["SEED"]))

# A sign is "city" if within 6 km of a place=city node, "town" if within 2.5 km of a
# place=town node, "village" if within 1 km of a village/hamlet, otherwise countryside.
# Checked in this order, first match wins.
RULES = [
    ("city",    {"city"},               6.0),
    ("town",    {"town"},               2.5),
    ("village", {"village", "hamlet"},  1.0),
]
CLASSES = [r[0] for r in RULES] + ["countryside"]


def xyz(lon, lat):
    """Lon/lat (degrees) to 3D coordinates in km, so Euclidean distance ~ real distance."""
    lon, lat = np.radians(lon), np.radians(lat)
    r = 6371.0
    return np.column_stack([r * np.cos(lat) * np.cos(lon),
                            r * np.cos(lat) * np.sin(lon),
                            r * np.sin(lat)])


with open(f"{TMP}/filtered.geojson") as f:
    features = json.load(f)["features"]

ads, places = [], []
for ft in features:
    geom = ft.get("geometry") or {}
    props = ft.get("properties") or {}
    if geom.get("type") != "Point":
        continue
    if "advertising" in props:
        if props.get("disused") == "yes" or props.get("abandoned") == "yes":
            continue
        ads.append(ft)
    elif "place" in props:
        places.append(ft)

print(f"  - {len(ads)} ads, {len(places)} place nodes")

# One KD-tree per rule
trees = {}
for name, kinds, _ in RULES:
    pts = [p["geometry"]["coordinates"] for p in places if p["properties"]["place"] in kinds]
    if pts:
        arr = np.array(pts)
        trees[name] = cKDTree(xyz(arr[:, 0], arr[:, 1]))

ad_xy = np.array([a["geometry"]["coordinates"][:2] for a in ads])
ad_xyz = xyz(ad_xy[:, 0], ad_xy[:, 1])
labels = np.full(len(ads), "countryside", dtype=object)
unassigned = np.ones(len(ads), dtype=bool)
for name, _, radius in RULES:
    if name not in trees:
        continue
    dist, _ = trees[name].query(ad_xyz)
    hit = unassigned & (dist <= radius)
    labels[hit] = name
    unassigned &= ~hit

groups = {c: [] for c in CLASSES}
for ft, lab in zip(ads, labels):
    ft["properties"]["area_class"] = lab
    groups[lab].append(ft)

avail = {c: len(groups[c]) for c in CLASSES}
print("  - available per class:", avail)

# Equal quotas; if a class is too small, redistribute its share to the others
take = {c: 0 for c in CLASSES}
remaining = TOTAL
active = [c for c in CLASSES if avail[c] > 0]
while remaining > 0 and active:
    share = max(1, remaining // len(active))
    for c in list(active):
        n = min(share, avail[c] - take[c], remaining)
        take[c] += n
        remaining -= n
        if take[c] >= avail[c]:
            active.remove(c)
        if remaining == 0:
            break


def spread(feats, n):
    """Pick n features, one per grid cell in turn, so no area dominates."""
    cells = {}
    for ft in feats:
        lon, lat = ft["geometry"]["coordinates"][:2]
        cells.setdefault((math.floor(lat / CELL), math.floor(lon / CELL)), []).append(ft)
    for lst in cells.values():
        rng.shuffle(lst)
    order = list(cells.values())
    rng.shuffle(order)
    out = []
    while len(out) < n and order:
        nxt = []
        for lst in order:
            out.append(lst.pop())
            if len(out) >= n:
                break
            if lst:
                nxt.append(lst)
        order = nxt
    return out


selected = []
for c in CLASSES:
    selected += spread(groups[c], take[c])
rng.shuffle(selected)

with open(OUT, "w") as f:
    json.dump({"type": "FeatureCollection", "features": selected}, f)

print(f"  - wrote {len(selected)} signs to {OUT}")
print("  - final mix:", dict(Counter(s["properties"]["area_class"] for s in selected)))
print("  - types:    ", dict(Counter(s["properties"]["advertising"] for s in selected)))
PY

echo "==> Done: $OUT"
