#!/usr/bin/env python3
"""Fetch Panoramax pictures likely to show an advertising sign mapped in OSM."""
import csv
import json
import math
import os
import random
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

####################################################################
# Constants
#
PANORAMAX_API = "https://api.panoramax.xyz/api"
OSM_FEATURES = "./ads.geojson"
OUTPUT_FOLDER = "./training_pictures"
METADATA_CSV = "./metadata.csv"

WANTED_PICTURES = 100
# Only keep large, street-visible ad types
KEEP_TYPES = {"billboard", "board", "totem", "column", "poster_box"}

# Search settings
PLACE_DISTANCE = "5-25"        # metres between camera and sign
IDEAL_DISTANCE = 12            # prefer the candidate closest to this distance
FOV_TOLERANCE = 20             # degrees: sign must be near the centre of the view
CANDIDATES_PER_SIGN = 5        # ask for several pictures, then pick the best
MAX_ANGLE_FROM_FACING = 60     # if OSM has `direction`, camera must be in front
MAX_PER_SEQUENCE = 3           # diversity: cap pictures from one sequence

# Network settings
SEARCH_WORKERS = 4
DOWNLOAD_WORKERS = 4
BATCH_SIZE = 20
REQUEST_DELAY = 0.2            # seconds, per search worker
TIMEOUT = 30

CARDINALS = {
    "N": 0, "NNE": 22.5, "NE": 45, "ENE": 67.5, "E": 90, "ESE": 112.5,
    "SE": 135, "SSE": 157.5, "S": 180, "SSW": 202.5, "SW": 225,
    "WSW": 247.5, "W": 270, "WNW": 292.5, "NW": 315, "NNW": 337.5,
}

####################################################################
# HTTP session with retries
#
session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
session.mount("https://", HTTPAdapter(max_retries=retry, pool_maxsize=16))


####################################################################
# Geometry helpers
#
def haversine(lat1, lon1, lat2, lon2):
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def bearing(lat1, lon1, lat2, lon2):
    """Compass bearing from point 1 to point 2, in degrees."""
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dl = math.radians(lon2 - lon1)
    y = math.sin(dl) * math.cos(p2)
    x = math.cos(p1) * math.sin(p2) - math.sin(p1) * math.cos(p2) * math.cos(dl)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


def angle_diff(a, b):
    d = abs(a - b) % 360
    return min(d, 360 - d)


def parse_directions(value):
    """OSM `direction` can be '90', 'NE' or '90;270'. Returns a list of degrees."""
    if not value:
        return []
    result = []
    for part in str(value).split(";"):
        part = part.strip().upper()
        try:
            result.append(float(part) % 360)
        except ValueError:
            if part in CARDINALS:
                result.append(CARDINALS[part])
    return result


####################################################################
# Load signs from OSM
#
def load_signs():
    with open(OSM_FEATURES) as f:
        data = json.load(f)
    signs = []
    for ft in data["features"]:
        geom = ft.get("geometry") or {}
        props = ft.get("properties") or {}
        if geom.get("type") != "Point":
            continue
        if props.get("advertising") not in KEEP_TYPES:
            continue
        signs.append({
            "osm_id": ft.get("id") or props.get("@id"),
            "type": props.get("advertising"),
            "lon": geom["coordinates"][0],
            "lat": geom["coordinates"][1],
            "directions": parse_directions(props.get("direction")),
        })
    return signs


####################################################################
# Panoramax search
#
def find_candidates(sign):
    """Return candidate pictures for one sign, best first."""
    time.sleep(REQUEST_DELAY)
    params = {
        "place_position": f"{sign['lon']},{sign['lat']}",
        "place_distance": PLACE_DISTANCE,
        "place_fov_tolerance": FOV_TOLERANCE,
        "limit": CANDIDATES_PER_SIGN,
    }
    try:
        r = session.get(f"{PANORAMAX_API}/search", params=params, timeout=TIMEOUT)
        r.raise_for_status()
        features = r.json().get("features", [])
    except (requests.RequestException, ValueError) as e:
        print(f"  - Search failed @ {sign['lon']},{sign['lat']}: {e}")
        return []

    candidates = []
    for f in features:
        assets = f.get("assets", {})
        asset = assets.get("hd") or assets.get("sd")
        if not asset:
            continue
        cam_lon, cam_lat = f["geometry"]["coordinates"][:2]

        # The camera must be in front of the sign, when OSM tells us where it faces
        if sign["directions"]:
            b = bearing(sign["lat"], sign["lon"], cam_lat, cam_lon)
            if not any(angle_diff(b, d) <= MAX_ANGLE_FROM_FACING for d in sign["directions"]):
                continue

        dist = haversine(sign["lat"], sign["lon"], cam_lat, cam_lon)
        candidates.append((abs(dist - IDEAL_DISTANCE), f, asset["href"], sign))

    candidates.sort(key=lambda c: c[0])
    return candidates


####################################################################
# Download
#
def download(item):
    pic_id, url = item["id"], item["url"]
    dest = f"{OUTPUT_FOLDER}/{pic_id}.jpg"
    tmp = dest + ".part"
    try:
        with session.get(url, stream=True, timeout=TIMEOUT) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                shutil.copyfileobj(r.raw, f)
        os.replace(tmp, dest)  # a file is only visible once fully written
        return item
    except (requests.RequestException, OSError) as e:
        print(f"  - Download failed for {url}: {e}")
        if os.path.exists(tmp):
            os.remove(tmp)
        return None


####################################################################
# Main
#
def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Loading OSM features...")
    signs = load_signs()
    random.seed(42)
    random.shuffle(signs)  # spread over the whole area instead of one neighbourhood
    print(f"  - {len(signs)} usable signs")

    # Resume support: do not redo what is already downloaded
    seen_ids = {os.path.splitext(n)[0] for n in os.listdir(OUTPUT_FOLDER) if n.endswith(".jpg")}
    remaining = WANTED_PICTURES - len(seen_ids)
    print(f"  - {len(seen_ids)} pictures already present, {max(remaining, 0)} to fetch")
    if remaining <= 0:
        return

    print("Searching Panoramax...")
    selected = []
    per_sequence = Counter()
    with ThreadPoolExecutor(SEARCH_WORKERS) as ex:
        for i in range(0, len(signs), BATCH_SIZE):
            if len(selected) >= remaining:
                break
            for candidates in ex.map(find_candidates, signs[i:i + BATCH_SIZE]):
                for _, f, url, sign in candidates:
                    pic_id, seq = f["id"], f.get("collection")
                    if pic_id in seen_ids or per_sequence[seq] >= MAX_PER_SEQUENCE:
                        continue
                    seen_ids.add(pic_id)
                    per_sequence[seq] += 1
                    props = f.get("properties", {})
                    selected.append({
                        "id": pic_id,
                        "url": url,
                        "sequence": seq,
                        "osm_id": sign["osm_id"],
                        "ad_type": sign["type"],
                        "sign_lon": sign["lon"],
                        "sign_lat": sign["lat"],
                        "license": props.get("license", ""),
                        "authors": ";".join(p.get("name", "") for p in f.get("providers", [])),
                    })
                    break  # one picture per sign
                if len(selected) >= remaining:
                    break
            print(f"  - {len(selected)}/{remaining} matches after {min(i + BATCH_SIZE, len(signs))} signs")

    print("Downloading pictures...")
    with ThreadPoolExecutor(DOWNLOAD_WORKERS) as ex:
        done = [d for d in ex.map(download, selected) if d]

    if done:
        new_file = not os.path.exists(METADATA_CSV)
        with open(METADATA_CSV, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(done[0].keys()))
            if new_file:
                writer.writeheader()
            writer.writerows(done)

    print(f"Done! {len(done)} new pictures, metadata in {METADATA_CSV}")


if __name__ == "__main__":
    main()
