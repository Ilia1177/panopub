#!/usr/bin/env python3
"""Turn dataset/images + dataset/labels (YOLO) into a Label Studio import file.

Each task gets the OWLv2 boxes as a *prediction*, so they show up as pre-labels
that you correct and submit.
"""
import glob
import json
import os
from urllib.parse import quote

from PIL import Image

IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"
OUTPUT = "ls_tasks.json"
# Names (without .jpg) of images already imported into Label Studio, one per line.
# They are skipped, so importing the new file never creates duplicate tasks.
IMPORTED_LOG = "ls_imported.txt"

# Must match the labeling config: <Image name="image"/> and <RectangleLabels name="label"/>
FROM_NAME = "label"
TO_NAME = "image"
CLASS_NAMES = {0: "ad_sign"}
MODEL_VERSION = "owlv2"

# Paths are relative to LABEL_STUDIO_LOCAL_FILES_DOCUMENT_ROOT (your project folder)
URL_PREFIX = "/data/local-files/?d="

already = set(open(IMPORTED_LOG).read().split()) if os.path.exists(IMPORTED_LOG) else set()
new_names = []
tasks = []
for img_path in sorted(glob.glob(f"{IMAGES_DIR}/*.jpg")):
    name = os.path.splitext(os.path.basename(img_path))[0]
    if name in already:
        continue
    with Image.open(img_path) as im:
        w, h = im.size

    results = []
    label_path = f"{LABELS_DIR}/{name}.txt"
    if os.path.exists(label_path):
        for line in open(label_path).read().splitlines():
            parts = line.split()
            if len(parts) != 5:
                continue
            cls, xc, yc, bw, bh = int(parts[0]), *map(float, parts[1:])
            results.append({
                "from_name": FROM_NAME,
                "to_name": TO_NAME,
                "type": "rectanglelabels",
                "original_width": w,
                "original_height": h,
                "image_rotation": 0,
                "value": {
                    # Label Studio uses percentages and the top-left corner
                    "x": (xc - bw / 2) * 100,
                    "y": (yc - bh / 2) * 100,
                    "width": bw * 100,
                    "height": bh * 100,
                    "rotation": 0,
                    "rectanglelabels": [CLASS_NAMES.get(cls, "ad_sign")],
                },
            })

    task = {"data": {"image": URL_PREFIX + quote(f"{IMAGES_DIR}/{name}.jpg")}}
    if results:
        task["predictions"] = [{"model_version": MODEL_VERSION, "result": results}]
    tasks.append(task)
    new_names.append(name)

with open(OUTPUT, "w") as f:
    json.dump(tasks, f)

# Recorded now, so run the import in Label Studio right after this script
with open(IMPORTED_LOG, "a") as f:
    f.writelines(n + "\n" for n in new_names)

with_boxes = sum(1 for t in tasks if "predictions" in t)
print(f"Wrote {len(tasks)} tasks to {OUTPUT} ({with_boxes} with pre-labels)")
