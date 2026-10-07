#!/usr/bin/env python3
"""Second pass: remove boxes that are really road signs from the YOLO pre-labels.

Run it after the OWLv2 script. For each box in dataset/labels/*.txt it crops the
image, classifies the crop with Panoramax/classify_fr_road_signs, and deletes the
box when the classifier is confident it is a road sign.

The original OWLv2 labels are backed up once in dataset/labels_raw/ and are always
the input, so you can change the settings and rerun as often as you like.
"""
import glob
import os
import shutil

import torch
from PIL import Image
from ultralytics import YOLO

IMAGES_DIR = "dataset/images"
LABELS_DIR = "dataset/labels"
RAW_DIR = "dataset/labels_raw"
SKIP_LIST = "ls_imported.txt"   # images already in Label Studio are left untouched
CLASSIFIER = "models/roadsigns/best.pt"

ROAD_SIGN_CONF = 0.6     # classifier confidence above which a box is a road sign
CROP_MARGIN = 0.1        # extra context around the box, as a fraction of its size
MAX_SIGN_FRAC = 0.04     # boxes covering more of the image than this are never classified
                         # (set to 1.0 to classify every box)
BATCH = 64               # crops per classifier call
SAVE_CROPS = True        # save crops in check_crops/ to tune ROAD_SIGN_CONF by eye
CLF_DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"

clf = YOLO(CLASSIFIER)
os.makedirs(RAW_DIR, exist_ok=True)
if SAVE_CROPS:
    os.makedirs("check_crops/road_sign", exist_ok=True)
    os.makedirs("check_crops/ad", exist_ok=True)

skip = set(open(SKIP_LIST).read().split()) if os.path.exists(SKIP_LIST) else set()

# 1. Read the raw labels and cut a crop for every box that may be a road sign
labels = {}   # image name -> list of YOLO lines
items = []    # (image name, line index, crop)
for label_path in sorted(glob.glob(f"{LABELS_DIR}/*.txt")):
    name = os.path.splitext(os.path.basename(label_path))[0]
    if name in skip:
        continue
    raw_path = f"{RAW_DIR}/{name}.txt"
    if not os.path.exists(raw_path):
        shutil.copy(label_path, raw_path)   # one-time backup of the OWLv2 output
    lines = [l for l in open(raw_path).read().splitlines() if l.strip()]
    labels[name] = lines

    img_path = f"{IMAGES_DIR}/{name}.jpg"
    if not lines or not os.path.exists(img_path):
        continue
    img = Image.open(img_path).convert("RGB")
    w, h = img.size
    for i, line in enumerate(lines):
        _, xc, yc, bw, bh = line.split()[:5]
        xc, yc, bw, bh = float(xc), float(yc), float(bw), float(bh)
        if bw * bh > MAX_SIGN_FRAC:
            continue
        x0, x1 = (xc - bw / 2 * (1 + 2 * CROP_MARGIN)) * w, (xc + bw / 2 * (1 + 2 * CROP_MARGIN)) * w
        y0, y1 = (yc - bh / 2 * (1 + 2 * CROP_MARGIN)) * h, (yc + bh / 2 * (1 + 2 * CROP_MARGIN)) * h
        x0, y0, x1, y1 = int(max(0, x0)), int(max(0, y0)), int(min(w, x1)), int(min(h, y1))
        if x1 <= x0 or y1 <= y0:
            continue
        items.append((name, i, img.crop((x0, y0, x1, y1))))

print(f"{len(labels)} label files, {sum(len(v) for v in labels.values())} boxes, "
      f"{len(items)} to classify")

# 2. Classify the crops in batches
road_signs = set()
for start in range(0, len(items), BATCH):
    chunk = items[start:start + BATCH]
    preds = clf.predict([c for _, _, c in chunk], device=CLF_DEVICE, verbose=False)
    for (name, i, crop), pred in zip(chunk, preds):
        conf = pred.probs.top1conf.item()
        sign_class = pred.names[pred.probs.top1]
        is_road_sign = conf >= ROAD_SIGN_CONF
        if is_road_sign:
            road_signs.add((name, i))
        if SAVE_CROPS:
            folder = "road_sign" if is_road_sign else "ad"
            crop.save(f"check_crops/{folder}/{conf:.2f}_{sign_class}_{name}_{i}.jpg")
    print(f"  {min(start + BATCH, len(items))}/{len(items)} boxes classified")

# 3. Write the filtered labels
removed = 0
for name, lines in labels.items():
    kept = [l for i, l in enumerate(lines) if (name, i) not in road_signs]
    removed += len(lines) - len(kept)
    with open(f"{LABELS_DIR}/{name}.txt", "w") as f:
        f.write("\n".join(kept))

print(f"Done: {removed} road-sign boxes removed. Originals are in {RAW_DIR}/")
