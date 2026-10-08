# PanoPub — Project Summary and Improvement Plan

Reviewed on 8 October 2026 from the scripts, dataset, and training logs available in this repository. These figures describe local artifacts, not a new evaluation of the saved models or an audit of the team's Label Studio server.

## 1. Purpose and current scope

PanoPub aims to identify and eventually map outdoor advertising in Panoramax street-level imagery, reducing the work needed for a manual inventory. Shop signs are an intended use case, but the current dataset has no separate shop-sign class. The team must decide whether they belong to `billboard`, require a new class, or should be excluded.

The repository contains an OSM-based collection pipeline, assisted annotation tools, a manually reviewed YOLO dataset, and training artifacts. The available validation scores show an experimental baseline with substantial room for improvement. Geolocation of newly detected objects remains a roadmap item.

## 2. Pipeline and implementation

### OSM candidate locations

`0_build_geojson_from_osm_pbf.sh` extracts existing advertising nodes from a France OSM extract. These are mapped candidates, not detections produced by our model. It classifies surroundings by distance to OSM place nodes, applies equal quotas, and spreads selection across a 0.1° grid.

The checked-in `ads.geojson` contains **6,000 locations**:

| Area category | Locations | Share |
| --- | ---: | ---: |
| City | 1,500 | 25% |
| Town | 1,500 | 25% |
| Village | 1,500 | 25% |
| Countryside | 1,500 | 25% |

Types: 2,231 billboards, 1,704 boards, 1,419 poster boxes, 558 totems, and 88 columns. These quotas describe OSM candidates; they do not prove that the downloaded or annotated dataset retains the same geographic balance.

### Panoramax collection

`1_fetch_panoramax_images.py` searches with `place_position=longitude,latitude` and `place_distance=5-25` metres. It requests a field-of-view tolerance of 20°, ranks candidates by proximity to an ideal distance of 12 m, and checks the sign-to-camera bearing against OSM facing directions when available, with a 60° tolerance.

It selects at most one image per OSM sign and three per sequence, reuses collection history, downloads concurrently, and records metadata. These filters increase the likelihood of relevant views; they cannot guarantee visibility or freedom from occlusion. The 12 m value is a selection heuristic, not a measured distance for every detected object.

### Assisted annotation and review

`2_pre-label.py` uses OWLv2 with advertising prompts to propose class-0 boxes. `3_filter-roadsigns.py` classifies small candidate crops to remove likely road signs. `4_yolo-to-labelstudio.py` imports proposals as predictions for human correction in Label Studio.

The annotation workflow uses `billboard`, `graffiti`, and `delete`. The `delete` category is a curation instruction: `5_clean_yolo_export.py` removes the entire image/label pair containing it and compacts the remaining IDs. The current cleaned dataset declares **0 = billboard, 1 = graffiti**.

Integration points to verify:

- The import script names class 0 `ad_sign`, whereas the exported dataset uses `billboard`.
- The road-sign filter removes crops based on top-class confidence without explicitly checking the predicted category against a road-sign/non-road-sign policy. Confidence alone does not establish that an advertising crop is a road sign; measure the filter's effect on advertising recall.
- The current training script declares only `billboard`, despite graffiti labels in the dataset. Resolve this contract before the next training run.

## 3. Dataset audit

The team reports approximately 1,000 annotated images. The local training dataset contains **847 image/label pairs**. Without the full annotation/export history, we cannot reconcile that reported total with the retained subset or infer how many images were removed.

| Metric | Training | Validation | Total |
| --- | ---: | ---: | ---: |
| Images and matching label files | 677 | 170 | 847 |
| Billboard boxes | 3,042 | 743 | 3,785 |
| Graffiti boxes | 67 | 19 | 86 |
| All boxes | 3,109 | 762 | 3,871 |
| Empty label files | 14 | 3 | 17 |

The split is approximately **80% / 20%**. Billboard boxes account for **97.8%** of annotations and graffiti for **2.2%**. All checked rows have five YOLO fields and coordinates within [0, 1], and every image has a matching label file. This structural check does not establish annotation completeness, tightness, or semantic accuracy.

One byte-identical image appears in both splits under different filenames:

- Training: `my_dataset/images/train/66039459__5fe4b12d-d4f2-450f-91a3-ccd23b80e8fa.jpg`
- Validation: `my_dataset/images/val/ebce42f7__f475d968-e0a5-4ee2-82cd-3f26e165af3c.jpg`

This is confirmed data leakage. SHA-256 comparison found one shared image hash; it does not detect visually similar frames. Related sequences or sites could create further leakage and require metadata-based checks. No independent test split is present locally.

### Why object size deserves priority

JPEG dimensions and normalized labels were used to estimate box sizes after fitting each full image within 640 × 640, preserving aspect ratio and before augmentation. **3,413 of 3,871 boxes (88.2%) have at least one side below 16 pixels**: 2,741 in training and 672 in validation. The 16-pixel threshold is an audit convention, not a universal detectability limit.

For example, a 5760 × 2880 panorama becomes approximately 640 × 320 before padding. A 90-pixel-wide object becomes only 10 pixels wide. This supports investigating crops or perspective views; it does not prove that object size alone causes low recall.

Annotate at a resolution where objects remain visible, then preserve normalized coordinates through resizing. Matching annotation resolution to `imgsz` is unnecessary. Cropping or reprojection requires transforming and checking the labels. Small or partially visible objects should follow a shared annotation policy rather than being automatically discarded.

## 4. Training results

Each row uses the epoch with the highest logged **validation mAP50–95 within that run**. Precision, recall, and mAP50 come from that same epoch. These are CSV values, not fresh evaluations of saved checkpoints.

| Run | Model in args.yaml | imgsz | Batch / workers | Logged epochs | Selected epoch | Precision | Recall | mAP50 | mAP50–95 |
| --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| train | YOLOv8n | 640 | 16 / 8 | 50 | 28 | 36.47% | 17.26% | 16.08% | 7.42% |
| train-2 | YOLO26s | 640 | 16 / 8 | 50 | 41 | 48.13% | 22.37% | 22.13% | 10.11% |
| train-3 | YOLO26s | 960 | 4 / 2 | 30 | 30 | 47.38% | 29.32% | 27.49% | 14.22% |
| train-4 | YOLO26s | 960 | 4 / 2 | 50 | 40 | 48.26% | 31.24% | 30.93% | 15.43% |

Sources: each run's `args.yaml` and `results.csv` under `runs/detect/`. All four use `device=mps` and `cache=false`. A checkpoint named `models/best-Yolo8s.pt` exists, but its filename does not establish its architecture; the available YOLOv8 training configuration specifies `yolov8n.pt`. A separate YOLOv8s experiment is therefore unverified.

The strongest logged run is `train-4`. Compared with `train-2`, its mAP50–95 is **5.32 percentage points higher** and recall **8.87 points higher**. This is an observed association, not a controlled demonstration of the effect of resolution: batch size, workers, and potentially dataset versions differ. Historical `data.yaml` files and immutable dataset manifests are unavailable here; saved run paths point to Forty2Challenge.

At epoch 50, `train-4` reports precision **52.29%**, recall **30.39%**, mAP50 **30.29%**, and mAP50–95 **15.33%**. The last epoch and the best logged epoch must not be confused. Aggregate metrics do not establish separate billboard, graffiti, or shop-sign performance. Low recall remains a central limitation.

### Training time

| Run | Final cumulative CSV time | Median interval between logged epochs |
| --- | ---: | ---: |
| train | 29.0 min | 34.1 s |
| train-2 | 59.5 min | 62.4 s |
| train-3 | 38.7 min | 77.1 s |
| train-4 | 124.3 min | 149.1 s |

Intervals are differences between consecutive `time` values, excluding the first epoch. They include the work represented by that timer and are not pure GPU compute benchmarks. The logs support roughly 1.3 minutes per epoch for `train-3`, but 2.5 minutes for `train-4`. The reported earlier slowdown of over one hour per epoch is not documented in these files. Because `cache=false` is common to every saved run, its individual contribution to a speed-up cannot be inferred.

The current `6_train_yolo_mac.py` loads **YOLOv8n at 960**, rather than reproducing the YOLO26s experiments, and declares only one class. Its hardcoded completion message also assumes the output directory is `train`, although successive runs can use another directory.

## 5. Prioritized improvements

| Priority | Action | Evidence or reason | How to measure progress |
| --- | --- | --- | --- |
| 1 | Agree target categories; align import names, export IDs, and training names | Shop signs are undefined; two export classes conflict with one training class | Zero class mismatches; per-class counts and AP |
| 1 | Remove cross-split duplicates; split by sequence/site; reserve an independent test set | One confirmed shared image; related frames remain unchecked | Zero shared hashes and group IDs between splits; held-out scores |
| 1 | Write an annotation guide and double-review a sample | Consistency matters more than adding unverified labels | Missing-object rate, class disagreement, matched-box IoU between annotators |
| 2 | Compare full panoramas with overlapping crops or perspective views | 88.2% of boxes have a side below 16 px at 640 | Recall/AP by size; original-image detection scores after merging views; inference time |
| 2 | Retain verified negative images, especially road signs, windows, and empty streets | Only 17 empty labels; filtering may remove valid ads | False positives per negative image; recall before/after filtering |
| 2 | Add reviewed failure cases and underrepresented environments | Small objects and graffiti are audit concerns; actual errors need inspection | Per-class and per-condition recall; geographic coverage |
| 3 | Run controlled model/resolution experiments with saved configurations | Several factors change in current comparisons | mAP50–95, per-class recall, runtime, memory, variation across seeds |
| 3 | Choose a confidence threshold for the intended workflow | Training aggregates do not define the deployment trade-off | Precision–recall curve; recall at agreed precision; manual review workload |
| 3 | Validate geolocation separately | Existing OSM locations are not newly estimated coordinates | Median and 95th-percentile position error in metres; deduplicated object count |

Keep all derived views of one panorama in the same split. For crop/perspective experiments, compare detections in a common original-image coordinate system so that the evaluation task stays comparable.

Do not remove every image without advertising: verified negatives teach the model to reject distracting objects. An empty label is useful only after a reviewer checks that no target was missed. Collect more images according to errors, diversity, and annotation quality rather than volume alone.

## 6. Metrics and evidence still needed

Precision measures how many predicted detections are correct; recall measures how many annotated targets are recovered. AP summarizes a precision–recall curve, and mAP averages AP across evaluated classes. mAP50 uses an IoU matching threshold of 0.50; mAP50–95 averages over thresholds from 0.50 to 0.95. IoU measures overlap between predicted and reference boxes. These metrics are not image classification accuracy; falling training losses alone do not prove generalization. See [Ultralytics validation documentation](https://docs.ultralytics.com/modes/val/).

YOLO handles resizing during training; annotation images do not need to be pre-resized to the training input size. See [Ultralytics training documentation](https://docs.ultralytics.com/modes/train/).

The next evaluation should save per-class scores, false positives and missed objects, size/condition breakdowns, confidence thresholds, and inference latency on named hardware. Collection yield, annotation time saved, road-sign filtering quality, final geographic coverage, and geolocation accuracy remain unmeasured here.

For geolocation, preserve camera position, panorama orientation, sequence identity, and source metadata. A bounding box alone does not provide metric depth. Projection with a fixed 12 m assumption produces an estimate requiring independent validation; investigate multi-view geometry or additional depth information.

The original summary correctly identified the pipeline and promising improvements: shared annotation rules, diverse data, panorama splitting, and learning from failure cases. The evidence adds three immediate priorities: consistent class definitions, an independent evaluation split, and enough pixels for small objects. The logged results show progress while leaving substantial room to improve recall and localization.
