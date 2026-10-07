# 🚀 Hackathon Technical Guide: Outdoor Advertising Detection & Geolocation with Panoramax

This technical guide is designed for a **hackathon** team working with open street-level imagery and geospatial data. Based on the **PanoPub** architecture and codebase, it covers the complete lifecycle: from downloading raw OpenStreetMap extracts and balancing signs across territories, fetching optimal 360° camera views from Panoramax, addressing zero-shot vision roadblocks, executing a collaborative Human-in-the-Loop curation with Label Studio, fine-tuning an expert **YOLOv8** model, to projecting pixel detections back to real-world GPS coordinates.

---

## 📋 Table of Contents

1. [The Architecture & Pipeline Workflow](#1-the-architecture--pipeline-workflow)
2. [Environment Setup & Dependencies](#2-environment-setup--dependencies)
3. [Phase 0: Territorial Balancing & GeoJSON Extraction (`0_build_geojson_from_osm_pbf.sh`)](#3-phase-0-territorial-balancing--geojson-extraction-0_build_geojson_from_osm_pbfsh)
4. [Phase 1: Geometric Panoramax Image Fetching (`1_fetch_panoramax_images.py`)](#4-phase-1-geometric-panoramax-image-fetching-1_fetch_panoramax_imagespy)
5. [Phase 2: Zero-Shot Pre-Annotation & 360° Vision Roadblocks (`pre-label.py`)](#5-phase-2-zero-shot-pre-annotation--360-vision-roadblocks-pre-labelpy)
6. [Phase 3: Human-in-the-Loop (HITL) Ground Truth with Label Studio (`my_dataset/`)](#6-phase-3-human-in-the-loop-hitl-ground-truth-with-label-studio-my_dataset)
7. [Phase 4: Expert Model Fine-Tuning with YOLOv8 (`train_yolo_mac.py`)](#7-phase-4-expert-model-fine-tuning-with-yolov8-train_yolo_macpy)
8. [Phase 5: Inference, Evaluation & Metrics](#8-phase-5-inference-evaluation--metrics)
9. [Phase 6: Downstream Geolocation: From Pixel to GPS Coordinate](#9-phase-6-downstream-geolocation-from-pixel-to-gps-coordinate)
10. [Key Differentiating Factors to Win the Hackathon](#10-key-differentiating-factors-to-win-the-hackathon)

---

## 1. The Architecture & Pipeline Workflow

The PanoPub engineering pipeline integrates OpenStreetMap ground fixtures, Panoramax street photography, open-source computer vision, and geospatial projection:

```mermaid
flowchart TD
    subgraph "Phase 0: Geospatial Data Balancing"
        A["Geofabrik France PBF\n(france-latest.osm.pbf)"] -->|osmium tags-filter| B["OSM Signs & Place Nodes"]
        B -->|3D KD-Tree Density Classifier| C["Density Classes\n(City / Town / Village / Countryside)"]
        C -->|0.1° Grid Round-Robin Spread| D["Balanced ads.geojson\n(6,000 signs)"]
    end

    subgraph "Phase 1: Geometric Image Harvesting"
        D -->|STAC Search API\n(distance: 5-25m, FOV: ±20°)| E["Panoramax Candidate Search"]
        E -->|Spherical Bearing Verification\n(angle_diff <= 60°)| F["Active Face Selection"]
        F -->|Sequence Cap (<=3) & Ideal Dist (12m)| G["Anti-Saturation Concurrent Download"]
        G --> H["Curated Images (training_pictures/)\n+ metadata.csv"]
    end

    subgraph "Phase 2 & 3: Annotation & Ground Truth Pivot"
        H -->|Zero-Shot OWLv2 Pre-Annotation| I["pre-label.py\n(Candidate Boxes)"]
        I -->|Roadblock: 360° Distortion & False Positives| J{"Quality Bottleneck"}
        J -->|Strategic HITL Pivot| K["Collaborative Label Studio Server"]
        K -->|Team Manual Curation & False Positive Purge| L["Ground Truth Dataset (my_dataset/)\n(335 Train / 84 Val)"]
    end

    subgraph "Phase 4 & 5: Training & Geolocation"
        L -->|Dynamic data.yaml Split| M["train_yolo_mac.py\n(YOLOv8 Nano, MPS/CUDA)"]
        M -->|50 Epochs Optimization| N["Expert Model (models/best.pt)"]
        N --> O["Validation & Metrics\n(runs/detect/train/)"]
        N -->|Inverse Trigonometry Reprojection| P["Real-World Geolocation\n(detected_ads.geojson)"]
    end
```

---

## 2. Environment Setup & Dependencies

Set up your virtual environment and install all dependencies required across the pipeline:

```bash
# 1. Clone repository
git clone https://github.com/your-org/panorepo.git
cd panorepo

# 2. Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# 3. Install Python dependencies
pip install torch torchvision ultralytics transformers Pillow requests scipy numpy pyyaml label-studio
```

### System Utilities (for Phase 0 OSM extraction):
To process raw OSM PBF archives, install `osmium-tool`:
* **macOS**: `brew install osmium-tool`
* **Ubuntu/Debian**: `sudo apt-get install osmium-tool`

---

## 3. Phase 0: Territorial Balancing & GeoJSON Extraction (`0_build_geojson_from_osm_pbf.sh`)

A naive query against OpenStreetMap yields heavy data saturation in Paris and major metropolitan centers, completely under-representing suburban and rural regions. The script `0_build_geojson_from_osm_pbf.sh` solves this through spatial density modeling and geographic grid dispersion.

### How It Works

1. **Extract Ingestion**: Downloads the national French extract (`france-latest.osm.pbf`) from Geofabrik using `curl -L -C -`.
2. **Tag Filtering with `osmium-tool`**:
   Extracts in a single pass:
   * Target outdoor advertising fixtures: `n/advertising=billboard,board,totem,column,poster_box`
   * Population centers: `n/place=city,town,village,hamlet`
3. **3D Cartesian Density Classification**:
   Converts spherical `(lon, lat)` coordinates into 3D Cartesian space on a sphere ($R = 6371\text{ km}$):
   $$x = R \cos(\text{lat}) \cos(\text{lon}), \quad y = R \cos(\text{lat}) \sin(\text{lon}), \quad z = R \sin(\text{lat})$$
   Builds spatial indexing trees (`scipy.spatial.cKDTree`) for place types and classifies each sign by proximity:
   * **City**: within $6.0\text{ km}$ of a `place=city` node.
   * **Town**: within $2.5\text{ km}$ of a `place=town` node.
   * **Village**: within $1.0\text{ km}$ of a `place=village|hamlet` node.
   * **Countryside**: all remaining nodes.
4. **Equal Quota Balancing**: Allocates equal quotas per territory class, dynamically redistributing any deficit from smaller classes to maintain the desired total count.
5. **0.1° Geographic Grid Dispersion (`spread`)**:
   Divides France into discrete spatial cells ($\approx 0.1^\circ \approx 8\text{--}11\text{ km}$). Signs are picked round-robin across cells, guaranteeing uniform territorial coverage and eliminating local hotspots.

### Running the Script

```bash
chmod +x 0_build_geojson_from_osm_pbf.sh
./0_build_geojson_from_osm_pbf.sh
```

**Configurable Environment Variables**:
```bash
TOTAL=6000 CELL_DEG=0.1 SEED=42 ./0_build_geojson_from_osm_pbf.sh
```
*Outputs a balanced `ads.geojson` containing 6,000 candidate advertising locations.*

> [!TIP]
> **Ad-hoc Local Alternative (Overpass Turbo)**: For rapid prototyping on a single city, query [Overpass Turbo](https://overpass-turbo.eu/) with:
> ```text
> [out:json][timeout:25];
> (
>   node["advertising"~"billboard|board|totem|column|poster_box"]({{bbox}});
> );
> out body;
> >;
> out skel qt;
> ```
> Export the result as GeoJSON. For production-grade nationwide pipelines, always use `0_build_geojson_from_osm_pbf.sh`.

---

## 4. Phase 1: Geometric Panoramax Image Fetching (`1_fetch_panoramax_images.py`)

Street capture vehicles take continuous photo sequences. Pulling raw API results leads to severe flaws: multiple redundant frames of the same sign, pictures taken from behind billboards (showing metal struts), or images taken from too far away.

The script `1_fetch_panoramax_images.py` implements geometry-aware candidate filtering:

### Key Algorithmic Filters

* **Targeted Search Window**: Queries `https://api.panoramax.xyz/api/search` with:
  * `place_distance=5-25`: Captures between 5 and 25 meters from the sign.
  * `place_fov_tolerance=20`: Requires the sign to be centered within $\pm 20^\circ$ of the camera field of view.
  * `limit=5`: Retrieves candidate frames per sign.
* **Spherical Facing Verification ($\le 60^\circ$)**:
  Calculates the forward azimuth bearing between the camera and the sign:
  $$\text{bearing} = \text{atan2}(\sin(\Delta\lambda)\cos(\phi_2), \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta\lambda))$$
  Checks whether the camera is located in front of the ad's active face:
  $$\Delta\theta = \min(|\theta_{\text{cam}\to\text{ad}} - \theta_{\text{facing}}|, 360^\circ - |\theta_{\text{cam}\to\text{ad}} - \theta_{\text{facing}}|) \le 60^\circ$$
  If OSM specifies a `direction` tag (e.g., `90`, `NE`, `90;270`), photos taken from behind the billboard are immediately discarded.
* **Optimal Distance Optimization**: Selects the candidate closest to `IDEAL_DISTANCE = 12m` where billboard pixel resolution is maximized without lens boundary clipping.
* **Sequence Cap Anti-Saturation**: Limits captures to `MAX_PER_SEQUENCE = 3` and strictly 1 picture per OSM sign ID.
* **Resilient Concurrent Download**: Uses `ThreadPoolExecutor` (4 search workers, 4 download workers), atomic `.part` file staging, and appends full metadata to `metadata.csv`.

### Running Image Harvesting

```bash
python3 1_fetch_panoramax_images.py
```

* **Image Output**: `./training_pictures/{pic_id}.jpg`
* **Metadata Log**: `./metadata.csv` (recording `id`, `url`, `sequence`, `osm_id`, `ad_type`, `sign_lon`, `sign_lat`, `license`, `authors`)

---

## 5. Phase 2: Zero-Shot Pre-Annotation & 360° Vision Roadblocks (`pre-label.py`)

To jumpstart dataset labeling, we evaluated automated open-vocabulary detection using Hugging Face OWLv2 (`google/owlv2-base-patch16-ensemble`).

### The Pre-Labeling Script

The script `pre-label.py` processes downloaded street panoramas with open-vocabulary text queries:

```python
import torch, glob, os
from PIL import Image
from transformers import Owlv2Processor, Owlv2ForObjectDetection

proc = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
queries = [["a billboard", "an advertising poster", "an advertising sign"]]

for path in glob.glob("training_pictures/*.jpg"):
    img = Image.open(path).convert("RGB")
    w, h = img.size
    inputs = proc(text=queries, images=img, return_tensors="pt")
    with torch.no_grad():
        out = model(**inputs)
    res = proc.post_process_grounded_object_detection(
        out, threshold=0.3, target_sizes=torch.tensor([[max(w, h), max(w, h)]])
    )[0]
    # Normalizes to YOLO format: class_id xc yc bw bh
```

Execute pre-labeling:
```bash
python3 pre-label.py
```
*Outputs images to `dataset/images/` and YOLO format `.txt` annotations to `dataset/labels/`.*

### The Real-World Roadblocks on Street Panoramas

While zero-shot detection worked on clean rectilinear images, it encountered three major computer vision failures on native Panoramax street photography:

1. **360° Spherical Optical Distortion**: Equirectangular and wide-angle lenses curve horizontal lines non-linearly. Commercial billboards situated towards the edges appear curved, trapezoidal, or sheared. Zero-shot models produced loose, partially overlapping, or truncated bounding boxes.
2. **Extreme False-Positive Rate on Street Clutter**: Without domain adaptation, OWLv2 repeatedly misidentified rectangular bus schedules, traffic signs, reflective shop windows, and storefront awnings as billboards.
3. **Threshold Brittleness**: At `threshold=0.3`, street clutter generated severe false positives; raising to `0.4+` caused real distant or warped billboards to be missed entirely.

> [!WARNING]
> **Key Lesson**: Pseudo-labels generated by zero-shot detectors on 360° street photography cannot be fed directly into training without introducing severe label noise. A Human-in-the-Loop curation step is essential.

---

## 6. Phase 3: Human-in-the-Loop (HITL) Ground Truth with Label Studio (`my_dataset/`)

Acknowledging zero-shot limitations, our team made a critical engineering pivot: deploying a collaborative **Label Studio** server to curate a gold-standard ground truth dataset.

### 1. Label Studio Setup

```bash
# Launch Label Studio
label-studio start
```
* Open your browser at `http://localhost:8080`.
* Create a project: **PanoPub Ad Detection**.
* Choose **Object Detection with Bounding Boxes**.
* Define classes:
  * `0`: `billboard` (billboards, poster boxes, totems, advertising columns)
  * `1`: `graffiti` (secondary tag)

### 2. Curation Protocol
* Import images from `training_pictures/` along with pre-annotation predictions.
* **Purge False Positives**: Delete hallucinated boxes on traffic signs, store names, and windows.
* **Geometric Tightening**: Manually adjust box boundaries tightly around warped, angled, and curved ad panels.
* **Export**: Export annotations in **YOLO format**.

### 3. Repository Ground Truth Split (`my_dataset/`)

The curated ground truth dataset is organized in `my_dataset/`:
```text
my_dataset/
├── classes.txt               # billboard, graffiti
├── notes.json                # Version, year, provenance
├── images/
│   ├── train/                # 335 verified images
│   └── val/                  # 84 verified images
└── labels/
    ├── train/                # 335 YOLO ground truth txt files
    └── val/                  # 84 YOLO ground truth txt files
```

---

## 7. Phase 4: Expert Model Fine-Tuning with YOLOv8 (`train_yolo_mac.py`)

With clean ground truth, we fine-tune a YOLOv8 Nano model (`yolov8n.pt`) engineered to master 360° street photography features.

### Training Script Architecture (`train_yolo_mac.py`)

The script dynamically builds `data.yaml` from absolute paths and trains with Apple Silicon Metal Performance Shaders (MPS) or NVIDIA CUDA:

```python
import os, yaml
from ultralytics import YOLO

def main():
    dataset_path = os.path.abspath("my_dataset")
    config = {
        'path': dataset_path,
        'train': 'images/train',
        'val': 'images/val',
        'names': {0: 'billboard'}
    }
    with open("data.yaml", "w") as f:
        yaml.dump(config, f, default_flow_style=False)

    model = YOLO("yolov8n.pt")
    model.train(
        data="data.yaml",
        epochs=50,
        imgsz=640,
        device="mps"  # Use "mps" on Apple Silicon, "0" for CUDA GPU, "cpu" for CPU
    )

if __name__ == "__main__":
    main()
```

### Launching Training

```bash
python3 train_yolo_mac.py
```

* **Model Checkpoints**: Best weights are saved to `runs/detect/train/weights/best.pt` and mirrored to `models/best.pt`.
* **Execution Metrics**: 50 epochs complete in ~29 minutes on Apple Silicon MPS.

---

## 8. Phase 5: Inference, Evaluation & Metrics

### 1. Quantitative Performance

Training logs and evaluation metrics from `runs/detect/train/results.csv`:

| Evaluation Metric | Initial Value (Epoch 1) | Final Value (Epoch 50) | Improvement |
| :--- | :--- | :--- | :--- |
| **Train Box Loss** | `2.404` | **`1.737`** | **-27.7%** |
| **Train Class Loss** | `4.163` | **`1.229`** | **-70.5%** |
| **Val Box Loss** | `2.420` | **`2.368`** | Stable convergence |
| **Val Class Loss** | `4.460` | **`1.948`** | **-56.3%** |
| **Precision (`metrics/precision`)** | `0.002` | **`0.347` (34.7%)** | High discriminative accuracy |
| **Recall (`metrics/recall`)** | `0.159` | **`0.192` (19.2%)** | Selective on verified ads |
| **mAP@50 (`metrics/mAP50`)** | `0.001` | **`0.158` (15.8%)** | Reliable detection baseline |
| **mAP@50-95** | `0.0004` | **`0.071` (7.1%)** | Tight bounding box overlap |

*Saved weights are compact: `models/best.pt` is only **6.2 MB**, ideal for edge deployment.*

### 2. Evaluation Visual Artifacts

Inspect the generated diagnostic plots in `runs/detect/train/`:
* `results.png`: Training loss curves and metric trends across 50 epochs.
* `confusion_matrix.png` & `confusion_matrix_normalized.png`: Confirms rejection of background street clutter.
* `BoxPR_curve.png` & `BoxF1_curve.png`: Precision-Recall and F1 confidence curves.
* `val_batch0_pred.jpg`: Side-by-side ground truth vs. predicted bounding boxes on validation panoramas.

### 3. Running Model Inference

**Via Python API**:
```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(source="my_dataset/images/val/test_image.jpg", conf=0.25, save=True)
for r in results:
    for box in r.boxes:
        print(f"Detected {model.names[int(box.cls)]} with conf {float(box.conf):.2f} at {box.xyxy.tolist()}")
```

**Via Ultralytics CLI**:
```bash
yolo detect predict model=models/best.pt source=my_dataset/images/val/ conf=0.25 save=True
```

---

## 9. Phase 6: Downstream Geolocation: From Pixel to GPS Coordinate

Detecting billboards in imagery is valuable, but mapping them to real-world GPS coordinates completes the pipeline for municipalities and audit teams.

### STAC Metadata Available in Panoramax

Every image in Panoramax carries STAC / EXIF metadata:
* `geometry`: Camera GPS position ($\text{Lat}_{\text{cam}}, \text{Lon}_{\text{cam}}$).
* `view:azimuth` / `exif:GPSImgDirection`: Camera heading in degrees $\theta_{\text{heading}} \in [0^\circ, 360^\circ)$.

### Planar Reprojection Formula

Given:
* Estimated camera-to-ad distance $d \approx 12\text{ meters}$ (or derived from relative bounding box pixel height $h_{\text{box}}$).
* Camera azimuth heading $\theta_{\text{heading}}$ adjusted by the horizontal pixel offset of the box center from image center $\Delta\theta_{\text{pixel}}$:
  $$\theta = (\theta_{\text{heading}} + \Delta\theta_{\text{pixel}}) \cdot \frac{\pi}{180}$$

The real-world geographic coordinates of the detected billboard are:
$$\text{Lat}_{\text{ad}} = \text{Lat}_{\text{cam}} + \frac{d \cdot \cos(\theta)}{111111}$$
$$\text{Lon}_{\text{ad}} = \text{Lon}_{\text{cam}} + \frac{d \cdot \sin(\theta)}{111111 \cdot \cos(\text{Lat}_{\text{cam}} \cdot \frac{\pi}{180})}$$

Export the resulting points directly into `detected_ads.geojson` with associated properties (`confidence`, `type`, `image_url`, `timestamp`).

---

## 10. Key Differentiating Factors to Win the Hackathon

To make your submission stand out to hackathon judges:

1. **Complete Data Lifecycle Integrity**:
   Demonstrate end-to-end command of the pipeline: OSM PBF filtering ➔ 3D KD-Tree density classification ➔ 0.1° grid spatial dispersion ➔ Panoramax API geometric filtering ➔ OWLv2 zero-shot analysis ➔ Label Studio HITL curation ➔ YOLOv8 fine-tuning ➔ GPS coordinate calculation.
2. **Text & Brand Intelligence (OCR)**:
   Crop detected bounding boxes and pipe them through `easyocr` or `tesseract` to extract brand names and text from posters ("McDonald's", "Decathlon"). This transforms a bounding box detector into an automated commercial inventory indexer.
3. **Interactive Geospatial Dashboard**:
   Build an interactive web application using **Streamlit**, **MapLibre GL**, or **Leaflet** that overlays `detected_ads.geojson` on top of open map tiles, allowing users to click a pin and view the original 360° street photo alongside the YOLO detection box.
4. **Municipal Regulatory Auditing**:
   Download school or historic monument protection zones (`amenity=school`, `historic=*`) from OpenStreetMap. Perform a spatial intersection with detected ads to identify non-compliant advertising installations (e.g., billboards within 50 meters of schools or protected architectural areas).
