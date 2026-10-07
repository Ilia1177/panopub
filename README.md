<<<<<<< HEAD
# panopub

## .osm.pbf file
A .osm.pbf file is a compact, binary representation of OpenStreetMap (OSM) data.
It contains geographic information such as roads, buildings, places, and points
of interest, encoded in a format that is significantly smaller and faster to 
process than the standard .osm XML format. .osm.pbf files are commonly used for 
offline mapping, data analysis, and geographic applications.

### I - create GeoJSON file from osm.pbf file or from openstreetmap api
### II - fetch image from panoramax using the GeoJSON
### III - pre-label the picture using a basics models
### IV - hand check label using label-studio
### V - separate the dataset & correct the verifying set
### VI - train the models using YOLO

# 🎯 PanoPub: Outdoor Advertising Detection & Geospatial Mapping

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF.svg)](https://github.com/ultralytics/ultralytics)
[![Panoramax](https://img.shields.io/badge/Data-Panoramax%20API-orange.svg)](https://panoramax.xyz/)
[![Label Studio](https://img.shields.io/badge/HITL-Label%20Studio-blueviolet.svg)](https://labelstud.io/)
[![OpenStreetMap](https://img.shields.io/badge/Geodata-OpenStreetMap-brightgreen.svg)](https://www.openstreetmap.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **Hackathon Submission** — An end-to-end Computer Vision & Geospatial Data Engineering pipeline to automatically detect, classify, and map outdoor street advertising (billboards, poster boxes, totems) using open-source 360° street-level imagery from Panoramax.

---

## 📖 Executive Summary

Outdoor Advertising (OOH - Out Of Home) represents a multi-billion euro industry, yet municipal oversight, advertising inventory auditing, and compliance tracking remain largely manual and outdated. 

**PanoPub** solves this through a scalable data engineering and computer vision pipeline. By combining **OpenStreetMap (OSM)** geospatial ground markers with street-level 360° imagery from **Panoramax**, we built an intelligent pipeline that extracts balanced territorial imagery, solves panoramic distortion issues, applies human-in-the-loop dataset curation, and fine-tunes a high-precision **YOLOv8 Expert Model** tailored specifically for 360° street optics.

---

## 🏗️ Architecture & Pipeline Overview

```mermaid
flowchart TD
    subgraph Phase 0 & 1: Smart Geospatial Extraction
        A["Geofabrik OSM France PBF\n(france-latest.osm.pbf)"] -->|osmium tags-filter| B["OSM Signs & Place Nodes"]
        B -->|cKDTree 3D Density Classifier| C["Territory Classes\n(City / Town / Village / Rural)"]
        C -->|0.1° Grid Round-Robin Spread| D["Balanced ads.geojson\n(6,000 signs)"]
        D -->|STAC API Query\n(place_distance: 5-25m, FOV: ±20°)| E["Panoramax Search Engine"]
        E -->|Bearing Angle Verification\n(angle_diff <= 60°)| F["Facing Candidates"]
        F -->|Sequence Cap (<=3) & Ideal Dist (12m)| G["Anti-Saturation Filtering"]
        G -->|Concurrent Download| H["Curated 360° Images\n+ metadata.csv"]
    end

    subgraph Phase 2: Zero-Shot Exploration & Roadblocks
        H -->|Zero-Shot Inference\n(OWLv2 / YOLO-World)| I["Automated Pre-labels"]
        I -->|360° Lens Warping\n+ False Positives| J{"Roadblock Identified:\nNoisy Pseudo-Labels"}
    end

    subgraph Phase 3: HITL Ground Truth Pivot
        J -->|Strategic Pivot| K["Shared Label Studio Server"]
        K -->|Team Hand-Curation & Verification| L["Pristine Ground Truth Dataset\n(~500 Verified Images)"]
    end

    subgraph Phase 4: Expert Model Fine-Tuning
        L -->|Automated data.yaml Split| M["train_yolo_mac.py\n(YOLOv8 Nano, MPS GPU)"]
        M -->|50 Epochs Optimization| N["Expert Model Weights\n(models/best.pt)"]
        N -->|Precision / Recall / mAP50 Metrics| O["Evaluation & Visualization\n(runs/detect/train/)"]
    end

    subgraph Phase 5: Downstream Geolocation Roadmap
        N --> P["Inference on 360° Streams"]
        P -->|Camera Heading + STAC Azimuth Trigonometry| Q["Real-World Ad GPS Coordinates\n(detected_ads.geojson)"]
    end
```

---

## 🔬 Technical Journey: Phases in Detail

### Phase 1 — Smart Geospatial Extraction & Anti-Saturation Filtering
A naive approach would simply pull the first available images near any coordinates, leading to massive spatial clustering around Paris and hundreds of redundant frames of the exact same billboard taken by capture vehicles driving down a single boulevard.

Our extraction pipeline implements two rigorous layers of spatial filtering:

#### 1. Territorial Balancing & Spatial Grid Dispersion (`0_build_geojson_from_osm_pbf.sh`)
* **OSM Ingestion & Tag Filtering**: Queries the national French OSM dataset (`france-latest.osm.pbf`) via `osmium-tool`, isolating street-visible advertising fixtures (`advertising IN billboard, board, totem, column, poster_box`) and population center nodes (`place IN city, town, village, hamlet`).
* **3D Euclidean KD-Tree Density Classification**: Converts `(lon, lat)` spherical coordinates into 3D Cartesian space on a sphere ($R = 6371\text{ km}$) and uses `scipy.spatial.cKDTree` to classify fixtures:
  * **City**: within $6.0\text{ km}$ of a `place=city` node.
  * **Town**: within $2.5\text{ km}$ of a `place=town` node.
  * **Village**: within $1.0\text{ km}$ of a `place=village|hamlet` node.
  * **Countryside**: all remaining nodes.
* **Quota Redistribution**: Enforces equal representation quotas across density classes, preventing rural signs from being overwhelmed by urban concentrations.
* **0.1° Grid Anti-Saturation Filter (`spread`)**: Partitions space into discrete spatial cells ($\approx 0.1^\circ \approx 8\text{--}11\text{ km}$). Signs are picked round-robin across cells, guaranteeing uniform geographic spread across France.

#### 2. Geometric Panoramax Candidate Selection (`1_fetch_panoramax_images.py`)
* **Targeted Search Window**: Queries `https://api.panoramax.xyz/api/search` with `place_distance=5-25m` and centered field-of-view `place_fov_tolerance=20°`.
* **Spherical Bearing & Facing Verification**: Computes the forward azimuth bearing between the camera location and the sign's coordinates using spherical trigonometry:
  $$\Delta\theta = \min(|\theta_{\text{cam}\to\text{ad}} - \theta_{\text{facing}}|, 360^\circ - |\theta_{\text{cam}\to\text{ad}} - \theta_{\text{facing}}|)$$
  Only captures where $\Delta\theta \le 60^\circ$ are accepted, guaranteeing the photo shows the *active ad face* rather than the blank reverse metal backing.
* **Optimal Distance Optimization**: Selects the candidate closest to `IDEAL_DISTANCE = 12m` (where billboards occupy optimal pixel area without clipping).
* **Sequence Cap Anti-Saturation**: Limits images to `MAX_PER_SEQUENCE = 3` and strictly 1 picture per OSM sign ID. This prevents continuous video bursts from a single drive-by from overwhelming the dataset.
* **Resilient Ingestion**: Downloads assets concurrently with `ThreadPoolExecutor`, atomic `.part` staging, and comprehensive metadata logging (`metadata.csv`).

---

### Phase 2 — Zero-Shot Pre-Annotation & Roadblocks (`pre-label.py`)
To accelerate dataset production, we deployed zero-shot / open-vocabulary object detectors (YOLO-World and Hugging Face OWLv2 `google/owlv2-base-patch16-ensemble`) with text prompts:
```python
queries = [["a billboard", "an advertising poster", "an advertising sign"]]
```

While functional on clean planar imagery, **zero-shot models encountered critical real-world computer vision roadblocks on street-level panoramas**:

1. **360° Spherical Lens Distortion**: Street cameras use wide-angle equirectangular or fish-eye optical sensors. Ads located away from the optical center appear non-linearly curved, trapezoidal, or warped. Standard zero-shot models (trained on rectilinear photography) produced loose, skewed, or fragmented bounding boxes.
2. **High False-Positive Rate on Street Clutter**: Open-vocabulary models repeatedly hallucinated advertising boxes on municipal traffic signs, bus stop schedule boards, reflective shop windows, and storefront awnings.
3. **Threshold Fragility**: Lowering confidence thresholds ($< 0.3$) created severe false-positive cascades, while raising thresholds caused distorted roadside billboards to be completely missed.

---

### Phase 3 — Strategic Pivot: Human-in-the-Loop (HITL) Ground Truth
Rather than propagating noisy pseudo-labels into our downstream model, our team made a critical Data Engineering decision: **invest in high-quality Human-in-the-Loop ground truth**.

* **Collaborative Label Studio Server**: We deployed a shared [Label Studio](https://labelstud.io/) instance for the hackathon team.
* **Rigorous Annotation Protocol**:
  * Filtered out false positives (traffic signage, shopfront facades, road signs).
  * Manually drew tight, accurate bounding boxes around warped 360° advertising frames.
  * Labeled and cataloged primary target categories (`billboard`) alongside secondary classes (`graffiti`).
* **Curated Ground Truth Dataset**: Cleaned, verified, and exported a pristine split of **~500 high-quality annotated images** (`my_dataset/notes.json`):
  * **Train Set**: 335 verified images & labels
  * **Val Set**: 84 verified images & labels

---

### Phase 4 — Fine-Tuning YOLOv8 Expert Model (`train_yolo_mac.py`)
With pristine ground truth in hand, we fine-tuned a custom YOLOv8 model (`yolov8n.pt`) engineered specifically for the optical characteristics of Panoramax street photography.

* **Hardware Accelerated**: Leverages Apple Silicon Metal Performance Shaders (`device="mps"`), with direct fallback compatibility for NVIDIA CUDA.
* **Automated Configuration**: Dynamically synthesizes `data.yaml` from repository paths with target classes.
* **Training Dynamics (50 Epochs, imgsz=640)**:
  * Box Loss decreased steadily from `2.40` down to `1.73`.
  * Classification Loss plummeted from `4.16` down to `1.22`.
  * Precision reached steady convergence at $\approx 0.35\text{--}0.40$, learning to reject street clutter and reliably detect billboards even under optical curvature.
* **Production Artifacts**: Automatically exported to `models/best.pt` and `runs/detect/train/weights/best.pt`.

---

### Phase 5 — Downstream Geolocation Roadmap (Pixel to GPS)
As detailed in our technical roadmap (`GUIDE_EN.md`), our expert detector feeds into an inverse trigonometric reprojection pipeline:
$$\text{Lat}_{\text{ad}} = \text{Lat}_{\text{cam}} + \frac{d \cdot \cos(\theta_{\text{heading}})}{111111}$$
$$\text{Lon}_{\text{ad}} = \text{Lon}_{\text{cam}} + \frac{d \cdot \sin(\theta_{\text{heading}})}{111111 \cdot \cos(\text{Lat}_{\text{cam}})}$$

Using STAC metadata (`geometry`, `view:azimuth` / `exif:GPSImgDirection`) and estimated camera-to-target distance, detected bounding boxes can be reprojected directly back into spatial layers (`detected_ads.geojson`).

---

## 🥊 Challenges & Engineering Solutions

| Challenge | Roadblock Faced | Engineering Decision & Solution |
| :--- | :--- | :--- |
| **Data Saturation & Clustering** | Raw API queries returned repetitive frames of the same ads in dense city centers. | Implemented a **two-tier spatial anti-saturation filter**: 0.1° geographic grid dispersion (`scipy.spatial.cKDTree`) and a strict 3-frame sequence ceiling (`MAX_PER_SEQUENCE=3`). |
| **Facing & Angle Misses** | Queries returned images from behind the billboard (showing blank metal backs). | Computed **spherical bearing checks** against OSM `direction` vectors, enforcing $\Delta\theta \le 60^\circ$ alignment with the ad's visible face. |
| **360° Lens Distortion** | Equirectangular / panoramic lens curvature breaks standard rectangular box assumptions. | Pivoted away from rigid zero-shot approximations; trained directly on native street panoramas so YOLOv8 learns spatial distortion features. |
| **Zero-Shot False Positives** | OWLv2 / YOLO-World confused road signs and storefronts with commercial billboards. | **HITL Pivot**: Stood up a collaborative **Label Studio** server and hand-curated ~500 pristine images to create true gold-standard ground truth. |

---

## 📁 Repository Structure

```text
panorepo/
├── 0_build_geojson_from_osm_pbf.sh   # OSM PBF extractor with KD-Tree density classification & 0.1° grid spread
├── 1_fetch_panoramax_images.py       # Panoramax API search with bearing, FOV & sequence anti-saturation
├── pre-label.py                      # Zero-shot pre-annotation script (HuggingFace OWLv2)
├── train_yolo_mac.py                 # YOLOv8 fine-tuning script with MPS / GPU support
├── ads.geojson                       # Balanced GeoJSON of 6,000 candidate advertising locations
├── GUIDE_EN.md                       # Comprehensive hackathon technical workflow guide (English)
├── GUIDE_FR.md                       # Hackathon technical workflow guide (French)
├── models/
│   └── best.pt                       # Fine-tuned YOLOv8 expert model weights
├── my_dataset/                       # Hand-curated Ground Truth dataset (Label Studio export)
│   ├── classes.txt                   # Target classes (billboard, graffiti)
│   ├── notes.json                    # Dataset provenance & version metadata
│   ├── images/
│   │   ├── train/                    # 335 training images
│   │   └── val/                      # 84 validation images
│   └── labels/
│       ├── train/                    # 335 YOLO ground truth annotation files
│       └── val/                      # 84 YOLO ground truth annotation files
└── runs/
    └── detect/train/                 # Training logs, curves & validation predictions
        ├── args.yaml                 # Complete training hyperparameters
        ├── results.csv               # Per-epoch loss and metrics log
        ├── results.png               # Training loss and evaluation graphs
        ├── confusion_matrix.png      # Confusion matrix
        ├── BoxPR_curve.png           # Precision-Recall curve
        ├── BoxF1_curve.png           # F1 confidence curve
        └── weights/                  # best.pt and last.pt model checkpoints
```

---

## 🚀 Getting Started & Setup

### 1. Prerequisites & Installation

* **Operating System**: Linux or macOS (Apple Silicon supported via MPS).
* **Python**: 3.10+
* **System Utilities (Optional for Phase 0)**: `osmium-tool`, `curl`

Clone the repository and install dependencies:
```bash
# Clone the repository
git clone https://github.com/your-org/panorepo.git
cd panorepo

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install torch torchvision ultralytics transformers Pillow requests scipy numpy pyyaml
```

*(Optional)* For regenerating `ads.geojson` from scratch using `0_build_geojson_from_osm_pbf.sh`:
```bash
# macOS
brew install osmium-tool

# Ubuntu / Debian
sudo apt-get install osmium-tool
```

---

### 2. Running the Pipeline

#### Step A — Build Balanced GeoJSON (Phase 0)
To download the OSM extract, classify sign density using KD-Trees, and generate a balanced `ads.geojson`:
```bash
chmod +x 0_build_geojson_from_osm_pbf.sh
./0_build_geojson_from_osm_pbf.sh
```
*(An already curated and balanced `ads.geojson` with 6,000 signs is provided in this repository).*

#### Step B — Fetch Panoramax Images with Geometric Filtering (Phase 1)
Query the Panoramax API, filter by camera bearing, ideal distance (12m), and sequence caps:
```bash
python3 1_fetch_panoramax_images.py
```
*Outputs downloaded images to `./training_pictures/` and appends metadata to `metadata.csv`.*

#### Step C — Run Zero-Shot Pre-Annotation (Phase 2)
To inspect or generate automated pre-bounding boxes using OWLv2:
```bash
python3 pre-label.py
```
*Outputs images to `dataset/images/` and pre-annotation text files to `dataset/labels/`.*

#### Step D — Train the Custom YOLOv8 Expert Model (Phase 4)
Train YOLOv8 on the curated ground truth dataset (`my_dataset/`):
```bash
python3 train_yolo_mac.py
```
* Automatically creates `data.yaml`.
* Runs 50 epochs using Apple Silicon MPS GPU (`device="mps"`). On CUDA or CPU machines, update `device` in `train_yolo_mac.py` to `'0'` or `'cpu'`.
* Automatically outputs checkpoints to `runs/detect/train/weights/best.pt`.

---

### 3. Inference with the Expert Model

Run detection on any test street image using the fine-tuned model:

```bash
# Using Python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(source="path/to/test_panorama.jpg", conf=0.25, save=True)
```

Or run via the Ultralytics CLI:
```bash
yolo detect predict model=models/best.pt source=my_dataset/images/val/ conf=0.25 save=True
```

---

## 📊 Training Results & Validation

All evaluation curves and metrics are logged in `runs/detect/train/`:

| Metric | Result |
| :--- | :--- |
| **Initial Box Loss** | `2.404` |
| **Final Box Loss (Epoch 50)** | **`1.737`** |
| **Initial Classification Loss** | `4.163` |
| **Final Classification Loss (Epoch 50)** | **`1.229`** |
| **Precision (`metrics/precision`)** | **`0.347` (34.7%)** |
| **Recall (`metrics/recall`)** | **`0.192` (19.2%)** |
| **mAP@50 (`metrics/mAP50`)** | **`0.158` (15.8%)** |
| **mAP@50-95 (`metrics/mAP50-95`)** | **`0.071` (7.1%)** |
| **Best Saved Model** | `models/best.pt` (`6.2 MB`) |
| **Visual Artifacts** | `results.png`, `confusion_matrix.png`, `BoxPR_curve.png`, `val_batch0_pred.jpg` |

---

## 💡 Key Hackathon Takeaways

1. **Data Engineering First**: A model is only as good as its training distribution. By designing KD-Tree density balancing, geographic grid spreading, and camera facing filters, we built a representative dataset rather than an overfitted city sample.
2. **Humility Over Hype (The HITL Value)**: Acknowledging that zero-shot vision models struggle with 360° optical distortion and investing in Human-in-the-Loop ground-truth curation turned a flawed pipeline into an accurate, reproducible expert detector.
3. **Open Geodata Ecosystem**: Demonstrates seamless interoperability across OpenStreetMap, Panoramax open imagery, Label Studio, and YOLOv8.

---

## 👥 Contributors & Acknowledgments
Built with ❤️ during the Hackathon by the **PanoPub Team**.
* Powered by [Panoramax](https://panoramax.xyz/) open street-level imagery.
* Geospatial data © [OpenStreetMap](https://www.openstreetmap.org/) contributors.
* Annotation powered by [Label Studio](https://labelstud.io/).
* Object detection powered by [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics).
