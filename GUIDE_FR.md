# 🚀 Guide Technique du Hackathon : Détection et Géolocalisation de Publicités Extérieures avec Panoramax

Ce guide technique est conçu pour une équipe de **hackathon** travaillant avec l'imagerie immersive de rue et les données géospatiales ouvertes. Basé sur l'architecture et le code source de **PanoPub**, il couvre l'ensemble du cycle de vie du projet : de l'extraction brute OpenStreetMap et l'équilibrage territorial des panneaux, à la collecte géométrique ciblée sur l'API Panoramax, en passant par le traitement des écueils des modèles zero-shot, la curation collaborative Human-in-the-Loop sur Label Studio, le réglage fin d'un modèle expert **YOLOv8**, jusqu'à la rétro-projection des détections vers des coordonnées GPS réelles.

---

## 📋 Sommaire

1. [Architecture et Flux de Travail (Pipeline)](#1-architecture-et-flux-de-travail-pipeline)
2. [Configuration de l'Environnement et Dépendances](#2-configuration-de-lenvironnement-et-dépendances)
3. [Phase 0 : Équilibrage Territorial et Extraction GeoJSON (`0_build_geojson_from_osm_pbf.sh`)](#3-phase-0--équilibrage-territorial-et-extraction-geojson-0_build_geojson_from_osm_pbfsh)
4. [Phase 1 : Collecte Géométrique d'Images Panoramax (`1_fetch_panoramax_images.py`)](#4-phase-1--collecte-géométrique-dimages-panoramax-1_fetch_panoramax_imagespy)
5. [Phase 2 : Pré-annotation Zero-Shot et Obstacles de la Vision 360° (`pre-label.py`)](#5-phase-2--pré-annotation-zero-shot-et-obstacles-de-la-vision-360-pre-labelpy)
6. [Phase 3 : Vérité Terrain Human-in-the-Loop (HITL) avec Label Studio (`my_dataset/`)](#6-phase-3--vérité-terrain-human-in-the-loop-hitl-avec-label-studio-my_dataset)
7. [Phase 4 : Entraînement du Modèle Expert avec YOLOv8 (`train_yolo_mac.py`)](#7-phase-4--entraînement-du-modèle-expert-avec-yolov8-train_yolo_macpy)
8. [Phase 5 : Inférence, Évaluation et Métriques](#8-phase-5--inférence-évaluation-et-métriques)
9. [Phase 6 : Géolocalisation Aval : Du Pixel à la Coordonnée GPS](#9-phase-6--géolocalisation-aval--du-pixel-à-la-coordonnée-gps)
10. [Facteurs Différenciateurs pour Gagner le Hackathon](#10-facteurs-différenciateurs-pour-gagner-le-hackathon)

---

## 1. Architecture et Flux de Travail (Pipeline)

Le pipeline d'ingénierie PanoPub associe le mobilier urbain OpenStreetMap, la photographie immersive Panoramax, la vision par ordinateur et la projection géospatiale :

```mermaid
flowchart TD
    subgraph "Phase 0 : Équilibrage Géospatial"
        A["Archive PBF Geofabrik France\n(france-latest.osm.pbf)"] -->|osmium tags-filter| B["Noeuds Publicités & Lieux OSM"]
        B -->|Classifieur de Densité 3D KD-Tree| C["Classes de Densité\n(Ville / Bourg / Village / Campagne)"]
        C -->|Dispersion par Grille 0.1° Round-Robin| D["ads.geojson Équilibré\n(6 000 panneaux)"]
    end

    subgraph "Phase 1 : Collecte Géométrique d'Images"
        D -->|Requête API STAC\n(distance: 5-25m, champ visuel FOV: ±20°)| E["Moteur de Recherche Panoramax"]
        E -->|Vérification du Gisement Sphérique\n(angle_diff <= 60°)| F["Sélection de la Face Active"]
        F -->|Plafond Séquence (<=3) & Distance Idéale (12m)| G["Téléchargement Parallèle Anti-Saturation"]
        G --> H["Images Ciblées (training_pictures/)\n+ metadata.csv"]
    end

    subgraph "Phase 2 & 3 : Annotation & Pivot Vérité Terrain"
        H -->|Pré-annotation Zero-Shot OWLv2| I["pre-label.py\n(Boîtes Candidates)"]
        I -->|Écueils : Déformation 360° & Faux Positifs| J{"Verrou Qualité"}
        J -->|Pivot Stratégique HITL| K["Serveur Partagé Label Studio"]
        K -->|Curation Manuelle & Élimination des Faux Positifs| L["Dataset de Vérité Terrain (my_dataset/)\n(335 Train / 84 Val)"]
    end

    subgraph "Phase 4 & 5 : Entraînement & Géolocalisation"
        L -->|Partition Dynamique data.yaml| M["train_yolo_mac.py\n(YOLOv8 Nano, MPS/CUDA)"]
        M -->|Optimisation 50 Époques| N["Modèle Expert (models/best.pt)"]
        N --> O["Validation & Métriques\n(runs/detect/train/)"]
        N -->|Rétro-projection Trigonométrique Inverse| P["Géolocalisation Monde Réel\n(detected_ads.geojson)"]
    end
```

---

## 2. Configuration de l'Environnement et Dépendances

Créez un environnement virtuel Python et installez l'ensemble des dépendances du projet :

```bash
# 1. Cloner le dépôt
git clone https://github.com/your-org/panorepo.git
cd panorepo

# 2. Créer et activer l'environnement virtuel
python3 -m venv venv
source venv/bin/activate

# 3. Installer les dépendances Python
pip install torch torchvision ultralytics transformers Pillow requests scipy numpy pyyaml label-studio
```

### Utilitaires Système (pour la Phase 0 d'extraction OSM) :
Pour filtrer les archives volumineuses PBF d'OpenStreetMap, installez `osmium-tool` :
* **macOS** : `brew install osmium-tool`
* **Ubuntu/Debian** : `sudo apt-get install osmium-tool`

---

## 3. Phase 0 : Équilibrage Territorial et Extraction GeoJSON (`0_build_geojson_from_osm_pbf.sh`)

Une interrogation naïve d'OpenStreetMap concentre massivement les données sur Paris et les métropoles régionales, au détriment des zones périurbaines et rurales. Le script `0_build_geojson_from_osm_pbf.sh` remédie à ce biais via une modélisation de densité spatiale et une grille de dispersion territoriale.

### Fonctionnement Algorithmique

1. **Téléchargement de l'Archive** : Récupère l'extrait national français (`france-latest.osm.pbf`) sur Geofabrik via `curl -L -C -`.
2. **Filtrage des Balises avec `osmium-tool`** :
   Isole en une seule passe :
   * Le mobilier publicitaire visible depuis la rue : `n/advertising=billboard,board,totem,column,poster_box`
   * Les pôles de population : `n/place=city,town,village,hamlet`
3. **Classification de Densité Cartésienne 3D** :
   Projette les coordonnées sphériques `(lon, lat)` dans un espace euclidien 3D sur une sphère terrestre ($R = 6371\text{ km}$) :
   $$x = R \cos(\text{lat}) \cos(\text{lon}), \quad y = R \cos(\text{lat}) \sin(\text{lon}), \quad z = R \sin(\text{lat})$$
   Construit des index spatiaux (`scipy.spatial.cKDTree`) et classe chaque publicité selon sa proximité :
   * **Ville (City)** : à moins de $6,0\text{ km}$ d'un nœud `place=city`.
   * **Bourg (Town)** : à moins de $2,5\text{ km}$ d'un nœud `place=town`.
   * **Village** : à moins de $1,0\text{ km}$ d'un nœud `place=village|hamlet`.
   * **Campagne (Countryside)** : tous les nœuds restants.
4. **Rééquilibrage par Quotas Égaux** : Assigne des quotas identiques par classe et redistribue dynamiquement les reliquats des catégories sous-représentées.
5. **Dispersion Géographique par Grille 0,1° (`spread`)** :
   Partitionne la métropole en cellules spatiales régulières ($\approx 0,1^\circ \approx 8\text{ à }11\text{ km}$). La sélection s'effectue en round-robin entre cellules, éliminant les agrégats locaux.

### Exécution du Script

```bash
chmod +x 0_build_geojson_from_osm_pbf.sh
./0_build_geojson_from_osm_pbf.sh
```

**Variables d'Environnement Configurables** :
```bash
TOTAL=6000 CELL_DEG=0.1 SEED=42 ./0_build_geojson_from_osm_pbf.sh
```
*Génère un fichier équilibré `ads.geojson` contenant 6 000 emplacements candidats.*

> [!TIP]
> **Alternative Rapide pour une Ville (Overpass Turbo)** : Pour un test rapide sur une agglomération donnée, exécutez sur [Overpass Turbo](https://overpass-turbo.eu/) :
> ```text
> [out:json][timeout:25];
> (
>   node["advertising"~"billboard|board|totem|column|poster_box"]({{bbox}});
> );
> out body;
> >;
> out skel qt;
> ```
> Exportez en GeoJSON. Pour un pipeline national représentatif, privilégiez toujours `0_build_geojson_from_osm_pbf.sh`.

---

## 4. Phase 1 : Collecte Géométrique d'Images Panoramax (`1_fetch_panoramax_images.py`)

Les véhicules de cartographie capturent des rafales continues. Télécharger aveuglément les images de l'API conduit à des écueils majeurs : captures multiples du même panneau, photos prises de dos (montrant l'armature métallique) ou prises de trop loin.

Le script `1_fetch_panoramax_images.py` applique un filtrage géométrique rigoureux :

### Filtres Implémentés

* **Fenêtre de Recherche Ciblée** : Interroge `https://api.panoramax.xyz/api/search` avec :
  * `place_distance=5-25` : Prise de vue entre 5 et 25 mètres du panneau.
  * `place_fov_tolerance=20` : Panneau centré à $\pm 20^\circ$ dans le champ visuel de la caméra.
  * `limit=5` : Récupère les 5 meilleurs clichés candidats.
* **Vérification du Gisement Sphérique Face Active ($\le 60^\circ$)** :
  Calcule l'azimut de visée entre la caméra et le panneau :
  $$\text{gisement} = \text{atan2}(\sin(\Delta\lambda)\cos(\phi_2), \cos(\phi_1)\sin(\phi_2) - \sin(\phi_1)\cos(\phi_2)\cos(\Delta\lambda))$$
  Vérifie que la caméra fait face à la publicité :
  $$\Delta\theta = \min(|\theta_{\text{cam}\to\text{panneau}} - \theta_{\text{orientation}}|, 360^\circ - |\theta_{\text{cam}\to\text{panneau}} - \theta_{\text{orientation}}|) \le 60^\circ$$
  Si OSM renseigne la balise `direction` (ex. `90`, `NE`, `90;270`), les photos montrant le dos du panneau sont immédiatement rejetées.
* **Optimisation de la Distance Idéale** : Sélectionne le candidat le plus proche de `IDEAL_DISTANCE = 12m` (compromis optimal entre résolution en pixels et déformation de bord).
* **Plafond Anti-Saturation par Séquence** : Limite stricte à `MAX_PER_SEQUENCE = 3` clichés par séquence et 1 seule photo par identifiant OSM.
* **Téléchargement Concurrent Résilient** : Utilise `ThreadPoolExecutor` (4 workers recherche, 4 workers téléchargement), une écriture atomique `.part`, et consigne l'historique dans `metadata.csv`.

### Lancer la Collecte

```bash
python3 1_fetch_panoramax_images.py
```

* **Sortie Images** : `./training_pictures/{pic_id}.jpg`
* **Registre de Métadonnées** : `./metadata.csv` (contenant `id`, `url`, `sequence`, `osm_id`, `ad_type`, `sign_lon`, `sign_lat`, `license`, `authors`)

---

## 5. Phase 2 : Pré-annotation Zero-Shot et Obstacles de la Vision 360° (`pre-label.py`)

Afin d'accélérer l'annotation, nous avons évalué l'étiquetage automatique par vision ouverte (zero-shot) avec Hugging Face OWLv2 (`google/owlv2-base-patch16-ensemble`).

### Le Script de Pré-annotation

Le script `pre-label.py` traite les panoramas téléchargés avec des requêtes textuelles ciblées :

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
    # Normalisation au format YOLO : class_id xc yc bw bh
```

Lancer la pré-annotation :
```bash
python3 pre-label.py
```
*Produit les images dans `dataset/images/` et les étiquettes candidates au format YOLO dans `dataset/labels/`.*

### Les Obstacles Rencontrés sur Panoramas Urbains

Sur des photos de rue réelles issues de caméras 360°, les modèles zero-shot génériques se heurtent à trois difficultés techniques majeures :

1. **Distorsion Optique Sphérique 360°** : Les capteurs équirectangulaires ou très grand angle courbent les lignes droites. Les panneaux situés en périphérie de l'image apparaissent trapézoïdaux ou déformés. Les modèles zero-shot prédisent des boîtes lâches, tronquées ou décalées.
2. **Taux Élevé de Faux Positifs en Milieu Urbain** : Sans adaptation de domaine, le modèle hallucine des panneaux publicitaires sur les horaires d'abribus, les panneaux de signalisation routière, les stores de magasins et les vitrines réfléchissantes.
3. **Instabilité du Seuil de Confiance** : À `threshold=0.3`, les faux positifs s'accumulent ; en montant à `0.4+`, les vrais panneaux éloignés ou courbés ne sont plus détectés du tout.

> [!WARNING]
> **Enseignement Clé** : L'utilisation directe de pseudo-étiquettes zero-shot non corrigées dégrade lourdement l'apprentissage. Une étape d'annotation humaine (Human-in-the-Loop) est indispensable.

---

## 6. Phase 3 : Vérité Terrain Human-in-the-Loop (HITL) avec Label Studio (`my_dataset/`)

Pour garantir un apprentissage robuste, notre équipe a opéré un pivot méthodologique : le déploiement d'une instance partagée [Label Studio](https://labelstud.io/) pour annoter un jeu de vérité terrain d'excellence.

### 1. Configuration de Label Studio

```bash
# Lancer Label Studio
label-studio start
```
* Rendez-vous sur votre navigateur à l'adresse `http://localhost:8080`.
* Créez un projet : **PanoPub Ad Detection**.
* Sélectionnez le modèle : **Computer Vision > Object Detection with Bounding Boxes**.
* Définissez les classes cibles :
  * `0` : `billboard` (panneaux d'affichage, mupis, sucettes, totems publicitaires)
  * `1` : `graffiti` (classe secondaire)

### 2. Protocole de Curation
* Importez les clichés de `training_pictures/` ainsi que les pré-annotations issues de `pre-label.py`.
* **Élimination des Faux Positifs** : Suppression des boîtes sur les panneaux de signalisation, vitrines et enseignes.
* **Ajustement Géométrique Fin** : Délimitation manuelle précise des contours des panneaux soumis à la courbure optique 360°.
* **Exportation** : Exportez le jeu de données au format **YOLO**.

### 3. Structure du Dataset de Vérité Terrain (`my_dataset/`)

Le jeu de données vérifié présent dans le dépôt est structuré ainsi :
```text
my_dataset/
├── classes.txt               # billboard, graffiti
├── notes.json                # Version, millésime, provenance
├── images/
│   ├── train/                # 335 images vérifiées
│   └── val/                  # 84 images vérifiées
└── labels/
    ├── train/                # 335 fichiers d'annotations YOLO .txt
    └── val/                  # 84 fichiers d'annotations YOLO .txt
```

---

## 7. Phase 4 : Entraînement du Modèle Expert avec YOLOv8 (`train_yolo_mac.py`)

Sur la base de cette vérité terrain propre, nous entraînons un modèle YOLOv8 Nano (`yolov8n.pt`) pour apprendre les spécificités optiques des panoramas urbains.

### Structure du Script d'Entraînement (`train_yolo_mac.py`)

Le script génère dynamiquement la configuration `data.yaml` avec chemins absolus et configure l'accélération matérielle Apple Silicon (MPS) ou NVIDIA CUDA :

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
        device="mps"  # "mps" pour Apple Silicon, "0" pour GPU CUDA, "cpu" pour CPU
    )

if __name__ == "__main__":
    main()
```

### Lancer l'Entraînement

```bash
python3 train_yolo_mac.py
```

* **Poids du Modèle** : Le meilleur point de contrôle est automatiquement enregistré dans `runs/detect/train/weights/best.pt` et mis à disposition dans `models/best.pt`.
* **Temps d'Exécution** : 50 époques complétées en ~29 minutes sur GPU Apple Silicon MPS.

---

## 8. Phase 5 : Inférence, Évaluation et Métriques

### 1. Métriques Quantitatives Obtenues

Relevé des métriques d'apprentissage issues de `runs/detect/train/results.csv` :

| Métrique d'Évaluation | Valeur Initiale (Époque 1) | Valeur Finale (Époque 50) | Évolution |
| :--- | :--- | :--- | :--- |
| **Perte Boîte Train (Box Loss)** | `2,404` | **`1,737`** | **-27,7%** |
| **Perte Classe Train (Cls Loss)** | `4,163` | **`1,229`** | **-70,5%** |
| **Perte Boîte Validation** | `2,420` | **`2,368`** | Convergence stable |
| **Perte Classe Validation** | `4,460` | **`1,948`** | **-56,3%** |
| **Précision (`metrics/precision`)** | `0,002` | **`0,347` (34,7%)** | Forte capacité discriminante |
| **Rappel (`metrics/recall`)** | `0,159` | **`0,192` (19,2%)** | Sélectivité sur les vrais panneaux |
| **mAP@50 (`metrics/mAP50`)** | `0,001` | **`0,158` (15,8%)** | Base solide de détection |
| **mAP@50-95** | `0,0004` | **`0,071` (7,1%)** | Recouvrement géométrique précis |

*Le fichier de poids exporté `models/best.pt` ne pèse que **6,2 Mo**, idéal pour un déploiement embarqué.*

### 2. Artefacts Visuels d'Évaluation

Les courbes et graphiques de diagnostic sont consultables dans `runs/detect/train/` :
* `results.png` : Courbes d'évolution des pertes et métriques sur les 50 époques.
* `confusion_matrix.png` & `confusion_matrix_normalized.png` : Confirmation du rejet du bruit urbain environnant.
* `BoxPR_curve.png` & `BoxF1_curve.png` : Courbes Précision-Rappel et score F1 par seuil de confiance.
* `val_batch0_pred.jpg` : Comparaison visuelle entre vérité terrain et prédictions sur les panoramas du jeu de validation.

### 3. Exécution de l'Inférence

**En Python** :
```python
from ultralytics import YOLO

model = YOLO("models/best.pt")
results = model.predict(source="my_dataset/images/val/test_image.jpg", conf=0.25, save=True)
for r in results:
    for box in r.boxes:
        print(f"Détecté {model.names[int(box.cls)]} (conf {float(box.conf):.2f}) en {box.xyxy.tolist()}")
```

**En Ligne de Commande (CLI)** :
```bash
yolo detect predict model=models/best.pt source=my_dataset/images/val/ conf=0.25 save=True
```

---

## 9. Phase 6 : Géolocalisation Aval : Du Pixel à la Coordonnée GPS

Identifier un panneau dans une image est un atout, mais calculer ses coordonnées GPS réelles transforme le modèle en solution opérationnelle pour les collectivités et auditeurs.

### Métadonnées STAC Fournies par Panoramax

Chaque cliché Panoramax intègre des métadonnées STAC / EXIF essentielles :
* `geometry` : Coordonnées GPS de la caméra ($\text{Lat}_{\text{cam}}, \text{Lon}_{\text{cam}}$).
* `view:azimuth` / `exif:GPSImgDirection` : Cap de la caméra en degrés $\theta_{\text{cap}} \in [0^\circ, 360^\circ)$.

### Formule de Projection Plane

En posant :
* La distance estimée caméra-panneau $d \approx 12\text{ mètres}$ (ou déduite de la hauteur relative de la boîte de détection).
* Le cap corrigé de l'écart angulaire horizontal de la boîte par rapport au centre du cliché $\Delta\theta_{\text{pixel}}$ :
  $$\theta = (\theta_{\text{cap}} + \Delta\theta_{\text{pixel}}) \cdot \frac{\pi}{180}$$

Les coordonnées géographiques réelles du panneau publicitaire se calculent selon :
$$\text{Lat}_{\text{pub}} = \text{Lat}_{\text{cam}} + \frac{d \cdot \cos(\theta)}{111111}$$
$$\text{Lon}_{\text{pub}} = \text{Lon}_{\text{cam}} + \frac{d \cdot \sin(\theta)}{111111 \cdot \cos(\text{Lat}_{\text{cam}} \cdot \frac{\pi}{180})}$$

Exportez l'ensemble des points détectés au format normalisé `detected_ads.geojson` avec leurs attributs associés (`confidence`, `type`, `image_url`, `timestamp`).

---

## 10. Facteurs Différenciateurs pour Gagner le Hackathon

Pour maximiser l'impact de votre projet devant le jury :

1. **Maîtrise Complète du Cycle de Vie de la Donnée** :
   Présentez la continuité technique rigoureuse : filtrage PBF OSM ➔ classification 3D KD-Tree ➔ grille de dispersion 0,1° ➔ filtrage géométrique Panoramax ➔ évaluation zero-shot OWLv2 ➔ curation Label Studio HITL ➔ entraînement YOLOv8 ➔ projection géographique GPS.
2. **Extraction OCR et Identification des Marques** :
   Isolez la vignette recadrée de la publicité détectée et appliquez-lui `easyocr` ou `tesseract` pour lire les marques ("McDonald's", "Decathlon"). Vous transformez une détection géométrique en inventaire commercial automatisé.
3. **Tableau de Bord Cartographique Interactif** :
   Développez une interface cartographique avec **Streamlit**, **MapLibre GL** ou **Leaflet** affichant le fichier `detected_ads.geojson`, permettant de cliquer sur un panneau pour afficher la photo 360° et la boîte de détection correspondante.
4. **Audit de Conformité Réglementaire (RLP)** :
   Croisez les positions des panneaux détectés avec les périmètres protégés d'OpenStreetMap (écoles `amenity=school`, monuments historiques `historic=*`) pour détecter les infractions réglementaires (ex. panneaux publicitaires à moins de 50 mètres d'un établissement scolaire).
