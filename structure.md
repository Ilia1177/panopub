***Context***

> * **The Drama:** France has 290,000 ad billboards. Between 30% and 50% play hide-and-seek with the tax authorities (TLPE) and are "ghost signs" 👻 (paying absolutely zero).
> * **The Hot Potato (2024):** The State threw the problem at the mayors: "Here, you police it!". But they gave them zero budget and zero tech. They literally expect them to patrol thousands of streets with a measuring tape 📏 and a notepad. Mission impossible!
> * **Our Magic (PanoPub):** The ultimate anti-evasion radar. We mix **Panoramax** (the open-source Street View) with an **AI (YOLOv8)** to play "Where's Waldo?" with the ads. We detect, geolocate, and snitch to the city hall so they can recover their lost millions without ever leaving their chairs. 💸🤖

> **Context and Justification Document for the Hackathon Pitch — PanoPub Project**  
> *Why are we solving this problem? Legal & tax framework, municipal oversight vacuum, economic impact, and technological justification.*

#Transition here (joana: alright you have a point, its very important to track these undeclared billboards. but how do you do it? 'slide goes to thw bald guy' do you hop on your bike and play hide and seek with the advertisements, it would take forever!

afina: thats how they would do it in the ice age, but thankfully we live in 2026. thanks to the open street maps project, we  have managed to provide a way quicker digital solution...)

2. How do we solved this

-Q: How did we got the images?
> * **The Lazy Dream vs. The Street Reality:** Did we hop on a bicycle with an iPhone to snap 6,000 billboards across France? Absolutely not, we are developers, we don't do outside cardio.
> * **The OSM Feast (`0_build_geojson_from_osm_pbf.sh`):** We downloaded the entire national French dataset (`france-latest.osm.pbf`) from Geofabrik and extracted street advertising fixtures (`billboard, board, totem, column, poster_box`) using `osmium-tool`.
> * **Curing the Paris Obsession (KD-Tree Balancing):** Because OpenStreetMap contributors love mapping every trash can in Paris while ignoring the rest of France, a naive query would have turned our dataset into a Parisian brochure. So we built a 3D Euclidean `cKDTree` on a 6,371 km sphere to classify every sign into City, Town, Village, or Countryside, slapping a strict 25% quota on each (1,500 signs each, total 6,000 in `ads.geojson`). Then we applied a 0.1° grid round-robin filter (~10 km cells) so Paris didn't swallow the whole cake.
> * **Panoramax API without taking pictures of metal poles (`1_fetch_panoramax_images.py`):** We queried the open Panoramax API (`5–25m`, sweet spot at `12m`, $\pm 20^\circ$ FOV). But capture cars shoot bursts of photos like tourists in Versailles:
>   - We capped candidates to **max 3 per sequence** and strictly 1 per billboard (no 40-photo albums of the exact same supermarket poster).
>   - We calculated spherical bearings ($\Delta\theta \le 60^\circ$) against OSM facing directions. Why? Because otherwise, half our dataset would be glorious HD photos of the galvanized steel scaffolding on the *back* of the billboard!
>   - Downloaded concurrently with `ThreadPoolExecutor` and logged to `metadata.csv`.

-Q: How we detect a billboard.
> * **The "Zero-Shot" Delusion (`2_pre-label.py`):** At first, we thought we were geniuses: "Let's just ask an open-vocabulary model (Hugging Face OWLv2) to find *'a billboard, an advertising poster, an advertising sign'* and go grab a beer!".
> * **The Funhouse Mirror Effect:** Panoramax uses 360° panoramic cameras. Optical physics had other plans: roadside billboards were bent into curved trapezoids and bananas. OWLv2 had an existential breakdown, missing real warped billboards and hallucinating ads on every bus schedule, shop awning, and reflective bakery window in the republic.
> * **The Road-Sign Exorcist (`3_filter-roadsigns.py`):** We cropped candidate boxes and ran them through a French road-sign classifier (`Panoramax/classify_fr_road_signs`, confidence $\ge 0.6$) to vaporize traffic signs masquerading as billboards.
> * **Human-In-The-Loop (HITL) Suffering (`4_yolo-to-labelstudio.py`):** We swallowed our AI pride, spun up a collaborative **Label Studio** server, and our team manually corrected bounding boxes around distorted panoramas, purged false positives, and built a clean ground truth dataset (~847 verified images).
> * **The Expert Model (`6_train_yolo_mac.py`):** We fine-tuned custom YOLOv8 / YOLO26s models directly on native 360° optics, teaching the network what curved French street advertising actually looks like.
> * **Reverse Geolocation (The "Where is it on the map?" magic):** Once detected in pixels, we take camera GPS + STAC heading (`exif:GPSImgDirection`) and use inverse trigonometry to project pixel bounding boxes back to real-world GPS coordinates (`detected_ads.geojson`).

-Q: Software and tools we used.
> * **Geodata & Harvesting:** OpenStreetMap (OSM), Geofabrik, Panoramax API (STAC), `osmium-tool`, `curl`.
> * **Spatial Geometry & Math:** `scipy.spatial.cKDTree` (3D Euclidean projection on sphere), spherical trigonometry (Haversine & azimuth bearing).
> * **Computer Vision & Deep Learning:** Hugging Face Transformers (`google/owlv2-base-patch16-ensemble`), Ultralytics YOLO (`YOLOv8n`, `YOLO26s`), PyTorch, Pillow.
> * **Traffic Sign Classifier:** Panoramax road sign model (`models/roadsigns/best.pt`).
> * **Annotation & Curation:** Label Studio (collaborative server), custom YOLO export cleaner (`5_clean_yolo_export.py`).
> * **Hardware / "Cooking with Carlos":** Apple Silicon MacBook Pro GPU (`device="mps"`), turning laptops into domestic room heaters for 2+ hours per training run.
> * **Formats:** GeoJSON (`ads.geojson`, `detected_ads.geojson`), YAML, CSV (`metadata.csv`).

-Q: How we improved the models.
> * **Garbage In, Garbage Out:** Ditching noisy OWLv2 pseudo-labels and replacing them with hand-curated ground truth in Label Studio. AI models love quality data more than toddlers love sugar.
> * **Filtering Road Signs:** Pruning traffic sign false positives before human review and training (`3_filter-roadsigns.py`).
> * **The "They are Too Damn Small!" Epiphany (Resolution Boost):** Our dataset audit (`doc/JM_summary.md`) revealed an embarrassing truth: **88.2% of our billboard bounding boxes had at least one side smaller than 16 pixels at 640x640!** They were basically microscopic specks!
>   - In `train-2` (YOLO26s @ 640), the model struggled with mAP50-95 of 10.11% and recall of 22.37%.
>   - We bumped resolution to `imgsz=960` (`train-4`):
>     - **mAP50-95 jumped to 15.43%** (+5.32 percentage points!).
>     - **Recall jumped to 31.24%** (+8.87 points!).
>     - **Precision reached 52.29%** at epoch 50.
> * **Architecture Leap:** Started with `YOLOv8n` (train: mAP50-95 of 7.42%, recall 17.26%) and leveled up to `YOLO26s` at 960x960.


3. Expected result / improvement in tax management
- Hypothetical estimations, graphs
> * **The Official Macro Context (What the repo confirms):**
>   - France collects **~€215 million/year** across ~2,300 municipalities.
>   - Audits reveal **30% to 50%+ of billboards are illegal or undeclared** (in places like Montluçon, non-compliance reached 57%).
>   - Mayors are losing tens of millions of euros while manual private audits cost **€10,000 to €50,000** and expire every 6 months. PanoPub does it for €0 marginal cost.
> * **Hypothetical Estimations & Revenue Simulation Graphs:**
>   - *[To be answered by a teammate: ________________________________________________________________________________________________________________________]*


4. What if?
- We had more time
> * **Eliminate Data Leakage:** Our internal audit caught a duplicate image present in both training and validation splits (`66039459__...` vs `ebce42f7__...`). With more time, we'd implement strict sequence-level / geographic site splitting and a held-out test split.
> * **Tiled Inference / High-Res Crops (SAHI):** Since 88.2% of billboards are tiny (< 16px at 640), instead of squishing a massive 5760x2880 panorama into a single square, we'd cut the panorama into overlapping perspective tiles so distant ads keep their pixels.
> * **OCR & Commercial Tax Roll Matching:** Pipe bounding boxes into EasyOCR or Tesseract to read "Decathlon", "McDonald's", or "Boulangerie Paul", and cross-match company names directly against the municipal SIRENE / TLPE tax registry.
> * **Class Distinction:** Clean up the class contract: properly train separate classes for commercial storefront signs (`enseigne`), billboards (`dispositif publicitaire`), and graffiti, instead of leaving graffiti neglected at 2.2%.
> * **True Multi-View Depth:** Replace our flat 12m heuristic with real triangulation across consecutive Panoramax frames.

- Better hardware
> * **Goodbye Laptop Barbecue:** Our training runs (`train-4`) took **124.3 minutes** on a MacBook Pro GPU (`device="mps"`), averaging ~2.5 minutes per epoch at batch size 4.
> * **Enter Dedicated NVIDIA / CUDA Clusters:** With beefy GPUs (A100 / RTX 4090):
>   - We could train **YOLOv8x / YOLO26x** (heavy backbones) instead of being stuck with Nano/Small models.
>   - Increase batch sizes from 4 to 32 or 64 so training stabilizes and converges much faster.
>   - Train at native high resolutions (1280+) without running out of unified memory or throttling the CPU.
>   - 100 epochs would finish in 10 minutes instead of a 2-hour coffee break while praying the laptop fan doesn't take off.

- Bigger dataset
> * **We have 6,000 candidates waiting!** Our `ads.geojson` already has 6,000 curated OSM candidate locations across France, but our current annotated dataset only has 847 verified images (677 train, 170 val). France has 290,000 billboards out there!
> * **Hard Negative Mining:** We currently have only 17 empty label files. Adding hundreds of verified negative images (empty streets, complex intersections, storefront windows, road signs) would teach YOLO to stop being trigger-happy and slash false positives.
> * **Environmental Diversity:** Snow, night panoramas, pouring rain, rural departmental roads, and dense industrial parks.


5. Future plans
- Expand data with billboards around the world
> * **Panoramax is Sovereign & Open:** Unlike Google Street View (which charges an arm and a leg and bans you from saving images), Panoramax is open-source, federated, and expanding globally under the IGN / OpenStreetMap umbrella.
> * **Worldwide OSM Coverage:** OpenStreetMap has millions of billboards mapped globally (`advertising=billboard`). Our `0_build_geojson_from_osm_pbf.sh` script can ingest any Geofabrik regional PBF extract (Spain, Germany, Latin America, USA).
> * **Beyond Billboards — Full Urban Asset Auditing:** Expand the model to detect EV charging stations, bus shelters, illegal banners, and commercial displays violating protected zones (e.g., historical monuments or schools).
> * **Turnkey Municipal SaaS:** Package the pipeline into an interactive map dashboard (Streamlit / MapLibre) where any mayor in the world uploads their local tax list, clicks "Run", and watches our AI point out which billboards are playing hide-and-seek.