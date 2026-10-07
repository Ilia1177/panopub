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
