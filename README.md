# Urban Accessibility Analysis

This repository contains the source code for a cross-city urban accessibility analysis bachelor project. The project measures and compares accessibility to key urban amenities using network-based travel times, population-based normalisation, and diversity metrics across functional amenity categories.

## Repository Structure

```text
.
├── README.md
├── source/
│   ├── reusable.py
│   └── ...
├── notebooks/
│   └── examples.ipynb
├── sample_cities/
│   └── ...
└── outputs/
    └── YOURCITY_scores.geojson
```

## Data Sources

The project uses the following data inputs:

- **OpenStreetMap** for points of interest and street network data.
- **Administrative boundary files** for defining study areas.
- **Population raster data** for population assignment and per-capita normalisation.
- **GTFS or public transport feeds** where available for multimodal routing.

Note: only administrative boundary data is optional for reproducing output based on a custom (new) city

## Installation

Clone the repository:

```bash
git clone https://github.com/giucu/Amenity-Accessibility-Index.git
cd Amenity-Accessibility-Index
```

Create and activate a virtual environment, then install the required packages.

```bash
pip install -r requirements.txt
```

## Usage

The main workflow is implemented in the source code and notebooks. Typical usage includes:

1. loading travel network and boundary data with OSM
2. extracting custom boundaries or through manual specification
3. generating and populating the (clipped) grid map
4. specifying origins and POIs (amenities)
5. computing travel times
6. generating accessibility metrics
7. exporting data, viz, and summary tables

Example:

```python
from reusable import *

result = buildcitygeojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/copenhagen_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)
```

## Outputs

The repository produces:

- exported GeoJSON / GIS-ready outputs describing accessibility metrics for input city.
- see notebook samples for examples of:
  - accessibility maps
  - summary tables by city and amenity category
  - population-weighted accessibility scores
  - diversity metrics

## License

This project is licensed under the MIT License. See the `LICENSE` file for details.
