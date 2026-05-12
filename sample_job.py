import rasterio
import rasterio.mask
import r5py as r5
import geopandas as gpd
from shapely.geometry import Polygon, box
from scipy.spatial.distance import jensenshannon
from scipy.stats import entropy
from pyrosm import OSM
import numpy as np
from pyrosm.data import sources
from pyrosm import get_data
import matplotlib.pyplot as plt
import pandas as pd
from datetime import timedelta
from source.reusable import *

maxTime = 60
beta = 0.026823965207235005

osm_path = r"cities/paris/Paris.osm.pbf"
gtfs_path = r"cities/paris/80921.20260312.121737.836569.zip"
pop_path = r"cities/paris/fra_pd_2020_1km.tif"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

wanted = [
    "Paris",
    "Hauts-de-Seine",
    "Seine-Saint-Denis",
    "Val-de-Marne",
]

selected = boundaries[boundaries["name"].isin(wanted)]

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

local_crs = selected.estimate_utm_crs()
city_proj = selected.to_crs(local_crs)

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/paris_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#_____________

osm_path = 'cities/torino/Turin.osm.pbf'
gtfs_path = 'cities/torino/mdb-2687-202602270130.zip'
pop_path = r"cities/milan/ita_general_2020_geotiff/ita_general_2020.tif"

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)
osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

torino = boundaries[
    (boundaries["boundary"] == "administrative") &
    (boundaries["admin_level"].astype(str) == "8") &
    (boundaries["name"].str.contains("Torin", case=False, na=False))
].copy()

def keep_largest_connected_component(gdf):
    gdf = gdf.copy().reset_index(drop=True)
    gdf = gdf[gdf.geometry.notna() & ~gdf.geometry.is_empty].copy()

    sindex = gdf.sindex
    pairs = []

    for i, geom in enumerate(gdf.geometry):
        for j in sindex.query(geom, predicate="intersects"):
            if j <= i:
                continue
            other = gdf.geometry.iloc[j]
            if geom.touches(other) or geom.intersects(other):
                pairs.append((i, j))

    parent = list(range(len(gdf)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for a, b in pairs:
        union(a, b)

    labels = pd.Series([find(i) for i in range(len(gdf))], index=gdf.index)
    largest_component = labels.value_counts().idxmax()

    return gdf.loc[labels == largest_component].copy()

torino_clean = keep_largest_connected_component(torino)

city_proj = torino_clean.copy()
city_proj = city_proj.dissolve()
city_proj = city_proj.to_crs(city_proj.estimate_utm_crs())

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/turin_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#------------------------------------------------------------

osm_path = r"cities/tallinn/Tallinn.osm.pbf"
gtfs_path = r"cities/tallinn/mdb-3047-202603010109.zip"
pop_path = r"cities/tallinn/est_pop_2015_CN_100m_R2025A_v1.tif"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

tallinn = boundaries[
    (boundaries["boundary"] == "administrative") &
    (boundaries["admin_level"].astype(str).str.match(r"^(6|8|9)$")) &
    (boundaries["name"].str.contains("Tallinn", case=False, na=False))
]

city_proj = tallinn.copy()
city_proj = city_proj.dissolve()
city_proj = city_proj.to_crs(city_proj.estimate_utm_crs())

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/tallinn_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#_____________

osm_path = "cities/milan/milan.osm.pbf"
gtfs_path = "cities/milan/gtfs_fixed_fixed.zip"
pop_path = r"cities/milan/ita_general_2020_geotiff/ita_general_2020.tif"
bound_path = "cities/milan/milan.geojson"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

local_crs = boundaries.estimate_utm_crs()
city_proj = boundaries.to_crs(local_crs)

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/milan_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#_____________

osm_path = r"cities/copenhagen/Copenhagen.osm.pbf"
gtfs_path = r"cities/copenhagen/GTFS_clean.zip"
pop_path = r"cities/copenhagen/dnk_pd_2020_1km.tif"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

city_boundary = boundaries[boundaries['admin_level'] == '7']
city_boundary = city_boundary[city_boundary['name'].str.endswith("Kommune")]

local_crs = city_boundary.estimate_utm_crs()
city_proj = city_boundary.to_crs(local_crs)

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
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

#_____________

osm_path = r"cities/brisbane/Brisbane.osm.pbf"
gtfs_path = r"cities/brisbane/mdb-3048-202602280045.zip"
pop_path = r"cities/brisbane/Australian_Population_Grid_2011.tif"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

city_boundary = boundaries[boundaries['name'] == 'City of Brisbane']

local_crs = city_boundary.estimate_utm_crs()
city_proj = city_boundary.to_crs(local_crs)

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/brisbane_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#_____________

osm_path = r'cities/bratislava/bratislavsky.pbf'
gtfs_path = 'cities/bratislava/tld-3363-202602280130.zip'
pop_path = r'cities/bratislava/svk_pop_2026_CN_100m_R2025A_v1.tif'
bound_path = r"cities/bratislava/bratislava_boundary.geojson"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path)

city_proj = gpd.read_file(bound_path)
city_proj = city_proj.to_crs(city_proj.estimate_utm_crs())
city_proj = city_proj.dissolve()

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/bratislava_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#_____________

osm_path = r'cities/singapore/Singapore.osm.pbf'
gtfs_path = r'cities/singapore/mdb-3051-202602171452.zip'
pop_path = r'cities/singapore/sgp_general_population_2020.tif'

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

city_proj = gpd.read_file("cities/singapore/sg_boundary.zip")
city_proj = city_proj.to_crs(city_proj.estimate_utm_crs())
city_proj = city_proj.dissolve()

grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/singapore_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)

#-----------------------
osm_path = r'cities/prague/praha-260508.osm.pbf'
gtfs_path = r'cities/prague/jrdata.zip'
pop_path = r"cities/prague/cze_pd_2020_1km.tif"

osm = OSM(osm_path)
boundaries = osm.get_boundaries()

tn = r5.TransportNetwork(osm_path, gtfs=gtfs_path)

prag = boundaries[
    (boundaries["boundary"] == "administrative") &
    (boundaries["admin_level"].astype(str) == "9") &
    (boundaries["name"].str.contains("Praha", case=False, na=False))
].copy()

city_proj = prag.copy()
city_proj = city_proj.dissolve()
city_proj = city_proj.to_crs(city_proj.estimate_utm_crs())
grid = create_hex_grid(city_proj, 500)

grid_pop = assign_population_to_grid(pop_path, grid, plot=False)

result_geo = build_city_geojson(
    grid=grid,
    grid_pop=grid_pop,
    osm=osm,
    tn=tn,
    output_path="data/prague_scores.geojson",
    transport_modes=[r5.TransportMode.TRANSIT, r5.TransportMode.WALK],
    max_time=maxTime,
    beta=beta,
    gravity_normalisation=100,
)
