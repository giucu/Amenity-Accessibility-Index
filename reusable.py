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
from reusable import *
from r5py.util.config import Config
import os, json

def create_hex_grid(gdf, radius_meters):
    xmin, ymin, xmax, ymax = gdf.total_bounds
    
    # constants for a flat-topped hexagon
    w = 2 * radius_meters
    h = np.sqrt(3) * radius_meters
    
    x_coords = np.arange(xmin - w, xmax + w, 1.5 * radius_meters)
    y_coords = np.arange(ymin - h, ymax + h, h)
    
    polys = []
    for i, x in enumerate(x_coords):
        for j, y in enumerate(y_coords):
            # Offset every odd column to lock the hexagons together
            y_curr = y + (h / 2) if i % 2 == 1 else y
            
            hex_points = [
                (x + radius_meters * np.cos(angle), y_curr + radius_meters * np.sin(angle))
                # 6 equidistant points around a circle
                for angle in np.linspace(0, 2 * np.pi, 7)[:-1] 
            ]
            polys.append(Polygon(hex_points))
            
    # Convert to GeoDataFrame
    grid = gpd.GeoDataFrame({'geometry': polys}, crs=gdf.crs)
    return gpd.clip(grid, gdf)

def assign_population_to_grid(pop_raster_path, grid, plot=True):
    """
    Assigns population from a GeoTIFF raster to a hex/polygon grid
    using areal disaggregation (proportional area-based allocation).

    Parameters
    ----------
    pop_raster_path : str
        Path to the population GeoTIFF file.
    grid : GeoDataFrame
        Polygon grid (hex or other) to assign population to.
        Can be in any CRS — will be handled internally.
    plot : bool
        If True, plots the resulting population map.

    Returns
    -------
    GeoDataFrame
        A copy of `grid` with a new `population` column.
    """
    # safely handle mixed geometry types without index conflicts
    grid_out = grid.copy()
    grid_out = grid_out.explode(index_parts=False)       # explode MultiPolygons → Polygons
    grid_out["geometry"] = grid_out["geometry"].buffer(0) # fix invalid geoms
    grid_out = grid_out[~grid_out.geometry.is_empty]      # drop empty geoms
    grid_out = grid_out.reset_index(drop=True)            # reset AFTER all geometry ops
    grid_out["id"] = grid_out.index.astype(str)

    with rasterio.open(pop_raster_path) as src:
        raster_crs = src.crs
        grid_raster_crs = grid_out.to_crs(raster_crs)

        # explode + clean the reprojected copy too for rasterio masking
        grid_raster_crs = grid_raster_crs.explode(index_parts=False).reset_index(drop=True)
        grid_raster_crs["geometry"] = grid_raster_crs["geometry"].buffer(0)
        grid_raster_crs = grid_raster_crs[~grid_raster_crs.geometry.is_empty]

        from shapely.geometry import mapping
        geoms = [mapping(geom) for geom in grid_raster_crs.geometry]

        try:
            out_image, out_transform = rasterio.mask.mask(src, geoms, crop=True)
        except Exception as e:
            raise ValueError(f"Raster masking failed: {e}")

        nodata = src.nodata
        res_x = abs(out_transform.a)
        res_y = abs(out_transform.e)

    arr = out_image[0].astype(float) # apply mask 
    if nodata is not None:
        arr[arr == nodata] = np.nan

    rows, cols = np.where((~np.isnan(arr)) & (arr > 0))

    if len(rows) == 0: # data validity check
        print("Warning: No valid population values found in raster within grid extent.")
        grid_out["population"] = 0.0
        return grid_out

    xs, ys = rasterio.transform.xy(out_transform, rows, cols)
    pop_values = arr[rows, cols]

    # build rectangular cell polygons around each raster cell centre
    cell_polygons = [
        box(x - res_x / 2, y - res_y / 2, x + res_x / 2, y + res_y / 2)
        for x, y in zip(xs, ys)
    ]
    raster_cells = gpd.GeoDataFrame(
        {"cell_pop": pop_values},
        geometry=cell_polygons,
        crs=raster_crs
    )

    # --- 3) Reproject raster cells to grid CRS ---
    raster_cells = raster_cells.to_crs(grid_out.crs)
    raster_cells["cell_area"] = raster_cells.geometry.area

    # --- 4) Intersect raster cells with hex polygons ---
    intersection = gpd.overlay(
        raster_cells[["cell_pop", "cell_area", "geometry"]],
        grid_out[["id", "geometry"]],
        how="intersection"
    )

    if intersection.empty:
        print("Warning: No intersection between raster cells and grid.")
        grid_out["population"] = 0.0
        return grid_out

    # --- 5) Proportional allocation ---
    intersection["int_area"] = intersection.geometry.area
    intersection["pop_share"] = intersection["int_area"] / intersection["cell_area"]
    intersection["pop_allocated"] = intersection["cell_pop"] * intersection["pop_share"]

    hex_pop = (
        intersection.groupby("id")["pop_allocated"]
        .sum()
        .reset_index()
        .rename(columns={"pop_allocated": "population"})
    )

    # --- 6) Merge back onto original grid ---
    grid_out = grid_out.merge(hex_pop, on="id", how="left")
    grid_out["population"] = grid_out["population"].fillna(0.0)

    # --- 7) Validate totals ---
    raster_total = float(pop_values.sum())
    hex_total = float(grid_out["population"].sum())
    print(f"Raster total population:  {raster_total:,.0f}")
    print(f"Hex grid total population: {hex_total:,.0f}")
    print(f"Coverage: {hex_total / raster_total * 100:.1f}%")

    # --- 8) Optional plot ---
    if plot:
        fig, ax = plt.subplots(1, 1, figsize=(14, 12))
        grid_out.plot(
            column="population",
            cmap="YlOrRd",
            scheme="Quantiles",
            k=9,
            legend=True,
            linewidth=0.02,
            edgecolor="black",
            legend_kwds={
                "loc": "lower right",
                "title": "Population\n(9 quantiles)",
                "fmt": "{:.0f}"
            },
            ax=ax
        )
        ax.set_title("Population by Hexagon (areal disaggregation)", fontsize=18)
        ax.set_axis_off()
        plt.tight_layout()
        plt.show()

    return grid_out

def create_origins(grid, grid_pop=None):
    """
    Creates a GeoDataFrame of origin points based on hex grid

    ----------
    grid : GeoDataFrame
        Polygon hex grid (any CRS)
    grid_pop : GeoDataFrame, optional
        Population-annotated grid
        (If provided, attaches a 'population' column to each origin)

    Returns: GeoDataFrame with columns:
    -------
        - 'id'         
        - 'geometry'   : Point at hex centroid (WGS84)
        - 'population' : hex population
    """

    origins = grid.copy().reset_index(drop=True)
    origins["id"] = origins.index.astype(str)

    utm_crs = origins.estimate_utm_crs()    # project to UTM for accurate centroid calculation
    origins_utm = origins.to_crs(utm_crs)   # accounting for earth's curvature
    origins_utm["geometry"] = origins_utm.geometry.centroid

    origins = origins_utm.to_crs("EPSG:4326")   # # convert back to WGS84 (must be flat) for r5py 

    if grid_pop is not None:
        grid_pop_copy = grid_pop.copy()
        grid_pop_copy["id"] = grid_pop_copy.index.astype(str)
        origins = origins.merge(
            grid_pop_copy[["id", "population"]],
            on="id",
            how="left"
        )
        origins["population"] = origins["population"].fillna(0.0)

    return origins

def create_POIs(osm, summary=False):
    destinations = osm.get_pois(custom_filter={'amenity': True})    # filter by check in column

    noise_categories = [
    'parking', 'bicycle_parking', 'bench', 'waste_basket', 'recycling', 'waste_disposal',
    'vending_machine', 'parking_entrance', 'post_box', 'hunting_stand', 'parking_space',
    'car_sharing', 'tourist_bus_parking', 'waste_transfer_station'
    ]

    destinations = destinations[~destinations['amenity'].isin(noise_categories)] # filter noise

    # drop rows with missing or invalid geometry
    destinations = destinations[destinations.geometry.notna()]
    destinations = destinations[~destinations.geometry.is_empty]

    domain_mapping = {
    # Food & Drink
    'restaurant': 'food_and_drink', 'cafe': 'food_and_drink', 'bbq': 'food_and_drink',
    'fast_food': 'food_and_drink', 'bar': 'food_and_drink', 'pub': 'food_and_drink',
    'food_court': 'food_and_drink',
    
    # Healthcare
    'hospital': 'healthcare', 'clinic': 'healthcare', 'social_facility': 'healthcare',
    'pharmacy': 'healthcare', 'doctors': 'healthcare', 'dentist': 'healthcare',
    
    # Education
    'school': 'education', 'kindergarten': 'education', 'library': 'education',
    'university': 'education', 'college': 'education', 'childcare': 'education',
    
    # Social / community 
    'community_centre': 'community', 'events_venue': 'community', 
    'place_of_worship': 'community', 

    # entertainment
    'theatre': 'entertainment', 'cinema': 'entertainment', "arts_centre": 'entertainment',

    
    # Essential Services
    'bank': 'essential_services', 'post_office': 'essential_services', 'telephone': 'essential_services',
    'police': 'essential_services', 'fire_station': 'essential_services', 'atm': 'essential_services'
    }

    destinations['amenity class'] = destinations['amenity'].replace(domain_mapping)

    utm_crs = destinations.estimate_utm_crs()
    destinations_utm = destinations.to_crs(utm_crs)
    destinations_utm["geometry"] = destinations_utm.geometry.centroid
    destinations = destinations_utm.to_crs("EPSG:4326")

    destinations = destinations.reset_index(drop=True)  # adding useful columns
    destinations["id"] = destinations.index.astype(str)
    destinations["opp_weight"] = 1

    keep_cols = ["id", "geometry", "amenity", "amenity class", "opp_weight"]

    if summary:
        print(f"Total amenities: {len(destinations)}")
        print("\nBy type:")
        print(destinations["amenity class"].value_counts().head(10).to_string())

    return destinations[keep_cols]

def suggested_beta(cutoff_minutes, target_weight=0.5):
    """
    Returns beta such that an amenity at cutoff_minutes retains target_weight of its value.
    e.g. cutoff=30, target_weight=0.1 → amenity at 30min = 10% weight
    """
    return -np.log(target_weight) / cutoff_minutes

def compute_gravity_scores(travel_time_matrix, destinations, beta=0.08):
    tt = travel_time_matrix.pivot(index="from_id", columns="to_id", values="travel_time").reset_index()
    tt.columns.name = None
    value_cols = [c for c in tt.columns if c != "from_id"]

    tt[value_cols] = tt[value_cols].apply(pd.to_numeric, errors="coerce")
    decayed = np.exp(-beta * tt[value_cols]).fillna(0)

    result = tt[["from_id"]].copy()
    for amenity_class, group in destinations.groupby("amenity class"):
        dest_ids = set(group["id"].tolist())
        matching_cols = [c for c in decayed.columns if c in dest_ids]

        col_name = f"gravity_{amenity_class.lower().replace(' ', '_')}"
        result[col_name] = decayed[matching_cols].sum(axis=1) if matching_cols else 0

    # clean up pivot index name and ensure from_id is first column
    result = result.rename(columns={"to_id": "from_id"}).reset_index(drop=True)
    result = result[["from_id"] + [c for c in result.columns if c.startswith("gravity_")]]

    # add average gravity score across all categories
    gravity_cols = [c for c in result.columns if c.startswith("gravity_")]
    result["avg_gravity"] = result[gravity_cols].mean(axis=1)

    return result

def min_travel_time(travel_time_matrix, destinations):
    # excluding 0 from min. calculation, i.e. in case that no amenity was reachable from given points of origin < threshold
    travel_time_matrix["travel_time"] = travel_time_matrix["travel_time"].replace(0, float("nan"))

    ttm = travel_time_matrix.merge(
        destinations[["id", "amenity class"]],
        left_on="to_id",
        right_on="id",
        how="left"
    )

    # aggregate min and mean
    stats = (
        ttm.groupby(["from_id", "amenity class"])["travel_time"]
        .agg(['min', 'mean'])
        .reset_index()
    )

    # pivot
    stats_wide = stats.pivot(
        index="from_id",
        columns="amenity class",
        values=["min", "mean"]
    )

    # rename columns to min/avg_tt_{category}
    stats_wide.columns = [
        f"{stat}_tt_{cat}" if stat == 'min' else f"avg_tt_{cat}"
        for stat, cat in stats_wide.columns
    ]
    stats_wide = stats_wide.reset_index()

    # global averages
    avg_tt_cols = [c for c in stats_wide.columns if c.startswith("avg_tt_")]
    stats_wide["avg_tt_all"]     = stats_wide[avg_tt_cols].mean(axis=1)

    # make sure to distinguish unreachable destinations (since this is skipped over when computing avg.)
    stats_wide["n_unreachable_categories"] = stats_wide[avg_tt_cols].isna().sum(axis=1)
    return stats_wide

def weight_pop(results, origins, pop_col="population", score_cols=None, normalisation=100, return_full=False):
    """
    normalisation : score per X inhabitants
    return_full : If True, returns full results df with weighted scores appended.
    """
    if score_cols is None:
        score_cols = [c for c in results.columns if c.startswith("gravity_")]
    merged = results[["from_id"] + score_cols].merge(
        origins[["id", "population"]],
        left_on="from_id",
        right_on="id",
        how="left"
    )
    for col in score_cols:  # compute per capita scores (score / pop) * norm_factor
        weighted_col = f"{col}_per_{normalisation}"
        merged[weighted_col] = (merged[col] / merged["population"]) * normalisation

    weighted_cols = [f"{c}_per_{normalisation}" for c in score_cols]

    merged[weighted_cols] = merged[weighted_cols].replace([np.nan, np.inf, -np.inf], 0)

    if return_full:
        return results.merge(merged[["from_id"] + weighted_cols], on="from_id", how="left")
    else:
        return merged[["from_id"] + weighted_cols]

def compute_diversity_scores(results, count_cols, id_col="from_id"):
    """
    Computes amenity diversity scores per hex relative to city-wide distribution.

        - shannon          : standard Shannon diversity (uniform baseline)
        - pielou_j         : Shannon normalised by log(k)
        - jsd_from_city    : Jensen-Shannon divergence from city-wide distribution
        - cross_entropy    : KL divergence from city-wide distribution
    """
    df = results[[id_col] + count_cols].copy()

    city_totals = df[count_cols].sum()
    q = (city_totals / city_totals.sum()).values  # city-wide proportions
    print("City-wide baseline distribution:")
    for col, val in zip(count_cols, q):
        print(f"  {col}: {val:.3f}")

    # --- per-hex proportions ---
    row_totals = df[count_cols].sum(axis=1)
    # avoid division by zero for hexes with no amenities
    proportions = df[count_cols].div(row_totals.replace(0, np.nan), axis=0).fillna(0)

    k = len(count_cols)
    epsilon = 1e-10  # small value to avoid log(0)

    scores = df[[id_col]].copy()
    # standard Shannon
    p = proportions.values + epsilon
    scores["shannon"] = -(p * np.log(p)).sum(axis=1)
    # KL divergence from city-wide distribution (cross-entropy)
    scores["kl_divergence"] = proportions.apply(
        lambda row: entropy(row.values + epsilon, q + epsilon),
        axis=1
    )
    # JSD from city-wide distribution
    scores["jsd"] = proportions.apply(
        lambda row: jensenshannon(row.values + epsilon, q + epsilon),
        axis=1
    )
    # similarity to city (1 - JSD, so higher = more like city baseline)
    scores["city_similarity"] = 1 - scores["jsd"]

    return scores

def merge_geometry(results, grid, from_id_col="from_id"):
    grid_indexed = grid[["geometry"]].copy().reset_index(drop=True)
    grid_indexed["_id"] = grid_indexed.index.astype(str)

    results_copy = results.copy()
    results_copy[from_id_col] = results_copy[from_id_col].astype(str)

    merged = results_copy.merge(
        grid_indexed,
        left_on=from_id_col,
        right_on="_id",
        how="left"
    ).drop(columns="_id")

    return gpd.GeoDataFrame(merged, geometry="geometry", crs=grid.crs)

def plot_hex_score(results_geo, score_col, from_id_col="from_id", scheme="Quantiles",
                   k=9, cmap="viridis", title=None, figsize=(14, 12), legend_fmt="{:.1f}"):

    n_missing = results_geo[score_col].isna().sum()
    if n_missing > 0:
        print(f"Note: {n_missing} hexes have NaN for '{score_col}' and will appear grey.")

    fig, ax = plt.subplots(1, 1, figsize=figsize)

    results_geo.plot(
        column=score_col,
        cmap=cmap,
        scheme=scheme,
        k=k,
        legend=True,
        linewidth=0.02,
        edgecolor="none",
        missing_kwds={"color": "lightgrey", "label": "No data"},
        legend_kwds={
            "loc": "lower right",
            "title": score_col.replace("_", " ").title(),
            "fmt": legend_fmt
        },
        ax=ax
    )

    ax.set_title(title if title else score_col.replace("_", " ").title(), fontsize=18)
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()


#---------------------------OTHER---------------------------------
def clear_cache():
    cache_dir = Config().CACHE_DIR
    print(f"Deleting cache at: {cache_dir}")

    shutil.rmtree(cache_dir)

    print("Cache cleared.")

def build_city_geojson(
    grid,
    grid_pop,
    osm,
    tn,
    output_path=None,
    transport_modes=None,
    max_time=30,
    beta=0.08,
    gravity_normalisation=100,
):
    """
    Runs the complete accessibility metrics pipeline on given grid + osm + travel network

    output_path : saves the output as GeoJSON
    transport_modes : list of r5py transport modes (defaults to [TransportMode.TRANSIT, TransportMode.WALK])
    max_time : travel time in minutes
    beta : decay parameter for gravity calculation
    gravity_normalisation : population normalisation factor for weighted scores

    Returns: GeoDataFrame: Fully scored hex grid with all accessibility metrics attached.
    """

    if transport_modes is None:
        transport_modes = [r5.TransportMode.TRANSIT, r5.TransportMode.WALK]

    # --- Origins and destinations from grid and OSM ---
    categories = ['food_and_drink','shelter','essential_services','education','healthcare','community']
    origins = create_origins(grid, grid_pop=grid_pop)
    destinations = create_POIs(osm)
    destinations = destinations[destinations["amenity class"].isin(categories)].copy()

    # --- Travel time matrix via r5py ---
    ttm = r5.TravelTimeMatrix(
        tn,
        origins=origins,
        destinations=destinations,
        transport_modes=transport_modes,
        max_time=timedelta(minutes=max_time),
    )

    # --- Gravity scores (exponential decay per amenity category) ---
    gravity = compute_gravity_scores(ttm.copy(), destinations, beta=beta)

    # --- Min + average travel time per amenity category ---
    tt_stats = min_travel_time(ttm.copy(), destinations)

    # --- Basic counts: gravity with beta=0 equals a simple reachable amenity count ---
    counts_raw = compute_gravity_scores(ttm.copy(), destinations, beta=0)
    counts_wide = counts_raw.rename(columns={
        c: c.replace("gravity_", "count_") for c in counts_raw.columns if c.startswith("gravity_")
    })
    counts_wide = counts_wide.drop(columns=["avg_gravity"], errors="ignore")

    # --- Population-weighted gravity (score per N inhabitants) ---
    gravity_cols = [c for c in gravity.columns if c.startswith("gravity_")]
    pop_weighted = weight_pop(
        gravity,
        origins,
        score_cols=gravity_cols,
        normalisation=gravity_normalisation,
        return_full=False,
    )

    # --- Diversity scores (Shannon, JSD, KL divergence, city similarity) ---
    count_cols = [c for c in counts_wide.columns if c.startswith("count_")]
    diversity = compute_diversity_scores(counts_wide, count_cols=count_cols, id_col="from_id")

    # --- Merge all score dataframes on from_id ---
    result = origins[["id", "population"]].rename(columns={"id": "from_id"}).copy()
    for df in [gravity, tt_stats, counts_wide, pop_weighted, diversity]:
        df["from_id"] = df["from_id"].astype(str)
        result = result.merge(df, on="from_id", how="left")

    # --- Attach hex geometry back to results ---
    result_geo = merge_geometry(result, grid)

    # Ensure GeoJSON-friendly CRS84 / EPSG:4326 for web mapping
    if result_geo.crs is None:
        result_geo = result_geo.set_crs("EPSG:4326")
    else:
        result_geo = result_geo.to_crs("EPSG:4326")
    if output_path:
        result_geo.to_file(output_path, driver="GeoJSON")

    return result_geo