import rasterio
import rasterio.mask
import r5py as r5
import h3
import geopandas as gpd
from shapely.geometry import Polygon, box
from pyrosm import OSM
import numpy as np
from pyrosm.data import sources
from pyrosm import get_data
import matplotlib.pyplot as plt
import pandas as pd
import shutil
from r5py.util.config import Config

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
    grid_out = grid.copy().reset_index(drop=True)
    grid_out["id"] = grid_out.index.astype(str)

    with rasterio.open(pop_raster_path) as src:
        raster_crs = src.crs
        grid_raster_crs = grid_out.to_crs(raster_crs) # reproject grid to raster CRS for clipping
        
        geoms = list(grid_raster_crs.geometry.values) # clip raster to grid boundaries
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

def create_POIs(osm):
    destinations = osm.get_pois(custom_filter={'amenity': True})    # filter by check in column
     


#---------------------------OTHER---------------------------------
def clear_cache():
    cache_dir = Config().CACHE_DIR
    print(f"Deleting cache at: {cache_dir}")

    shutil.rmtree(cache_dir)

    print("Cache cleared.")