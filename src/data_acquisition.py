"""
Data Acquisition Module
------------------------
This module handles:
1. Google Earth Engine (GEE) Python API template functions for downloading real Sentinel-1 and Sentinel-2 data.
2. A high-fidelity, offline Synthetic Data Generator that produces aligned geospatial datasets (GeoJSON vector fields and multi-band GeoTIFF rasters)
   with realistic spectral responses for different crops, moisture conditions, and growth stages.

Libraries Used:
- rasterio: For writing geospatial multi-band GeoTIFF images.
- geopandas: For creating and managing spatial vector fields (shapefiles/GeoJSON).
- shapely.geometry: For constructing polygon boundaries of agricultural fields.
- numpy: For matrix operations, adding spatial noise, and generating spectral curves.
"""

import os
import json
import numpy as np
import pandas as pd
import geopandas as gpd
from shapely.geometry import Polygon
import rasterio
from rasterio.transform import from_origin

# ==========================================
# PART 1: Google Earth Engine (GEE) Templates
# ==========================================
# (These functions show how real ISRO/ESA satellite imagery is downloaded via GEE)

def initialize_earth_engine():
    """
    Initializes and authenticates the Google Earth Engine Python API.
    Raises an exception if authentication fails.
    """
    try:
        import ee
        print("[GEE] Attempting to initialize Earth Engine...")
        ee.Initialize()
        print("[GEE] Earth Engine initialized successfully.")
        return True
    except Exception as e:
        print("[GEE WARNING] GEE initialization failed. Ensure you have run 'ee.Authenticate()' in your command line.")
        print(f"[GEE ERROR DETAILS] {e}")
        print("[GEE NOTE] Falling back to the local high-fidelity data generator for development.")
        return False

def get_real_sentinel2_collection(roi, start_date, end_date):
    """
    Template showing how to retrieve cloud-masked Sentinel-2 Level-2A imagery.
    
    Parameters:
    - roi: ee.Geometry representing Region of Interest
    - start_date: str (YYYY-MM-DD)
    - end_date: str (YYYY-MM-DD)
    """
    try:
        import ee
        
        # Cloud masking function for Sentinel-2 QA band
        def mask_s2_clouds(image):
            qa = image.select('QA60')
            cloud_bit_mask = 1 << 10
            cirrus_bit_mask = 1 << 11
            mask = qa.bitwiseAnd(cloud_bit_mask).eq(0).And(
                   qa.bitwiseAnd(cirrus_bit_mask).eq(0))
            return image.updateMask(mask).divide(10000) # Scale reflectance to 0-1

        collection = (ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED')
                      .filterBounds(roi)
                      .filterDate(start_date, end_date)
                      .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20))
                      .map(mask_s2_clouds)
                      .select(['B2', 'B3', 'B4', 'B8', 'B11', 'B12'])) # Blue, Green, Red, NIR, SWIR1, SWIR2
        return collection
    except Exception as e:
        print(f"[GEE Error] Could not query Sentinel-2 collection: {e}")
        return None


# ==========================================
# PART 2: Local High-Fidelity Data Generator
# ==========================================

class SatelliteDataSimulator:
    """
    Simulates aligned Sentinel-1 (Radar) and Sentinel-2 (Optical) data.
    Generates agricultural fields as a GeoJSON file and temporal sequences of rasters
    that realistically simulate crop growth cycles, chlorophyll content, and water stress.
    """
    def __init__(self, output_dir="data", base_lat=16.50, base_lon=80.50, grid_size=300):
        """
        Parameters:
        - output_dir: Directory where datasets will be saved.
        - base_lat, base_lon: Center coordinates for the simulation grid (e.g., Andhra Pradesh fields).
        - grid_size: Dimensions of the raster grid (300x300 pixels). At 10m resolution, this is 3x3 km.
        """
        self.output_dir = output_dir
        self.raw_dir = os.path.join(output_dir, "raw")
        os.makedirs(self.raw_dir, exist_ok=True)
        
        self.base_lat = base_lat
        self.base_lon = base_lon
        self.grid_size = grid_size
        self.pixel_size_deg = 0.0001  # Approximately 10-11 meters resolution per pixel (matches Sentinel-2 resolution)
        
        # Spatial extent calculation
        self.west = base_lon - (grid_size / 2) * self.pixel_size_deg
        self.east = base_lon + (grid_size / 2) * self.pixel_size_deg
        self.north = base_lat + (grid_size / 2) * self.pixel_size_deg
        self.south = base_lat - (grid_size / 2) * self.pixel_size_deg
        
        # Define affine transformation matrix for geo-referencing the raster
        # (maps pixel column/row coordinates to longitude/latitude coordinates)
        self.transform = from_origin(self.west, self.north, self.pixel_size_deg, self.pixel_size_deg)
        self.crs = "EPSG:4326"  # WGS 84 Coordinate Reference System
        
        # Crop parameters database
        # Defines base index profiles: (NDVI, NDWI, VV, VH) for Sowing, Vegetative, Flowering, Maturity stages
        self.crop_profiles = {
            "Rice": {
                "Sowing":     {"ndvi": 0.15, "ndwi": 0.70, "vv": -16.0, "vh": -25.0},
                "Vegetative": {"ndvi": 0.55, "ndwi": 0.40, "vv": -12.0, "vh": -20.0},
                "Flowering":  {"ndvi": 0.78, "ndwi": 0.30, "vv": -9.0,  "vh": -15.0},
                "Maturity":   {"ndvi": 0.45, "ndwi": 0.15, "vv": -11.0, "vh": -18.0}
            },
            "Wheat": {
                "Sowing":     {"ndvi": 0.12, "ndwi": 0.15, "vv": -12.0, "vh": -20.0},
                "Vegetative": {"ndvi": 0.60, "ndwi": 0.25, "vv": -10.0, "vh": -16.0},
                "Flowering":  {"ndvi": 0.82, "ndwi": 0.20, "vv": -8.0,  "vh": -12.0},
                "Maturity":   {"ndvi": 0.35, "ndwi": 0.05, "vv": -11.0, "vh": -17.0}
            },
            "Cotton": {
                "Sowing":     {"ndvi": 0.10, "ndwi": 0.10, "vv": -14.0, "vh": -22.0},
                "Vegetative": {"ndvi": 0.48, "ndwi": 0.20, "vv": -11.0, "vh": -17.0},
                "Flowering":  {"ndvi": 0.72, "ndwi": 0.15, "vv": -8.5,  "vh": -13.0},
                "Maturity":   {"ndvi": 0.50, "ndwi": 0.08, "vv": -10.0, "vh": -15.0}
            },
            "Maize": {
                "Sowing":     {"ndvi": 0.14, "ndwi": 0.18, "vv": -13.0, "vh": -21.0},
                "Vegetative": {"ndvi": 0.65, "ndwi": 0.30, "vv": -9.5,  "vh": -15.0},
                "Flowering":  {"ndvi": 0.80, "ndwi": 0.22, "vv": -7.5,  "vh": -11.5},
                "Maturity":   {"ndvi": 0.40, "ndwi": 0.10, "vv": -10.5, "vh": -16.5}
            },
            "Sugarcane": {
                "Sowing":     {"ndvi": 0.18, "ndwi": 0.22, "vv": -12.0, "vh": -19.0},
                "Vegetative": {"ndvi": 0.50, "ndwi": 0.28, "vv": -10.0, "vh": -15.0},
                "Flowering":  {"ndvi": 0.70, "ndwi": 0.24, "vv": -8.0,  "vh": -13.0},
                "Maturity":   {"ndvi": 0.62, "ndwi": 0.18, "vv": -9.0,  "vh": -14.0}
            }
        }

    def generate_field_boundaries(self):
        """
        Creates vector boundaries for 6 agricultural fields layout in a grid within the ROI,
        assigning crop types, sowing dates, growth stages, and soil moisture stress conditions.
        Saves them as a GeoJSON file.
        """
        print("[Simulator] Generating agricultural fields layout...")
        fields = []
        
        # Grid parameters for 6 fields (3 columns x 2 rows)
        x_slices = np.linspace(self.west + 0.002, self.east - 0.002, 4)
        y_slices = np.linspace(self.south + 0.002, self.north - 0.002, 3)
        
        crop_types = ["Rice", "Wheat", "Cotton", "Maize", "Sugarcane", "Rice"]
        # Distribute growth stages and stress levels to get a rich variety in testing
        growth_stages = ["Vegetative", "Flowering", "Sowing", "Maturity", "Vegetative", "Flowering"]
        stress_levels = ["No Stress", "Mild Stress", "Severe Stress", "No Stress", "Mild Stress", "No Stress"]
        sowing_dates = ["2026-04-15", "2026-05-01", "2026-06-10", "2026-03-20", "2026-04-20", "2026-05-15"]
        
        idx = 0
        for i in range(2):  # rows
            for j in range(3):  # cols
                # Create coordinate boundaries with some spacing
                w = x_slices[j] + 0.0005
                e = x_slices[j+1] - 0.0005
                s = y_slices[i] + 0.0005
                n = y_slices[i+1] - 0.0005
                
                # Polygon coordinates: clockwise order
                coords = [(w, n), (e, n), (e, s), (w, s), (w, n)]
                polygon = Polygon(coords)
                
                field_data = {
                    "geometry": polygon,
                    "field_id": int(idx + 1),
                    "crop_type": crop_types[idx],
                    "sowing_date": sowing_dates[idx],
                    "growth_stage": growth_stages[idx],
                    "stress_level": stress_levels[idx],
                    "area_ha": float(polygon.area * 111120 * 111120) / 10000.0  # approximate conversion of sq degrees to hectares
                }
                fields.append(field_data)
                idx += 1
                
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(fields, crs=self.crs)
        geojson_path = os.path.join(self.raw_dir, "fields.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"[Simulator] Saved field boundaries to: {geojson_path}")
        return gdf

    def _generate_raster_band_values(self, fields_gdf, target_date, satellite_type="Sentinel-2"):
        """
        Creates numpy arrays representing pixel-level spectral bands based on fields boundaries and crop characteristics.
        
        Parameters:
        - fields_gdf: GeoDataFrame containing fields vector data
        - target_date: str (YYYY-MM-DD)
        - satellite_type: "Sentinel-1" (Radar) or "Sentinel-2" (Optical)
        """
        h, w = self.grid_size, self.grid_size
        
        # Base values (background non-agriculture soil/shrub)
        if satellite_type == "Sentinel-2":
            # 6 bands: Blue, Green, Red, NIR, SWIR1, SWIR2
            bands = {
                "Blue": np.random.normal(0.08, 0.01, (h, w)),
                "Green": np.random.normal(0.10, 0.01, (h, w)),
                "Red": np.random.normal(0.12, 0.01, (h, w)),
                "NIR": np.random.normal(0.16, 0.02, (h, w)),
                "SWIR1": np.random.normal(0.24, 0.02, (h, w)),
                "SWIR2": np.random.normal(0.20, 0.02, (h, w))
            }
        else:
            # 2 bands: VV, VH (Radar backscatter in dB scale)
            bands = {
                "VV": np.random.normal(-13.0, 0.5, (h, w)),
                "VH": np.random.normal(-21.0, 0.5, (h, w))
            }
            
        # Add smooth geospatial noise (simulating terrain/background soil variability)
        y, x = np.meshgrid(np.linspace(0, 3, h), np.linspace(0, 3, w))
        spatial_noise = np.sin(x) * np.cos(y) * 0.02
        
        for key in bands:
            if satellite_type == "Sentinel-2":
                bands[key] += spatial_noise
            else:
                bands[key] += spatial_noise * 10  # more noise scaling in radar dB
                
        # Rasterize crop values: for each pixel falling inside a field polygon, update its band values
        for _, row in fields_gdf.iterrows():
            poly = row["geometry"]
            crop = row["crop_type"]
            stage = row["growth_stage"]
            stress = row["stress_level"]
            
            # Fetch target values from profile
            profile = self.crop_profiles[crop][stage]
            ndvi_val = profile["ndvi"]
            ndwi_val = profile["ndwi"]
            vv_val = profile["vv"]
            vh_val = profile["vh"]
            
            # Apply adjustments for moisture stress level
            # Severe moisture stress drops NDVI (chlorophyll decay) and significantly drops NDWI & SAR VV/VH backscatter (loss of canopy water)
            if stress == "Mild Stress":
                ndvi_val -= 0.08
                ndwi_val -= 0.12
                vv_val -= 1.0
                vh_val -= 1.5
            elif stress == "Severe Stress":
                ndvi_val -= 0.18
                ndwi_val -= 0.28
                vv_val -= 2.5
                vh_val -= 3.5
                
            # Keep indices inside physical bounds
            ndvi_val = max(0.05, ndvi_val)
            ndwi_val = max(-0.1, ndwi_val)
            
            # Convert indices (NDVI, NDWI) back to physical band reflectance values for Sentinel-2
            # NDVI = (NIR - Red) / (NIR + Red)
            # NDWI = (NIR - SWIR1) / (NIR + SWIR1)
            # We solve these equations to get Red, NIR, SWIR1 values relative to Blue & Green
            if satellite_type == "Sentinel-2":
                base_ref = 0.1  # base scaling
                # Calculate Red and NIR to satisfy NDVI
                red = base_ref * (1.0 - ndvi_val) / 2.0
                nir = base_ref * (1.0 + ndvi_val) / 2.0
                
                # Settle Green to represent vegetation chlorophyll reflection (high NDVI = higher Green)
                green = red * 1.1 + (ndvi_val * 0.12)
                blue = red * 0.8
                
                # Calculate SWIR1 to satisfy NDWI
                # NDWI = (NIR - SWIR1) / (NIR + SWIR1) => SWIR1 = NIR * (1 - NDWI) / (1 + NDWI)
                swir1 = nir * (1.0 - ndwi_val) / (1.0 + ndwi_val)
                # SWIR2 tracks SWIR1 but is generally slightly lower for crop foliage
                swir2 = swir1 * 0.85
                
                field_bands = {
                    "Blue": blue, "Green": green, "Red": red, 
                    "NIR": nir, "SWIR1": swir1, "SWIR2": swir2
                }
            else:
                field_bands = {
                    "VV": vv_val, "VH": vh_val
                }

            # Map the polygon coordinates to index masks in the grid
            for r in range(h):
                lat = self.north - r * self.pixel_size_deg
                for c in range(w):
                    lon = self.west + c * self.pixel_size_deg
                    # Check if pixel coordinate is inside polygon using a fast bounding box check + geometry check
                    if poly.contains(Polygon([(lon, lat), (lon+self.pixel_size_deg, lat), 
                                              (lon+self.pixel_size_deg, lat-self.pixel_size_deg), 
                                              (lon, lat-self.pixel_size_deg)])):
                        for band_name in bands:
                            # Assign value with slight intra-field pixel variation (standard deviation = 3%)
                            val = field_bands[band_name]
                            if satellite_type == "Sentinel-2":
                                bands[band_name][r, c] = np.random.normal(val, val * 0.03)
                            else:
                                bands[band_name][r, c] = np.random.normal(val, 0.2)
                                
        return bands

    def generate_satellite_images(self, fields_gdf, dates_list):
        """
        Creates and writes temporal Sentinel-1 and Sentinel-2 rasters to TIFF files.
        
        Parameters:
        - fields_gdf: GeoDataFrame containing fields boundaries.
        - dates_list: list of str dates (e.g., ["2026-06-01", "2026-06-15"])
        """
        generated_files = []
        for dt in dates_list:
            print(f"[Simulator] Synthesizing imagery for target date: {dt}...")
            
            # --- 1. Sentinel-2 (Optical Multi-band) ---
            s2_bands = self._generate_raster_band_values(fields_gdf, dt, "Sentinel-2")
            s2_path = os.path.join(self.raw_dir, f"sentinel2_{dt}.tif")
            
            # Write 6-band S2 GeoTIFF
            with rasterio.open(
                s2_path, 'w',
                driver='GTiff',
                height=self.grid_size,
                width=self.grid_size,
                count=6,
                dtype=rasterio.float32,
                crs=self.crs,
                transform=self.transform,
            ) as dst:
                band_order = ["Blue", "Green", "Red", "NIR", "SWIR1", "SWIR2"]
                for i, b_name in enumerate(band_order, start=1):
                    # Save bands as float32 reflectance
                    dst.write(s2_bands[b_name].astype(np.float32), i)
                    # Label band descriptions
                    dst.update_tags(i, name=b_name)
                    
            print(f"  -> Created Sentinel-2 (6 bands): {s2_path}")
            generated_files.append(s2_path)
            
            # --- 2. Sentinel-1 (Radar Dual-band VV/VH) ---
            s1_bands = self._generate_raster_band_values(fields_gdf, dt, "Sentinel-1")
            s1_path = os.path.join(self.raw_dir, f"sentinel1_{dt}.tif")
            
            # Write 2-band S1 GeoTIFF
            with rasterio.open(
                s1_path, 'w',
                driver='GTiff',
                height=self.grid_size,
                width=self.grid_size,
                count=2,
                dtype=rasterio.float32,
                crs=self.crs,
                transform=self.transform,
            ) as dst:
                band_order_s1 = ["VV", "VH"]
                for i, b_name in enumerate(band_order_s1, start=1):
                    dst.write(s1_bands[b_name].astype(np.float32), i)
                    dst.update_tags(i, name=b_name)
                    
            print(f"  -> Created Sentinel-1 (2 bands): {s1_path}")
            generated_files.append(s1_path)
            
        return generated_files


if __name__ == "__main__":
    # Test script execution
    print("=== Testing Data Acquisition Module ===")
    
    # Try GEE initialization
    gee_success = initialize_earth_engine()
    
    # Run simulation
    simulator = SatelliteDataSimulator(output_dir="data")
    fields = simulator.generate_field_boundaries()
    
    # Generate temporal observations (a 4-step observation series over 2 months)
    observation_dates = ["2026-05-01", "2026-05-15", "2026-06-01", "2026-06-15"]
    files = simulator.generate_satellite_images(fields, observation_dates)
    
    print(f"\n[Success] Simulation complete. Created {len(files)} raster files and fields GeoJSON.")
