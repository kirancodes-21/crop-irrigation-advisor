"""
Preprocessing & Spatial Processing Module
------------------------------------------
This module is responsible for loading raw Sentinel-1 and Sentinel-2 raster files,
calculating vegetation and water indices, and saving the preprocessed bands.

Calculated Spectral Indices:
1. NDVI (Normalized Difference Vegetation Index):
   Formula: (NIR - Red) / (NIR + Red)
   Purpose: Measures greenness, leaf area index, and overall photosynthetic activity.
   
2. NDWI (Normalized Difference Water Index):
   Formula: (NIR - SWIR1) / (NIR + SWIR1)
   Purpose: Measures liquid water content in the vegetation canopy. Sensitive to moisture stress.
   
3. MSI (Moisture Stress Index):
   Formula: SWIR1 / NIR
   Purpose: Higher values represent higher water stress. Very useful for crop drought monitoring.

4. Radar Ratio (VV/VH):
   Formula: VV - VH (in dB scale) or VV / VH (in linear scale)
   Purpose: Normalizes soil roughness effects to track vegetation moisture and growth canopy density.

Libraries Used:
- rasterio: To read and write geospatial raster bands while preserving spatial alignment.
- numpy: For fast, element-wise grid computations.
"""

import os
import numpy as np
import rasterio

class GeospatialPreprocessor:
    """
    Handles raster operations: reads bands, computes physical indices, 
    manages coordinate reference systems (CRS), and writes processed bands.
    """
    def __init__(self, raw_dir="data/raw", processed_dir="data/processed"):
        self.raw_dir = raw_dir
        self.processed_dir = processed_dir
        os.makedirs(self.processed_dir, exist_ok=True)

    def calculate_ndvi(self, red_band, nir_band):
        """
        Computes NDVI. Handles divide-by-zero by setting output to 0.0 where denominator is zero.
        """
        # Suppress divide-by-zero warnings since we handle them explicitly
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = nir_band + red_band
            ndvi = np.where(denominator != 0, (nir_band - red_band) / denominator, 0.0)
            # Clip values to valid physical range [-1.0, 1.0]
            ndvi = np.clip(ndvi, -1.0, 1.0)
        return ndvi

    def calculate_ndwi(self, nir_band, swir1_band):
        """
        Computes NDWI. Sensitive to vegetation canopy water content.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            denominator = nir_band + swir1_band
            ndwi = np.where(denominator != 0, (nir_band - swir1_band) / denominator, 0.0)
            ndwi = np.clip(ndwi, -1.0, 1.0)
        return ndwi

    def calculate_msi(self, nir_band, swir1_band):
        """
        Computes Moisture Stress Index (SWIR1 / NIR). Higher values = higher stress.
        """
        with np.errstate(divide='ignore', invalid='ignore'):
            # Avoid division by zero by setting output to 0.0 where NIR is 0
            msi = np.where(nir_band != 0, swir1_band / nir_band, 0.0)
            # Clip to a practical range to prevent infinite values (e.g., [0, 5])
            msi = np.clip(msi, 0.0, 5.0)
        return msi

    def calculate_sar_ratio(self, vv_band, vh_band):
        """
        Computes VV/VH ratio.
        Since raw S1 data is generated in decibel (dB) scale, the ratio in linear scale 
        corresponds to subtraction in dB scale: VV_dB - VH_dB.
        """
        return vv_band - vh_band

    def process_date(self, date_str):
        """
        Loads Sentinel-1 and Sentinel-2 rasters for a specific date, computes spectral/radar indices,
        and saves them as a single multi-band GeoTIFF.
        
        Parameters:
        - date_str: Target date string (YYYY-MM-DD)
        """
        s2_filename = f"sentinel2_{date_str}.tif"
        s1_filename = f"sentinel1_{date_str}.tif"
        
        s2_path = os.path.join(self.raw_dir, s2_filename)
        s1_path = os.path.join(self.raw_dir, s1_filename)
        
        if not os.path.exists(s2_path) or not os.path.exists(s1_path):
            raise FileNotFoundError(f"Missing raw satellite data for date: {date_str}")
            
        print(f"[Preprocessor] Processing raster files for date: {date_str}...")
        
        # Open Sentinel-2 optical data
        with rasterio.open(s2_path) as s2_src:
            # Read bands (1-indexed in rasterio)
            # Band mapping: 1:Blue, 2:Green, 3:Red, 4:NIR, 5:SWIR1, 6:SWIR2
            red = s2_src.read(3)
            nir = s2_src.read(4)
            swir1 = s2_src.read(5)
            
            # Keep geospatial metadata
            meta = s2_src.meta.copy()
            
        # Open Sentinel-1 radar data
        with rasterio.open(s1_path) as s1_src:
            # Band mapping: 1:VV, 2:VH
            vv = s1_src.read(1)
            vh = s1_src.read(2)
            
        # Compute indices
        ndvi = self.calculate_ndvi(red, nir)
        ndwi = self.calculate_ndwi(nir, swir1)
        msi = self.calculate_msi(nir, swir1)
        sar_ratio = self.calculate_sar_ratio(vv, vh)
        
        # Prepare processed directory and metadata
        processed_path = os.path.join(self.processed_dir, f"processed_{date_str}.tif")
        
        # Update metadata for output: 8 bands (Red, NIR, SWIR1, VV, VH, NDVI, NDWI, MSI, SAR_Ratio)
        meta.update(
            count=9,
            dtype=rasterio.float32
        )
        
        # Write processed 9-band raster
        with rasterio.open(processed_path, 'w', **meta) as dst:
            # Original source bands
            dst.write(red.astype(np.float32), 1)
            dst.update_tags(1, name="Red")
            dst.write(nir.astype(np.float32), 2)
            dst.update_tags(2, name="NIR")
            dst.write(swir1.astype(np.float32), 3)
            dst.update_tags(3, name="SWIR1")
            
            # Radar bands
            dst.write(vv.astype(np.float32), 4)
            dst.update_tags(4, name="VV")
            dst.write(vh.astype(np.float32), 5)
            dst.update_tags(5, name="VH")
            
            # Engineered indices
            dst.write(ndvi.astype(np.float32), 6)
            dst.update_tags(6, name="NDVI")
            dst.write(ndwi.astype(np.float32), 7)
            dst.update_tags(7, name="NDWI")
            dst.write(msi.astype(np.float32), 8)
            dst.update_tags(8, name="MSI")
            dst.write(sar_ratio.astype(np.float32), 9)
            dst.update_tags(9, name="SAR_Ratio")
            
        print(f"  -> Created processed multi-band raster: {processed_path}")
        return processed_path


if __name__ == "__main__":
    print("=== Testing Preprocessing Module ===")
    preprocessor = GeospatialPreprocessor()
    
    # Process all raw dates generated during Day 1
    dates = ["2026-05-01", "2026-05-15", "2026-06-01", "2026-06-15"]
    for d in dates:
        preprocessor.process_date(d)
        
    print("\n[Success] Preprocessing complete. Spectral and radar indices calculated and saved.")
