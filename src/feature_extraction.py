"""
Feature Extraction Module
--------------------------
This module is responsible for bridging geospatial raster and vector data.
It overlays agricultural field boundary polygons onto multi-band processed satellite images,
extracts the pixels within each field, and calculates descriptive statistics (mean and standard deviation)
for each band/index.

Calculated Features:
- NDVI (mean, std)
- NDWI (mean, std)
- MSI (mean, std)
- VV (mean, std)
- VH (mean, std)
- SAR_Ratio (mean, std)

Output:
A unified tabular CSV file (`data/processed/features_dataset.csv`) where each row represents
a single field observation at a specific date, containing both input features and target labels
(crop type, growth stage, moisture stress level).

Libraries Used:
- geopandas: To load and parse the field boundary polygon shapefiles/GeoJSON.
- rasterio & rasterio.mask: To mask/clip the satellite raster grid based on vector geometries.
- numpy: For statistical aggregation (mean, standard deviation).
- pandas: For compiling and saving the structured tabular dataset.
"""

import os
import numpy as np
import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask

class FeatureExtractor:
    """
    Extracts tabular features from satellite rasters overlaying vector field boundaries.
    """
    def __init__(self, fields_geojson_path="data/raw/fields.geojson", processed_dir="data/processed"):
        self.fields_path = fields_geojson_path
        self.processed_dir = processed_dir
        
        # Load fields boundary vector dataset
        if not os.path.exists(self.fields_path):
            raise FileNotFoundError(f"Fields vector file not found at: {self.fields_path}")
        self.fields_gdf = gpd.read_file(self.fields_path)
        print(f"[FeatureExtractor] Loaded {len(self.fields_gdf)} field boundary polygons.")

    def extract_field_features(self, raster_path, field_polygon):
        """
        Clips the satellite raster to a specific field polygon geometry,
        and calculates statistics for all bands.
        
        Parameters:
        - raster_path: Path to the processed multi-band GeoTIFF.
        - field_polygon: shapely.geometry.Polygon representing the field boundary.
        
        Returns:
        - dict containing the calculated statistics for each band.
        """
        # Load processed raster
        with rasterio.open(raster_path) as src:
            # Mask the raster with the polygon geometry
            # shapes expects a list of GeoJSON-like geometries
            # crop=True clips the raster extent to the bounding box of the polygon
            # nodata value is filled for pixels outside the polygon
            out_image, out_transform = mask(src, [field_polygon], crop=True, nodata=-9999.0)
            
            # Read band descriptions/tags
            # Band mapping:
            # 1:Red, 2:NIR, 3:SWIR1, 4:VV, 5:VH, 6:NDVI, 7:NDWI, 8:MSI, 9:SAR_Ratio
            band_names = [src.tags(i).get("name", f"Band_{i}") for i in range(1, src.count + 1)]
            
        features = {}
        
        for idx, band_name in enumerate(band_names):
            band_data = out_image[idx]
            
            # Filter out nodata values (-9999.0) and NaNs/Infs to calculate statistics only on valid field pixels
            valid_pixels = band_data[(band_data != -9999.0) & (~np.isnan(band_data)) & (~np.isinf(band_data))]
            
            if len(valid_pixels) > 0:
                features[f"{band_name.lower()}_mean"] = float(np.mean(valid_pixels))
                features[f"{band_name.lower()}_std"] = float(np.std(valid_pixels))
            else:
                features[f"{band_name.lower()}_mean"] = 0.0
                features[f"{band_name.lower()}_std"] = 0.0
                
        return features

    def build_dataset(self, dates_list):
        """
        Loops over all dates and fields, extracts spatial features, joins them with 
        field metadata (crop type, growth stage, stress), and saves as a CSV.
        """
        dataset_rows = []
        
        for date_str in dates_list:
            raster_filename = f"processed_{date_str}.tif"
            raster_path = os.path.join(self.processed_dir, raster_filename)
            
            if not os.path.exists(raster_path):
                print(f"[FeatureExtractor WARNING] Processed raster not found for date {date_str}. Skipping.")
                continue
                
            print(f"[FeatureExtractor] Extracting features from raster: {raster_filename}...")
            
            # Process each field polygon
            for idx, row in self.fields_gdf.iterrows():
                field_id = row["field_id"]
                geom = row["geometry"]
                
                # Extract statistical spatial features
                field_features = self.extract_field_features(raster_path, geom)
                
                # Base metadata row
                row_data = {
                    "field_id": int(field_id),
                    "date": date_str,
                    "crop_type": row["crop_type"],
                    "growth_stage": row["growth_stage"],
                    "stress_level": row["stress_level"],
                    "area_ha": row["area_ha"]
                }
                
                # Merge spatial features
                row_data.update(field_features)
                dataset_rows.append(row_data)
                
        # Convert to Pandas DataFrame
        df = pd.DataFrame(dataset_rows)
        
        # Save dataset to CSV
        output_csv_path = os.path.join(self.processed_dir, "features_dataset.csv")
        df.to_csv(output_csv_path, index=False)
        print(f"[FeatureExtractor] Saved compiled ML features dataset to: {output_csv_path}")
        return df


if __name__ == "__main__":
    print("=== Testing Feature Extraction Module ===")
    extractor = FeatureExtractor()
    
    # Process all dates
    observation_dates = ["2026-05-01", "2026-05-15", "2026-06-01", "2026-06-15"]
    dataset = extractor.build_dataset(observation_dates)
    
    # Display sample rows
    print("\nDataset Shape:", dataset.shape)
    print("Sample Columns:\n", dataset.head(2).to_string())
