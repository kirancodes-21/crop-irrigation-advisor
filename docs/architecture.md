# System Architecture

This document describes the high-level system architecture and component structure of the **AI-Driven Automated Crop Type, Moisture Stress Detection, and Irrigation Advisory System**.

## Workflow Diagram

The system follows a modular pipeline from satellite data ingestion to farmer-facing visualization:

```mermaid
graph TD
    %% Source Satellite Data
    subgraph Raw Satellite Data Ingestion
        A1[Sentinel-2 Multispectral Optical] -->|6 Bands: B, G, R, NIR, SWIR1, SWIR2| B1[Ingestion Pipeline]
        A2[Sentinel-1 SAR Radar] -->|2 Bands: VV, VH polarization| B1
    end

    %% Preprocessing Pipeline
    subgraph Preprocessing & Index Calculation
        B1 --> C1[Geospatial Preprocessor]
        C1 -->|Index Calculation| D1[NDVI calculation]
        C1 -->|Index Calculation| D2[NDWI calculation]
        C1 -->|Index Calculation| D3[MSI calculation]
        C1 -->|Radar subtraction| D4[VV/VH Ratio]
        
        D1 & D2 & D3 & D4 --> E1[9-Band Processed Raster output]
    end

    %% Vector Integration & Masking
    subgraph Feature Extraction (Vector Overlay)
        E1 --> F1[Raster Masking & Polygon Overlay]
        G1[Fields GeoJSON Vector Boundaries] --> F1
        F1 -->|Zonal Statistics| H1[Statistical Summaries: Mean & Std Dev]
        H1 -->|Tabular Compilation| I1[Compiled ML CSV Dataset]
    end

    %% Machine Learning Models
    subgraph AI Model Inference
        I1 --> J1[Crop Type Classifier]
        I1 --> J2[Moisture Stress Classifier]
        
        J1 -->|Predicted Crop Type| J3[Growth Stage Classifier]
        I1 --> J3
    end

    %% Advisory Rule Engine
    subgraph Expert Rule Advisory Engine
        J1 -->|Predicted Crop| K1[Irrigation Advisory Rule Engine]
        J2 -->|Predicted Stress| K1
        J3 -->|Predicted Stage| K1
        L1[Microclimate Sliders: Temp, Forecast Rain] --> K1
        L2[Soil Category: Sandy, Loamy, Clay] --> K1
        
        K1 --> M1[Water Requirement depth in mm]
        K1 --> M2[Urgency Status & Recommendation Alert]
    end

    %% Presentation Layer
    subgraph User Presentation Layer
        M1 & M2 --> N1[Streamlit Web Dashboard]
        G1 -->|Map Layers| N1
        I1 -->|Temporal Charts| N1
        N1 --> O1[Farmer & Agricultural Officers UI]
    end

    style A1 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    style A2 fill:#e1f5fe,stroke:#0288d1,stroke-width:2px;
    style E1 fill:#e8f5e9,stroke:#2e7d32,stroke-width:2px;
    style I1 fill:#fff3e0,stroke:#ef6c00,stroke-width:2px;
    style K1 fill:#f3e5f5,stroke:#6a1b9a,stroke-width:2px;
    style O1 fill:#e0f2f1,stroke:#00695c,stroke-width:2px;
```

## System Components

### 1. Ingestion Layer (`src/data_acquisition.py`)
- Interfaces with Google Earth Engine (GEE) Python API.
- Implements cloud masking for Sentinel-2 QA bands to filter cloudy observations.
- Provides simulated raster structures matching local CRS (EPSG:4326) for offline capability.

### 2. Preprocessing & Sensor Fusion (`src/preprocessing.py`)
- Standardizes spatial projections.
- Combines biochemical reflections (NIR, Red) and canopy water indices (SWIR, NIR) with biophysical radar structures (VV, VH) to create a cohesive 9-band processed dataset.

### 3. Zonal Feature Extractor (`src/feature_extraction.py`)
- Uses `rasterio.mask` to overlay vector parcel boundaries (farms) onto the raster coordinate space.
- Calculates spatial statistics (mean and standard deviation) to filter sensor pixel-level noise, making the output robust to variable parcel shapes and sizes.

### 4. Core Predictive ML Layer (`src/models/`)
- **Crop Classifier**: Multi-class Random Forest predicting crop labels.
- **Moisture Stress Classifier**: Evaluates canopy water loss.
- **Growth Stage Classifier**: Uses crop-type dummy features combined with temporal values to map biological stages.

### 5. Agronomic Advisory Engine (`src/advisory_engine.py`)
- Integrates temperature evapotranspiration adjustments, soil texture infiltration capacities, and rain forecast delays (to save water).

### 6. Interface Dashboard (`src/app.py`)
- Visualizes geospatial boundaries on an interactive Folium leaflet map.
- Displays dynamic timeseries line charts of crop health indicators.
- Injects a dark glassmorphic styling system to provide a premium user experience.
