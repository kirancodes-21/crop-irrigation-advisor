# Project Research Documentation

This document compiles the research foundations, literature review, and methodology details for the crop classification, moisture stress, and irrigation advisory system.

---

## 1. Abstract
Agricultural water management is key to global food security under shifting climatic patterns. Traditional methods for irrigation planning rely on spot soil measurements or weather forecasts, which lack spatial resolution and fails to scale. This project presents a sensor-fusion methodology combining multispectral optical (Sentinel-2) and Synthetic Aperture Radar (Sentinel-1) satellite imagery to automate crop classification, growth stage tracking, and canopy moisture stress detection. By overlaying crop parcel vector boundaries, zonal statistics are extracted to compile temporal feature vectors. Multi-output Random Forest Classifiers are trained to identify crop type, growth phase, and moisture stress. These predictions feed an agronomic advisory engine that integrates soil physics (water retention) and real-time microclimatic variables (evapotranspiration, rain forecasts) to output precise daily irrigation recommendations (depth in mm). The system is deployed via an interactive, high-fidelity GIS dashboard, offering scalable digital agriculture decisions for regional water management.

---

## 2. Introduction
Agriculture consumes approximately 70% of global freshwater withdrawals, with significant inefficiencies due to over-irrigation or misaligned water timing. Optimal scheduling requires knowing:
1. What crop is being grown (different water demands).
2. What stage of development the crop is in (flowering is highly sensitive to water stress compared to maturity).
3. The actual moisture status of the canopy and soil.

By combining optical datasets (which capture chlorophyll activity and water absorption) and microwave radar (which capture physical biomass structure and soil surface moisture, bypassing cloud cover limitations), we build a robust, weather-agnostic remote sensing crop advisor.

---

## 3. Literature Review
1. **Spectral Indices for Vegetation Monitoring**: Rouse et al. (1974) established the Normalized Difference Vegetation Index (NDVI), utilizing the red absorption and near-infrared reflection of leaves. Gao (1996) developed the Normalized Difference Water Index (NDWI) replacing red with SWIR bands, proving its sensitivity to liquid water content of vegetation canopies.
2. **Synthetic Aperture Radar (SAR) in Agriculture**: McNairn et al. (2009) demonstrated that C-band SAR backscatter polarizations (VV, VH) are sensitive to canopy volume scattering and dielectric constants (associated with water content). The ratio $VV/VH$ is widely used to cancel surface soil roughness effects, isolating structural changes of crops.
3. **Sensor Fusion**: Recent studies (Veloso et al., 2017) confirm that coupling Sentinel-1 and Sentinel-2 data overcomes optical cloud-mask data loss, improving classification accuracy by up to 12% in monsoonal cropping regions (such as India).

---

## 4. Methodology
The methodology consists of four key stages:

### A. Data Source Specification & Bands
- **Sentinel-2 Multispectral Instrument (MSI)**:
  - Band 4 (Red, 665nm) – spatial resolution 10m
  - Band 8 (NIR, 842nm) – spatial resolution 10m
  - Band 11 (SWIR-1, 1610nm) – spatial resolution 20m (resampled to 10m)
- **Sentinel-1 C-band SAR**:
  - VV (Vertical polarization) – spatial resolution 10m
  - VH (Horizontal polarization) – spatial resolution 10m

### B. Preprocessing & Indices Formulas
1. **NDVI**: 
   $$\text{NDVI} = \frac{\text{NIR} - \text{Red}}{\text{NIR} + \text{Red}} = \frac{B_8 - B_4}{B_8 + B_4}$$
2. **NDWI**: 
   $$\text{NDWI} = \frac{\text{NIR} - \text{SWIR}_1}{\text{NIR} + \text{SWIR}_1} = \frac{B_8 - B_{11}}{B_8 + B_{11}}$$
3. **MSI**: 
   $$\text{MSI} = \frac{\text{SWIR}_1}{\text{NIR}} = \frac{B_{11}}{B_8}$$
4. **SAR Backscatter Ratio**: 
   $$\text{SAR Ratio} = \text{VV (dB)} - \text{VH (dB)}$$

### C. Zonal Feature Extraction
Field vector boundaries (polygons) are overlayed onto the raster datasets. For each parcel, pixels inside the boundary are masked, and zonal statistics are calculated:
$$\mu = \frac{1}{N} \sum_{i=1}^{N} P_i, \quad \sigma = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (P_i - \mu)^2}$$
where $P_i$ represents the value of pixel $i$ inside the parcel and $N$ is the total count of pixels.

### D. Machine Learning & Model Evaluation
Three separate Random Forest Classifiers are trained using the zonal statistical features:
1. **Crop Classifier**: Model predicting crop category (Rice, Wheat, Cotton, Maize, Sugarcane).
2. **Growth Stage Detector**: Model predicting growth phase (Sowing, Vegetative, Flowering, Maturity) using dummy variable fields of predicted crop.
3. **Moisture Stress Model**: Model predicting stress category (No, Mild, Severe).

---

## 5. Limitations
1. **Spatial Resolution**: At 10m resolution (Sentinel-1 and Sentinel-2), small, fragmented farm plots (common in Indian smallholder farming, often $< 0.5$ hectares) lead to "mixed pixel" issues, where a single pixel contains spectral signatures of multiple crops.
2. **Temporal Revisit Times**: Sentinel-2 has a 5-day revisit cycle. Heavy cloud cover during monsoon periods can lead to data gaps spanning multiple weeks.
3. **Active Radar Noise**: SAR imagery is prone to "speckle" noise (salt-and-pepper texture), requiring spatial filtering or temporal smoothing.

---

## 6. Future Scope
1. **Incorporation of High-Resolution Imagery**: Integrating ISRO’s LISS-IV (5.8m resolution) or Cartosat datasets to resolve boundary mixed-pixel issues.
2. **Deep Learning Sequence Models**: Utilizing LSTM or Transformer models to process the sequential time-series inputs, exploiting temporal trajectories directly.
3. **Soil Moisture Hydrological Coupling**: Integrating physical soil hydrology equations (e.g. Richard's equation) to combine remote sensing surface observations with deep soil profile root-zone moisture estimates.

---

## 7. References
- Gao, B. C. (1996). NDWI—A normalized difference water index for remote sensing of vegetation liquid water from space. *Remote Sensing of Environment*, 58(3), 257-266.
- Rouse, J. W., Haas, R. H., Schell, J. A., & Deering, D. W. (1974). Monitoring vegetation systems in the Great Plains with ERTS. *Third ERTS Symposium*, NASA SP-351 I, 309-317.
- Veloso, A., Mermoz, S., Bouvet, A., Le Toan, T., Planells, M., Dejoux, J. F., & Ceschia, E. (2017). Understanding the temporal behavior of crops using Sentinel-1 and Sentinel-2 coherence and backscatter. *Remote Sensing of Environment*, 199, 108-119.
