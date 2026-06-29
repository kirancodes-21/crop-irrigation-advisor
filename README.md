# AI-Driven Crop Advisor & Irrigation System

An AI-driven Automated Crop Type, Moisture Stress Detection, and Irrigation Advisory System using moderate-resolution spectral signatures from Optical (Sentinel-2) and Microwave/Radar (Sentinel-1) satellite datasets across growth stages.

Developed as a prototype for the ISRO Hackathon.

---

## 🛰️ Remote Sensing Core Concepts

This project integrates two primary satellite constellations operated by the European Space Agency (ESA) under the Copernicus program, which are also accessible via Google Earth Engine (GEE):

### 1. Sentinel-2 (Optical Imagery)
Sentinel-2 carries a Multispectral Instrument (MSI) that measures reflected solar radiation across 13 spectral bands, ranging from the Visible (RGB) and Near-Infrared (NIR) to Shortwave Infrared (SWIR).
* **Why we use it:** Vegetation interacts strongly with specific wavelengths of light. Healthy green leaves absorb red light (for photosynthesis) and strongly reflect near-infrared (NIR) light due to leaf cellular structure. Water content in leaves absorbs shortwave infrared (SWIR) light.
* **Key Spectral Bands for this Project:**
  * **Band 4 (Red - 665nm):** Vegetation absorption.
  * **Band 8 (NIR - 842nm):** Canopy density and leaf structure reflection. Used for **NDVI** (Normalized Difference Vegetation Index).
  * **Band 8A (Narrow NIR - 865nm):** Sensitive to chlorophyll and biomass.
  * **Band 11 & 12 (SWIR - 1610nm & 2190nm):** Highly sensitive to moisture content in vegetation and soil. Used for **NDWI** (Normalized Difference Water Index) and **MSI** (Moisture Stress Index).
* **Limitation:** Optical sensors cannot penetrate clouds. If it is cloudy, we get no data.

### 2. Sentinel-1 (Microwave / SAR)
Sentinel-1 is a Synthetic Aperture Radar (SAR) mission operating at C-band frequencies. It is an active sensor—it transmits microwave pulses and measures the strength of the backscattered signal returned to the satellite.
* **Why we use it:**
  * **All-weather capability:** Microwaves penetrate clouds, rain, and operate in darkness.
  * **Structural sensitivity:** Radar backscatter is sensitive to the physical structure (roughness, geometry) and dielectric properties (water content) of the soil and plant canopy.
* **Key Polarization Modes:**
  * **VV (Vertical transmit, Vertical receive):** Highly sensitive to surface soil moisture and vertical canopy structure (like stalks).
  * **VH (Vertical transmit, Horizontal receive):** Highly sensitive to volume scattering within the plant canopy (leaves, twigs, overall biomass).
  * **Ratio (VV/VH):** Helps normalize structural differences to track growth progress and soil moisture variation.

### 3. Sensor Fusion: Combining Optical & Microwave
By combining optical and microwave data, we overcome the cloud cover limitations of Sentinel-2 and capture both:
1. **Biochemical properties (Optical):** Chlorophyll content, leaf water absorption, greenness.
2. **Biophysical properties (Microwave):** Biomass volume, canopy structure, moisture stress.

---

## ⚙️ Installation & Setup

1. **Clone/Move to Workspace:**
   Ensure you are in the workspace root:
   ```bash
   cd C:\Users\UDAY KIRAN\.gemini\antigravity\scratch\crop_irrigation_advisor
   ```

2. **Set up Virtual Environment (Recommended):**
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run Dashboard:**
   ```bash
   streamlit run src/app.py
   ```
