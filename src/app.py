"""
Streamlit Web Dashboard Application
-------------------------------------
This is the core presentation layer for the hackathon prototype. It implements:
1. Glassmorphic premium dark UI theme via CSS injection.
2. Interactive geospatial map using Folium and streamlit-folium.
3. Live prediction inference:
   - Loads trained crop, stage, and stress classifiers.
   - Computes prediction confidence percentages from model probabilities.
4. Interactive parameter testing (Soil type, Temperature, Rain forecast sliders).
5. Dynamic time-series plots of NDVI, NDWI, and radar backscatter using Plotly.
6. Highly polished agronomic advisory panels.
"""

import os
import joblib
import numpy as np
import pandas as pd
import geopandas as gpd
import streamlit as st
import folium
from streamlit_folium import st_folium
import plotly.graph_objects as go
from advisory_engine import IrrigationAdvisoryEngine

# ==========================================
# PAGE CONFIGURATION & STYLING
# ==========================================
st.set_page_config(
    page_title="ISRO Hackathon: AI Crop & Irrigation Advisor",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Premium Dark Glassmorphism Styling
custom_css = """
<style>
    /* Dark Theme Core */
    .stApp {
        background: linear-gradient(135deg, #0e1117 0%, #151a24 100%);
        color: #e2e8f0;
        font-family: 'Outfit', 'Inter', sans-serif;
    }
    
    /* Header & Titles */
    h1 {
        background: linear-gradient(90deg, #10b981 0%, #3b82f6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
    }
    
    h2, h3 {
        color: #10b981 !important;
        font-weight: 600 !important;
    }
    
    /* Glassmorphism Containers */
    div[data-testid="metric-container"], .card {
        background: rgba(30, 41, 59, 0.45) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border: 1px solid rgba(255, 255, 255, 0.08) !important;
        border-radius: 16px !important;
        padding: 20px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.3) !important;
        transition: transform 0.2s ease-in-out, border-color 0.2s;
    }
    
    div[data-testid="metric-container"]:hover, .card:hover {
        transform: translateY(-2px);
        border-color: rgba(16, 185, 129, 0.3) !important;
    }
    
    /* Custom Badges */
    .badge {
        padding: 6px 12px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 0.85rem;
        display: inline-block;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .badge-critical { background-color: rgba(239, 68, 68, 0.2); color: #f87171; border: 1px solid #ef4444; }
    .badge-moderate { background-color: rgba(245, 158, 11, 0.2); color: #fbbf24; border: 1px solid #f59e0b; }
    .badge-normal { background-color: rgba(16, 185, 129, 0.2); color: #34d399; border: 1px solid #10b981; }
    .badge-postponed { background-color: rgba(59, 130, 246, 0.2); color: #60a5fa; border: 1px solid #3b82f6; }
    
    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0b0f16 !important;
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)


# ==========================================
# DATA & MODEL LOADING FUNCTIONS
# ==========================================
@st.cache_resource
def load_models_and_data():
    """
    Loads saved Scikit-Learn models and GeoJSON boundary fields.
    """
    models_dir = "src/models"
    data_dir = "data/raw"
    processed_dir = "data/processed"
    
    crop_model = joblib.load(os.path.join(models_dir, "crop_classifier.pkl"))
    stage_model = joblib.load(os.path.join(models_dir, "growth_stage.pkl"))
    stress_model = joblib.load(os.path.join(models_dir, "moisture_stress.pkl"))
    
    fields_gdf = gpd.read_file(os.path.join(data_dir, "fields.geojson"))
    features_df = pd.read_csv(os.path.join(processed_dir, "features_dataset.csv"))
    
    # Convert Timestamp/Datetime columns in GeoDataFrame to string to prevent Folium JSON serialization errors
    for col in fields_gdf.columns:
        if col != 'geometry':
            if pd.api.types.is_datetime64_any_dtype(fields_gdf[col]) or fields_gdf[col].dtype == 'object':
                fields_gdf[col] = fields_gdf[col].astype(str)
                
    return crop_model, stage_model, stress_model, fields_gdf, features_df

try:
    crop_model_data, stage_model_data, stress_model_data, fields_gdf, features_df = load_models_and_data()
except Exception as e:
    st.error(f"Error loading system models or data files: {e}")
    st.info("Make sure you have completed Day 1 to 4 steps: simulation, preprocessing, and training models.")
    st.stop()

# Initialize Advisory Engine
advisory_engine = IrrigationAdvisoryEngine()


# ==========================================
# SIDEBAR CONTROLS (Weather & Field Picker)
# ==========================================
st.sidebar.markdown("<h2 style='text-align: center; color: #10b981;'>🛰️ Controls Panel</h2>", unsafe_allow_html=True)
st.sidebar.write("---")

# Field selection dropdown (fallback for map click)
field_ids = sorted(fields_gdf["field_id"].unique())
selected_field_id = st.sidebar.selectbox("Select Target Field:", field_ids, index=0)

# Observation Date selection
available_dates = sorted(features_df["date"].unique())
selected_date = st.sidebar.select_slider("Select Observation Date:", options=available_dates, value=available_dates[-1])

st.sidebar.write("---")
st.sidebar.markdown("### 🌡️ Microclimate Parameters")
temp_c = st.sidebar.slider("Ambient Temperature (°C):", min_value=15.0, max_value=45.0, value=31.5, step=0.5)
rain_forecast_mm = st.sidebar.slider("48-Hour Rain Forecast (mm):", min_value=0.0, max_value=50.0, value=0.0, step=1.0)
soil_type = st.sidebar.selectbox("Soil Profile Type:", ["Loamy", "Sandy", "Clayey"], index=0)

st.sidebar.write("---")
st.sidebar.markdown(
    """
    <div style='font-size: 0.85rem; color: #64748b; text-align: center;'>
        ISRO Hackathon Prototype v1.0<br>
        Sentinel-1 & Sentinel-2 Sensor Fusion
    </div>
    """, 
    unsafe_allow_html=True
)


# ==========================================
# MAIN PAGE LAYOUT
# ==========================================
st.markdown("<h1>Crop Analytics & AI Irrigation Advisory System</h1>", unsafe_allow_html=True)
st.markdown(
    f"<p style='color: #94a3b8; font-size: 1.1rem; margin-top: -10px; margin-bottom: 25px;'> "
    f"Active Field Boundary: <b>Field {selected_field_id}</b> | Sensor Date: <b>{selected_date}</b>"
    f"</p>", 
    unsafe_allow_html=True
)

# Extract features matching chosen field and date
field_date_features = features_df[(features_df["field_id"] == selected_field_id) & (features_df["date"] == selected_date)]

if field_date_features.empty:
    st.warning("No feature signatures found for the selected Field and Date combination.")
    st.stop()
    
# Extract singular feature row
feat_row = field_date_features.iloc[0]


# ==========================================
# ML MODEL INFERENCE (Real-time Prediction)
# ==========================================
# 1. Predict Crop Type
crop_features = crop_model_data["features"]
X_crop = pd.DataFrame([feat_row[crop_features]])
pred_crop = crop_model_data["model"].predict(X_crop)[0]
crop_conf = np.max(crop_model_data["model"].predict_proba(X_crop)) * 100

# 2. Predict Growth Stage (needs dummy variables for crop_type)
stage_features = stage_model_data["features"]
# Rebuild dummy columns
X_stage_dict = {col: 0.0 for col in stage_features}
# Copy numeric features
for col in stage_features:
    if col in feat_row:
        X_stage_dict[col] = feat_row[col]
# Set the dummy category predicted by crop_classifier
dummy_col = f"crop_{pred_crop}"
if dummy_col in X_stage_dict:
    X_stage_dict[dummy_col] = 1.0
    
X_stage = pd.DataFrame([X_stage_dict])[stage_features]
pred_stage = stage_model_data["model"].predict(X_stage)[0]
stage_conf = np.max(stage_model_data["model"].predict_proba(X_stage)) * 100

# 3. Predict Moisture Stress
stress_features = stress_model_data["features"]
X_stress = pd.DataFrame([feat_row[stress_features]])
pred_stress = stress_model_data["model"].predict(X_stress)[0]
stress_conf = np.max(stress_model_data["model"].predict_proba(X_stress)) * 100

# Generate agronomic advisory
advisory = advisory_engine.generate_advisory(
    crop=pred_crop, 
    stage=pred_stage, 
    stress=pred_stress, 
    temp_c=temp_c, 
    forecast_rain_mm=rain_forecast_mm, 
    soil_type=soil_type
)


# ==========================================
# LAYOUT SPLIT: Row 1 - Cards, Map, Advisory
# ==========================================
col_map, col_details = st.columns([1.1, 0.9])

with col_map:
    st.subheader("🗺️ Interactive Geospatial Field Map")
    
    # Coordinates of simulation base
    field_coords = fields_gdf[fields_gdf["field_id"] == selected_field_id].geometry.iloc[0].centroid
    
    # Initialize Folium Map
    m = folium.Map(location=[field_coords.y, field_coords.x], zoom_start=14, tiles="cartodbpositron")
    
    # Style fields layer
    def get_style(feature):
        f_id = feature["properties"]["field_id"]
        if f_id == selected_field_id:
            return {
                "fillColor": "#10b981",
                "color": "#10b981",
                "weight": 3,
                "fillOpacity": 0.5
            }
        else:
            return {
                "fillColor": "#3b82f6",
                "color": "#3b82f6",
                "weight": 1.5,
                "fillOpacity": 0.2
            }
            
    # Add GeoJSON layer
    folium.GeoJson(
        fields_gdf,
        style_function=get_style,
        tooltip=folium.GeoJsonTooltip(fields=["field_id", "crop_type", "area_ha"], aliases=["Field ID:", "Simulated Crop:", "Area (ha):"])
    ).add_to(m)
    
    # Render map
    map_data = st_folium(m, height=400, width=700, key="fields_map")
    
    # Capture map clicks to update selected_field_id dynamically
    if map_data and map_data.get("last_active_drawing"):
        clicked_properties = map_data["last_active_drawing"]["properties"]
        clicked_id = clicked_properties.get("field_id")
        if clicked_id and clicked_id != selected_field_id:
            # We can re-trigger page update by modifying selected_field_id
            st.session_state["selected_field_id"] = clicked_id
            st.rerun()

with col_details:
    st.subheader("🔮 Real-Time Satellite Analytics")
    
    # Visual grid of predictions
    c1, c2 = st.columns(2)
    with c1:
        st.metric(
            label="Predicted Crop Type", 
            value=pred_crop, 
            delta=f"{crop_conf:.1f}% Confidence"
        )
        st.metric(
            label="Estimated Growth Stage", 
            value=pred_stage, 
            delta=f"{stage_conf:.1f}% Confidence"
        )
    with c2:
        # Custom display for stress delta
        st.metric(
            label="Moisture Stress", 
            value=pred_stress, 
            delta=f"{stress_conf:.1f}% Confidence"
        )
        # Display Hectares Area
        st.metric(
            label="Field Area Coverage", 
            value=f"{feat_row['area_ha']:.2f} Hectares",
            delta="Derived from GeoJSON"
        )

# ==========================================
# ROW 2: ADVISORY BOX (Vibrant & Premium)
# ==========================================
st.write("---")
st.subheader("💡 Dynamic Irrigation Advisory Panel")

urgency_val = advisory["urgency"]
if urgency_val == "Critical":
    badge_html = '<span class="badge badge-critical">Critical Action Required</span>'
elif urgency_val == "Moderate":
    badge_html = '<span class="badge badge-moderate">Moderate Priority</span>'
elif urgency_val == "Postponed":
    badge_html = '<span class="badge badge-postponed">Irrigation Postponed</span>'
else:
    badge_html = '<span class="badge badge-normal">No Actions Required</span>'
    
# Advisory card container
st.markdown(
    f"""
    <div class="card" style="border-left: 6px solid #10b981 !important;">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 15px;">
            <span style="font-size: 1.25rem; font-weight: 700; color: #e2e8f0;">Agronomic Recommendation</span>
            {badge_html}
        </div>
        <div style="margin-bottom: 12px;">
            <span style="color: #94a3b8; font-size: 0.95rem;">Recommended Irrigation Depth:</span>
            <span style="font-size: 1.8rem; font-weight: 800; color: #10b981; margin-left: 10px;">
                {advisory['irrigation_depth_mm']} mm
            </span>
        </div>
        <p style="font-size: 1.1rem; line-height: 1.6; color: #cbd5e1; background: rgba(15, 23, 42, 0.5); padding: 15px; border-radius: 8px;">
            <b>Advisory Details:</b> {advisory['advisory_text']}
        </p>
        <div style="font-size: 0.9rem; color: #64748b;">
            <b>Irrigation Method:</b> {advisory['method']} | <b>Soil Layer:</b> {soil_type}
        </div>
    </div>
    """,
    unsafe_allow_html=True
)


# ==========================================
# ROW 3: TEMPORAL CHARTS & SPECTRAL SIGNATURES
# ==========================================
st.write("---")
col_chart1, col_chart2 = st.columns(2)

# Load full history for this specific field to plot the time-series curve
field_history = features_df[features_df["field_id"] == selected_field_id].sort_values("date")

with col_chart1:
    st.subheader("📈 Temporal Spectral Indices Curve")
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=field_history["date"], 
        y=field_history["ndvi_mean"],
        mode='lines+markers',
        name='NDVI (Vegetation Index)',
        line=dict(color='#10b981', width=3),
        marker=dict(size=8)
    ))
    fig.add_trace(go.Scatter(
        x=field_history["date"], 
        y=field_history["ndwi_mean"],
        mode='lines+markers',
        name='NDWI (Water Index)',
        line=dict(color='#3b82f6', width=3),
        marker=dict(size=8)
    ))
    
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, range=[-0.2, 1.0]),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

with col_chart2:
    st.subheader("📡 Temporal SAR Microwave Backscatter Curve")
    
    fig_sar = go.Figure()
    fig_sar.add_trace(go.Scatter(
        x=field_history["date"], 
        y=field_history["vv_mean"],
        mode='lines+markers',
        name='SAR VV (Surface/Soil Moisture)',
        line=dict(color='#fbbf24', width=3),
        marker=dict(size=8)
    ))
    fig_sar.add_trace(go.Scatter(
        x=field_history["date"], 
        y=field_history["vh_mean"],
        mode='lines+markers',
        name='SAR VH (Canopy Structure/Volume)',
        line=dict(color='#ec4899', width=3),
        marker=dict(size=8)
    ))
    
    fig_sar.update_layout(
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#94a3b8'),
        margin=dict(l=40, r=40, t=10, b=40),
        xaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True),
        yaxis=dict(gridcolor='rgba(255,255,255,0.05)', showgrid=True, title="Backscatter Coefficient (dB)"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig_sar, use_container_width=True)

# Footer Info
st.markdown("<p style='text-align: center; color: #64748b; font-size: 0.85rem; margin-top: 40px;'>Designed by Antigravity under Gemini for the ISRO Hackathon 2026.</p>", unsafe_allow_html=True)
