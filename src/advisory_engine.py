"""
Irrigation Advisory Engine
---------------------------
This module compiles:
1. Predicted Crop Type
2. Predicted Growth Stage
3. Predicted Moisture Stress Level
4. Weather Parameters (Temperature, Past Rain, Rain Forecast)
5. Soil Type (Clayey, Loamy, Sandy)

It executes an agronomic rule-engine to calculate the precise irrigation volume (depth in mm)
needed, generates a priority alert level, and writes a detailed advisory report for the farmer.

Agronomic Logic:
- Rice requires flooding during Sowing (transplanting) and Flowering, but no water during Maturity.
- Cotton and Maize are sensitive to waterlogging; drip irrigation is recommended.
- Heavy rain forecast (>10mm) automatically postpones irrigation to conserve water and prevent run-off.
- Sandy soils require smaller, more frequent water applications, while Clayey soils hold water longer.
"""

class IrrigationAdvisoryEngine:
    def __init__(self):
        # Soil permeability and water retention factors
        self.soil_factors = {
            "Sandy": {"freq": "High", "volume_mult": 0.8, "desc": "Low water retention, requires frequent small applications."},
            "Loamy": {"freq": "Medium", "volume_mult": 1.0, "desc": "Ideal drainage and water retention properties."},
            "Clayey": {"freq": "Low", "volume_mult": 1.2, "desc": "High water retention, prone to waterlogging. Apply slowly."}
        }
        
        # Base irrigation depths (mm) under Severe Stress for Crop x Stage
        self.base_requirements = {
            "Rice": {
                "Sowing": 40.0,
                "Vegetative": 45.0,
                "Flowering": 60.0,
                "Maturity": 0.0
            },
            "Wheat": {
                "Sowing": 25.0,
                "Vegetative": 35.0,
                "Flowering": 45.0,
                "Maturity": 0.0
            },
            "Cotton": {
                "Sowing": 20.0,
                "Vegetative": 30.0,
                "Flowering": 40.0,
                "Maturity": 0.0
            },
            "Maize": {
                "Sowing": 20.0,
                "Vegetative": 35.0,
                "Flowering": 50.0,
                "Maturity": 0.0
            },
            "Sugarcane": {
                "Sowing": 30.0,
                "Vegetative": 50.0,
                "Flowering": 60.0,
                "Maturity": 10.0
            }
        }

    def generate_advisory(self, crop, stage, stress, temp_c=30.0, forecast_rain_mm=0.0, soil_type="Loamy"):
        """
        Generates irrigation recommendation details.
        
        Parameters:
        - crop: Predicted crop type (str)
        - stage: Predicted growth stage (str)
        - stress: Predicted moisture stress level (str)
        - temp_c: Temperature in Celsius (float)
        - forecast_rain_mm: Expected rain in next 48 hours (float)
        - soil_type: Soil type - Sandy, Loamy, Clayey (str)
        
        Returns:
        - dict: contains irrigation_depth_mm, urgency_level, method, and advisory_text
        """
        # Validate inputs
        if crop not in self.base_requirements:
            crop = "Maize" # fallback
        if stage not in self.base_requirements[crop]:
            stage = "Vegetative" # fallback
        if soil_type not in self.soil_factors:
            soil_type = "Loamy"

        # 1. Base requirements check
        max_depth = self.base_requirements[crop][stage]
        
        # Calculate depth based on stress level
        if stress == "No Stress" or max_depth == 0:
            irrigation_depth = 0.0
            urgency = "Normal"
        elif stress == "Mild Stress":
            irrigation_depth = max_depth * 0.5
            urgency = "Moderate"
        else: # Severe Stress
            irrigation_depth = max_depth
            urgency = "Critical"
            
        # 2. Adjust for Temperature (high heat = high evapotranspiration = more water needed)
        if temp_c > 35.0 and irrigation_depth > 0:
            irrigation_depth *= 1.15  # increase by 15%
            
        # 3. Adjust for Soil Type
        soil_info = self.soil_factors[soil_type]
        irrigation_depth *= soil_info["volume_mult"]
        
        # Round value
        irrigation_depth = round(irrigation_depth, 1)
        
        # 4. Adjust for Weather Forecast (If it rains, we don't need to irrigate!)
        postpone = False
        postpone_reason = ""
        if forecast_rain_mm >= 15.0 and irrigation_depth > 0:
            postpone = True
            urgency = "Postponed"
            postpone_reason = f"A heavy rain of {forecast_rain_mm}mm is forecast. Postpone irrigation to save water."
        elif forecast_rain_mm > 5.0 and irrigation_depth > 0:
            # partial rain adjustment
            irrigation_depth = max(0.0, round(irrigation_depth - forecast_rain_mm, 1))
            if irrigation_depth == 0:
                urgency = "Postponed"
                postpone = True
                postpone_reason = f"Light rain ({forecast_rain_mm}mm) forecast is sufficient to cover requirements."
            else:
                postpone_reason = f"Irrigation reduced by {forecast_rain_mm}mm due to light rain forecast."

        # 5. Determine Irrigation Method
        if crop == "Rice":
            method = "Basin Flooding (maintain shallow standing water)"
        elif crop in ["Cotton", "Sugarcane"]:
            method = "Drip Irrigation (slow emission near root zone)"
        else:
            method = "Sprinkler Irrigation or Furrow Irrigation"
            
        # 6. Compose Advisory Text
        if irrigation_depth == 0:
            if stage == "Maturity":
                advisory_text = f"The {crop} crop is in the Maturity stage. Draw down field water to prepare for harvesting and increase crop quality."
            elif postpone:
                advisory_text = postpone_reason
            else:
                advisory_text = f"The {crop} field shows no signs of moisture stress. Soil moisture levels are optimal. Keep monitoring."
        else:
            advisory_text = f"Apply {irrigation_depth}mm of water using {method}. "
            advisory_text += f"The crop is in the sensitive '{stage}' stage. "
            if urgency == "Critical":
                advisory_text += f"CRITICAL: Severe moisture stress detected! Irrigate immediately to prevent yield loss. "
            elif urgency == "Moderate":
                advisory_text += f"Mild moisture stress detected. Schedule irrigation within the next 24-48 hours. "
                
            advisory_text += f"Soil note: {soil_info['desc']}"
            if postpone_reason:
                advisory_text += f" {postpone_reason}"
                
        return {
            "crop": crop,
            "stage": stage,
            "stress": stress,
            "irrigation_depth_mm": irrigation_depth,
            "urgency": urgency,
            "method": method,
            "advisory_text": advisory_text,
            "soil_type": soil_type
        }


if __name__ == "__main__":
    print("=== Testing Irrigation Advisory Engine ===")
    engine = IrrigationAdvisoryEngine()
    
    # Test case 1: Rice, Flowering, Severe Stress, High Temp
    adv1 = engine.generate_advisory("Rice", "Flowering", "Severe Stress", temp_c=38.0, forecast_rain_mm=0.0)
    print("\nTest Case 1 (Rice, Flowering, Severe Stress):")
    print(f"Urgency: {adv1['urgency']}")
    print(f"Depth: {adv1['irrigation_depth_mm']} mm")
    print(f"Method: {adv1['method']}")
    print(f"Advisory: {adv1['advisory_text']}")
    
    # Test case 2: Maize, Flowering, Severe Stress, Rain Forecast
    adv2 = engine.generate_advisory("Maize", "Flowering", "Severe Stress", temp_c=30.0, forecast_rain_mm=20.0)
    print("\nTest Case 2 (Maize, Flowering, Severe Stress, Rain Forecast):")
    print(f"Urgency: {adv2['urgency']}")
    print(f"Depth: {adv2['irrigation_depth_mm']} mm")
    print(f"Advisory: {adv2['advisory_text']}")
