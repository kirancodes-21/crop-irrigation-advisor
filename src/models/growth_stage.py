"""
Growth Stage Detection Model
----------------------------
This module trains and evaluates a Random Forest classifier to identify the growth stage of a crop
(e.g., Sowing, Vegetative, Flowering, Maturity) based on temporal Sentinel-1/2 features and crop type.

Why include Crop Type?
- Different crops have distinct growth curves. Rice in 'Sowing' is flooded (very high NDWI, low NDVI),
  whereas Wheat in 'Sowing' is dry (low NDWI, low NDVI).
- We use One-Hot Encoding to feed the crop type as a categorical feature to the ML model alongside raster features.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class GrowthStageTrainer:
    def __init__(self, dataset_path="data/processed/features_dataset.csv", model_dir="src/models"):
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Features dataset CSV not found at: {self.dataset_path}")
            
        self.df = pd.read_csv(self.dataset_path)

    def augment_dataset(self, target_samples=250):
        """
        Augments the dataset while maintaining realistic growth stage associations for each crop.
        """
        print(f"[GrowthStage] Augmenting dataset from {len(self.df)} to {target_samples} samples...")
        np.random.seed(42)
        
        # Numeric spatial features
        feature_cols = [col for col in self.df.columns if "_mean" in col or "_std" in col]
        
        augmented_rows = []
        # Group by crop type & growth stage to preserve class distributions
        for (crop_name, stage_name), group in self.df.groupby(["crop_type", "growth_stage"]):
            num_base_samples = len(group)
            # Distribute target samples across combinations
            samples_to_add = max(5, target_samples // 20 - num_base_samples)
            
            # Original samples
            for _, row in group.iterrows():
                augmented_rows.append(row.to_dict())
                
            # Synthesize samples
            for _ in range(samples_to_add):
                base_row = group.sample(n=1).iloc[0].to_dict()
                
                # Add slight noise to spectral/radar values
                for col in feature_cols:
                    base_val = base_row[col]
                    noise = np.random.normal(0, abs(base_val) * 0.05 + 0.001)
                    base_row[col] = base_val + noise
                    
                base_row["stress_level"] = np.random.choice(["No Stress", "Mild Stress", "Severe Stress"])
                base_row["field_id"] = np.random.randint(10, 100)
                augmented_rows.append(base_row)
                
        return pd.DataFrame(augmented_rows)

    def train(self):
        # 1. Prepare augmented data
        aug_df = self.augment_dataset()
        
        # 2. Extract numeric features and one-hot encode Crop Type
        numeric_features = [col for col in aug_df.columns if "_mean" in col or "_std" in col]
        
        # Perform One-Hot encoding of 'crop_type'
        crop_dummies = pd.get_dummies(aug_df["crop_type"], prefix="crop")
        
        # Combine numerical features and one-hot encoded columns
        X = pd.concat([aug_df[numeric_features], crop_dummies], axis=1)
        y = aug_df["growth_stage"]
        
        # Keep track of all feature names for inference step
        feature_names = list(X.columns)
        
        # Convert bool dummies to float32
        for col in crop_dummies.columns:
            X[col] = X[col].astype(np.float32)
            
        # 3. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"[GrowthStage] Total input features: {len(feature_names)}")
        print(f"[GrowthStage] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # 4. Model training
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        model.fit(X_train, y_train)
        
        # 5. Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n==============================================")
        print(f"GROWTH STAGE DETECTION ACCURACY: {accuracy * 100:.2f}%")
        print("==============================================")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # 6. Feature Importance
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 5 Most Discriminative Features:")
        for rank in range(5):
            idx = indices[rank]
            print(f"{rank + 1}. {feature_names[idx]}: {importances[idx]:.4f}")
            
        # 7. Save model, features list, and original crop category classes
        model_path = os.path.join(self.model_dir, "growth_stage.pkl")
        joblib.dump({
            "model": model, 
            "features": feature_names,
            "crop_classes": list(aug_df["crop_type"].unique())
        }, model_path)
        print(f"\n[GrowthStage] Saved trained model to: {model_path}")
        
        return model, feature_names


if __name__ == "__main__":
    print("=== Training Growth Stage Detection Model ===")
    trainer = GrowthStageTrainer()
    trainer.train()
