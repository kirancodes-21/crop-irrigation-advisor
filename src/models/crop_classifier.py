"""
Crop Type Classification Model
-------------------------------
This module trains and evaluates a Random Forest classifier to identify crop types
(e.g., Rice, Wheat, Cotton, Maize, Sugarcane) based on Sentinel-1 and Sentinel-2 features.

Why Random Forest?
- Handles multi-collinear remote sensing bands effectively.
- High accuracy on tabular datasets.
- Provides Feature Importance metrics, explaining which bands (e.g., NIR vs VH) were critical.

Data Augmentation:
- Since our physical simulation yields 24 spatial-temporal rows, we apply a robust augmentation 
  step by introducing Gaussian noise to create a realistic 200-sample training dataset.
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class CropClassifierTrainer:
    def __init__(self, dataset_path="data/processed/features_dataset.csv", model_dir="src/models"):
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Features dataset CSV not found at: {self.dataset_path}")
            
        self.df = pd.read_csv(self.dataset_path)
        
    def augment_dataset(self, target_samples=250):
        """
        Augments the small simulation dataset by adding minor Gaussian noise to numerical features
        for each class, simulating diverse field conditions across different farms.
        """
        print(f"[CropClassifier] Augmenting dataset from {len(self.df)} to {target_samples} samples...")
        np.random.seed(42)
        
        # Features columns (all columns starting with band names or indices)
        feature_cols = [col for col in self.df.columns if "_mean" in col or "_std" in col]
        
        augmented_rows = []
        # Group by crop type to preserve statistical characteristics per crop
        for crop_name, group in self.df.groupby("crop_type"):
            num_base_samples = len(group)
            samples_to_add = target_samples // len(self.df.crop_type.unique()) - num_base_samples
            
            # Keep original samples
            for _, row in group.iterrows():
                augmented_rows.append(row.to_dict())
                
            # Synthesize new samples
            for _ in range(samples_to_add):
                # Pick a random base row
                base_row = group.sample(n=1).iloc[0].to_dict()
                
                # Add slight Gaussian noise to numeric columns (standard deviation ~ 5% of base value)
                for col in feature_cols:
                    base_val = base_row[col]
                    noise = np.random.normal(0, abs(base_val) * 0.05 + 0.001)
                    base_row[col] = base_val + noise
                    
                # Growth stage and moisture stress are randomized slightly to match multi-temporal behaviors
                base_row["growth_stage"] = np.random.choice(["Sowing", "Vegetative", "Flowering", "Maturity"])
                base_row["stress_level"] = np.random.choice(["No Stress", "Mild Stress", "Severe Stress"])
                base_row["field_id"] = np.random.randint(10, 100) # new dummy field ID
                
                augmented_rows.append(base_row)
                
        return pd.DataFrame(augmented_rows)

    def train(self):
        # 1. Prepare data
        aug_df = self.augment_dataset()
        
        # Define predictors and target
        feature_cols = [col for col in aug_df.columns if "_mean" in col or "_std" in col]
        X = aug_df[feature_cols]
        y = aug_df["crop_type"]
        
        # 2. Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"[CropClassifier] Feature dimension: {X_train.shape[1]} features.")
        print(f"[CropClassifier] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # 3. Model Initialization
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=8)
        
        # 4. Training
        model.fit(X_train, y_train)
        
        # 5. Predictions & Evaluation
        y_pred = model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        print("\n==============================================")
        print(f"CROP CLASSIFICATION ACCURACY: {accuracy * 100:.2f}%")
        print("==============================================")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        # 6. Feature Importance analysis
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nTop 5 Most Discriminative Features:")
        for rank in range(5):
            idx = indices[rank]
            print(f"{rank + 1}. {feature_cols[idx]}: {importances[idx]:.4f}")
            
        # 7. Save Model & Feature List
        model_path = os.path.join(self.model_dir, "crop_classifier.pkl")
        joblib.dump({"model": model, "features": feature_cols}, model_path)
        print(f"\n[CropClassifier] Saved trained model to: {model_path}")
        
        return model, feature_cols


if __name__ == "__main__":
    print("=== Training Crop Classifier Model ===")
    trainer = CropClassifierTrainer()
    trainer.train()
