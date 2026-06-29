"""
Moisture Stress Detection Model
-------------------------------
This module trains and evaluates a Random Forest classifier to detect moisture stress levels
(No Stress, Mild Stress, Severe Stress) in agricultural fields using Sentinel-1 and Sentinel-2 features.

Biophysical Indicators of Stress:
- NDWI (Canopy Water Content): Drops significantly under stress.
- MSI (Moisture Stress Index): Rises as leaf water content declines.
- SAR VV/VH Backscatter: Drops as dielectric constant of vegetation and soil declines (less water).
"""

import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

class MoistureStressTrainer:
    def __init__(self, dataset_path="data/processed/features_dataset.csv", model_dir="src/models"):
        self.dataset_path = dataset_path
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)
        
        if not os.path.exists(self.dataset_path):
            raise FileNotFoundError(f"Features dataset CSV not found at: {self.dataset_path}")
            
        self.df = pd.read_csv(self.dataset_path)

    def augment_dataset(self, target_samples=250):
        """
        Augments the dataset while maintaining realistic stress correlations
        (e.g., Severe Stress must correlate with high MSI and low NDWI).
        """
        print(f"[MoistureStress] Augmenting dataset from {len(self.df)} to {target_samples} samples...")
        np.random.seed(42)
        
        feature_cols = [col for col in self.df.columns if "_mean" in col or "_std" in col]
        
        augmented_rows = []
        # Group by stress level to preserve signatures
        for stress_name, group in self.df.groupby("stress_level"):
            num_base_samples = len(group)
            samples_to_add = target_samples // len(self.df.stress_level.unique()) - num_base_samples
            
            # Original samples
            for _, row in group.iterrows():
                augmented_rows.append(row.to_dict())
                
            # Synthesize samples
            for _ in range(samples_to_add):
                base_row = group.sample(n=1).iloc[0].to_dict()
                
                # Add minor noise
                for col in feature_cols:
                    base_val = base_row[col]
                    noise = np.random.normal(0, abs(base_val) * 0.04 + 0.001)
                    base_row[col] = base_val + noise
                    
                base_row["field_id"] = np.random.randint(10, 100)
                augmented_rows.append(base_row)
                
        return pd.DataFrame(augmented_rows)

    def train(self):
        # 1. Prepare data
        aug_df = self.augment_dataset()
        
        # Define predictors and target
        feature_cols = [col for col in aug_df.columns if "_mean" in col or "_std" in col]
        X = aug_df[feature_cols]
        y = aug_df["stress_level"]
        
        # 2. Train-Test Split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        print(f"[MoistureStress] Feature dimension: {X_train.shape[1]} features.")
        print(f"[MoistureStress] Train size: {X_train.shape[0]}, Test size: {X_test.shape[0]}")
        
        # 3. Model Initialization
        model = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=6)
        
        # 4. Training
        model.fit(X_train, y_train)
        
        # 5. Prediction & Evaluation
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print("\n==============================================")
        print(f"MOISTURE STRESS CLASSIFICATION ACCURACY: {accuracy * 100:.2f}%")
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
            print(f"{rank + 1}. {feature_cols[idx]}: {importances[idx]:.4f}")
            
        # 7. Save Model
        model_path = os.path.join(self.model_dir, "moisture_stress.pkl")
        joblib.dump({"model": model, "features": feature_cols}, model_path)
        print(f"\n[MoistureStress] Saved trained model to: {model_path}")
        
        return model, feature_cols


if __name__ == "__main__":
    print("=== Training Moisture Stress Detection Model ===")
    trainer = MoistureStressTrainer()
    trainer.train()
