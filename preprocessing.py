"""
=============================================================================
preprocessing.py — Climate-Disease-Africa Project
=============================================================================
PURPOSE:
    Clean and engineer features from the raw climate-disease dataset.
    Creates features that capture how climate conditions drive outbreaks
    with time delays (lags) and accumulated effects (rolling windows).

AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"


class DiseasePreprocessor:
    """
    Cleans data and engineers features for disease outbreak prediction.
    """

    def __init__(self):
        self.scaler          = StandardScaler()
        self.feature_columns = None
        print("✅ DiseasePreprocessor initialized")

    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicates, clip outliers, fill missing values."""
        print("\n🧹 Cleaning data...")
        df = df.copy().drop_duplicates()

        # Clip disease cases to realistic range
        for col in ["malaria_cases", "cholera_cases", "dengue_cases"]:
            if col in df.columns:
                upper = df[col].quantile(0.99)
                df[col] = df[col].clip(0, upper)

        # Fill any missing numeric values
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        print(f"   ✅ Clean shape: {df.shape}")
        return df.reset_index(drop=True)

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features that capture climate-disease relationships.

        KEY INSIGHT:
        Diseases don't respond to climate instantly. Mosquitoes need
        time to breed after rain. Cholera spreads after flood waters
        contaminate wells. We capture this with lag features.

        Features created:
        ── Lag features (delayed climate effects)
        ── Rolling windows (accumulated climate stress)
        ── Heat-humidity stress index
        ── Flood severity score
        ── Seasonal indicators (wet/dry season)
        ── Country one-hot encoding
        """
        print("\n⚙️  Engineering features...")
        df = df.copy().sort_values(["country", "date"])

        # ── Cyclical month encoding
        df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
        df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)

        # ── Wet season flag (months 4-10 for most of Africa)
        df["wet_season"] = df["month"].between(4, 10).astype(int)

        # ── Lag features (1, 2, 3 months)
        for lag in [1, 2, 3]:
            for col in ["precipitation", "temperature_mean",
                        "flood_index", "humidity", "ndvi"]:
                df[f"{col}_lag{lag}"] = (
                    df.groupby("country")[col].shift(lag))

        # ── Rolling windows
        for window in [3, 6]:
            df[f"precip_roll{window}"] = (
                df.groupby("country")["precipitation"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean()))
            df[f"temp_roll{window}"] = (
                df.groupby("country")["temperature_mean"]
                .transform(lambda x: x.rolling(window, min_periods=1).mean()))
            df[f"flood_roll{window}"] = (
                df.groupby("country")["flood_index"]
                .transform(lambda x: x.rolling(window, min_periods=1).sum()))

        # ── Heat-Humidity Stress Index
        # High temperature + high humidity = ideal for mosquito breeding
        df["heat_humid_stress"] = (
            df["temperature_mean"] * df["humidity"] / 100)

        # ── Temperature anomaly (deviation from country monthly mean)
        df["temp_anomaly"] = (
            df["temperature_mean"] -
            df.groupby(["country","month"])["temperature_mean"]
            .transform("mean"))

        # ── Precipitation anomaly
        df["precip_anomaly"] = (
            df["precipitation"] -
            df.groupby(["country","month"])["precipitation"]
            .transform("mean"))

        # ── Country encoding
        df = pd.get_dummies(df, columns=["climate_zone"], prefix="zone")

        # ── Drop NaN rows from lagging
        before = len(df)
        df = df.dropna()
        print(f"   Dropped {before-len(df)} NaN rows from lagging")

        df.to_csv(DATA_PROC / "features_engineered.csv", index=False)
        print(f"   ✅ Features: {df.shape[1]} columns, {len(df):,} rows")
        return df.reset_index(drop=True)

    def prepare_for_model(self, df: pd.DataFrame,
                           target: str = "outbreak_risk") -> dict:
        """
        Select features, split train/test, and scale.

        Args:
            df: Feature-engineered DataFrame
            target: Column to predict

        Returns:
            dict with X_train, X_test, y_train, y_test
        """
        print(f"\n🎯 Preparing for model (target: {target})...")

        exclude = ["date", "country", "malaria_cases", "cholera_cases",
                   "dengue_cases", "outbreak_malaria", "outbreak_cholera",
                   "outbreak_dengue", "outbreak_risk", "latitude",
                   "longitude", "population", "water_access_pct"]

        self.feature_columns = [
            c for c in df.columns
            if c not in exclude
            and df[c].dtype in [np.float64, np.int64, np.uint8, bool]
        ]

        X = df[self.feature_columns]
        y = df[target]

        # Temporal split — last 20% of time is test set
        split = int(len(X) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        X_train_s = self.scaler.fit_transform(X_train)
        X_test_s  = self.scaler.transform(X_test)

        print(f"   Features  : {len(self.feature_columns)}")
        print(f"   Train rows: {len(X_train):,}")
        print(f"   Test rows : {len(X_test):,}")

        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "X_train_scaled": X_train_s,
            "X_test_scaled":  X_test_s,
            "feature_names":  self.feature_columns,
        }

    def run_pipeline(self, df: pd.DataFrame) -> dict:
        """Run full preprocessing pipeline."""
        df_clean    = self.clean_data(df)
        df_features = self.engineer_features(df_clean)
        result      = self.prepare_for_model(df_features)
        result["df_features"] = df_features
        return result


if __name__ == "__main__":
    from data_loader import DiseaseDataLoader
    loader = DiseaseDataLoader()
    df     = loader.load_all_data()
    prep   = DiseasePreprocessor()
    result = prep.run_pipeline(df)
    print(f"\nX_train shape: {result['X_train'].shape}")
