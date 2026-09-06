"""
=============================================================================
data_loader.py — Climate-Disease-Africa Project
=============================================================================
PURPOSE:
    Generate realistic synthetic disease outbreak and climate data for
    Africa. Simulates WHO / Africa CDC records for Malaria, Cholera,
    and Dengue across 10 African countries from 2010-2023.

    The synthetic data preserves real-world patterns:
    - Malaria peaks in wet season (high rainfall months)
    - Cholera spikes after flooding events
    - Dengue rises with temperature above 25°C and high humidity
    - All diseases show strong seasonal cycles

AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
=============================================================================
"""

import numpy as np
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Project paths ─────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW     = PROJECT_ROOT / "data" / "raw"
DATA_PROC    = PROJECT_ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROC.mkdir(parents=True, exist_ok=True)

# ── Study countries ───────────────────────────────────────────────────────────
# 10 African countries with high disease burden (WHO data)
COUNTRIES = {
    "Nigeria":          {"lat": 9.0,  "lon": 8.6,  "pop": 220_000_000, "zone": "tropical"},
    "Democratic Republic of Congo": {"lat": -4.0, "lon": 21.7, "pop": 100_000_000, "zone": "equatorial"},
    "Ghana":            {"lat": 7.9,  "lon": -1.0, "pop": 33_000_000,  "zone": "tropical"},
    "Ethiopia":         {"lat": 9.1,  "lon": 40.5, "pop": 120_000_000, "zone": "semi-arid"},
    "Tanzania":         {"lat": -6.3, "lon": 34.8, "pop": 63_000_000,  "zone": "tropical"},
    "Mozambique":       {"lat": -18.7,"lon": 35.5, "pop": 33_000_000,  "zone": "tropical"},
    "Mali":             {"lat": 17.6, "lon": -4.0, "pop": 22_000_000,  "zone": "sahel"},
    "Cameroon":         {"lat": 3.8,  "lon": 11.5, "pop": 27_000_000,  "zone": "equatorial"},
    "Uganda":           {"lat": 1.4,  "lon": 32.2, "pop": 47_000_000,  "zone": "equatorial"},
    "Senegal":          {"lat": 14.5, "lon": -14.5,"pop": 17_000_000,  "zone": "sahel"},
}

START_YEAR = 2010
END_YEAR   = 2023


class DiseaseDataLoader:
    """
    Loads or generates climate-disease data for Africa.

    Attributes:
        random_seed (int): For reproducibility
        countries (dict):  Country metadata
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed = random_seed
        np.random.seed(random_seed)
        print("✅ DiseaseDataLoader initialized")
        print(f"   Countries : {list(COUNTRIES.keys())}")
        print(f"   Period    : {START_YEAR}–{END_YEAR}")

    def load_all_data(self, force_regenerate: bool = False) -> pd.DataFrame:
        """Load existing data or generate fresh synthetic dataset."""
        path = DATA_PROC / "climate_disease_merged.csv"
        if path.exists() and not force_regenerate:
            print(f"📂 Loading existing data from {path}")
            df = pd.read_csv(path, parse_dates=["date"])
            print(f"   {len(df):,} records loaded")
            return df

        print("🔄 Generating synthetic climate-disease dataset...")
        climate_df = self._generate_climate_data()
        disease_df = self._generate_disease_data(climate_df)
        df = self._merge_data(climate_df, disease_df)
        df.to_csv(path, index=False)
        print(f"💾 Saved to {path} — {len(df):,} records")
        return df

    def _generate_climate_data(self) -> pd.DataFrame:
        """
        Generate monthly climate data for each country.

        Variables:
        - temperature_mean: Monthly average (°C)
        - precipitation: Monthly total rainfall (mm)
        - humidity: Relative humidity (%)
        - ndvi: Vegetation index (proxy for mosquito habitat)
        - flood_index: 0-1 score based on rainfall extremes
        """
        print("  🌡️ Generating climate data...")
        dates = pd.date_range(f"{START_YEAR}-01-01", f"{END_YEAR}-12-01", freq="MS")
        records = []

        for country, meta in COUNTRIES.items():
            zone  = meta["zone"]
            n     = len(dates)
            months = dates.month

            # ── Base temperature by climate zone
            base_temp = {
                "tropical": 27, "equatorial": 25,
                "semi-arid": 26, "sahel": 30
            }[zone]

            temp = (base_temp
                    + 3 * np.sin(2*np.pi*months/12)
                    + np.linspace(0, 0.4, n)          # warming trend
                    + np.random.normal(0, 0.8, n))

            # ── Precipitation (strong seasonal cycle)
            base_precip = {
                "tropical": 120, "equatorial": 160,
                "semi-arid": 60,  "sahel": 70
            }[zone]

            precip = np.maximum(0,
                base_precip
                + base_precip * 0.7 * np.sin(2*np.pi*months/12 + np.pi/2)
                + np.random.exponential(base_precip * 0.3, n))

            # ── Humidity
            humidity = np.clip(
                55 + 0.15 * precip + np.random.normal(0, 4, n), 20, 100)

            # ── NDVI (greens up after rain with ~1 month lag)
            ndvi = np.clip(
                0.35 + 0.15 * np.sin(2*np.pi*months/12 + np.pi/2 + 0.3)
                + np.random.normal(0, 0.03, n), 0.05, 0.85)

            # ── Flood index (extreme rainfall events)
            flood_index = np.clip(
                (precip - precip.mean()) / (precip.std() + 1e-6) * 0.2
                + np.random.exponential(0.05, n), 0, 1)

            for i, date in enumerate(dates):
                records.append({
                    "date":             date,
                    "country":          country,
                    "latitude":         meta["lat"],
                    "longitude":        meta["lon"],
                    "population":       meta["pop"],
                    "climate_zone":     zone,
                    "temperature_mean": round(float(temp[i]),   2),
                    "precipitation":    round(float(precip[i]), 2),
                    "humidity":         round(float(humidity[i]),1),
                    "ndvi":             round(float(ndvi[i]),   4),
                    "flood_index":      round(float(flood_index[i]), 4),
                })

        df = pd.DataFrame(records)
        df.to_csv(DATA_RAW / "climate_raw.csv", index=False)
        print(f"     {len(df):,} climate records generated")
        return df

    def _generate_disease_data(self, climate_df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate disease case counts driven by climate variables.

        MALARIA:
            Driven by rainfall (mosquito breeding) and temperature.
            Peaks 1-2 months after peak rainfall.
            Anopheles mosquitoes thrive at 20-30°C and high humidity.

        CHOLERA:
            Driven by flooding and lack of clean water.
            Spikes immediately after heavy rainfall / flood events.
            Linked to water contamination.

        DENGUE:
            Driven by temperature > 25°C and stagnant water.
            Aedes aegypti mosquitoes thrive in warm, humid conditions.
            Rising with climate change across Africa.
        """
        print("  🦟 Generating disease data...")
        records = []

        for country, meta in COUNTRIES.items():
            country_df = climate_df[climate_df["country"] == country].sort_values("date")
            n   = len(country_df)
            pop = meta["pop"]

            temp   = country_df["temperature_mean"].values
            precip = country_df["precipitation"].values
            humid  = country_df["humidity"].values
            flood  = country_df["flood_index"].values

            # ── MALARIA
            # Peaks when rainfall is high and temperature is 20-30°C
            # 1-month lag from rainfall to case spike
            precip_lag1     = np.roll(precip, 1); precip_lag1[0] = precip[0]
            temp_factor     = np.clip((temp - 15) / 15, 0, 1)
            malaria_base    = pop / 1_000_000 * 800
            malaria_climate = precip_lag1 * 3 * temp_factor * (humid / 100)
            malaria_cases   = np.maximum(0,
                malaria_base
                + malaria_climate
                + np.random.negative_binomial(5, 0.5, n) * 10).astype(int)

            # ── CHOLERA
            # Spikes after flooding, linked to contaminated water
            water_access    = {"tropical": 0.65, "equatorial": 0.55,
                               "semi-arid": 0.50, "sahel": 0.45}[meta["zone"]]
            cholera_base    = pop / 1_000_000 * 50
            cholera_climate = flood * 2000 * (1 - water_access)
            cholera_cases   = np.maximum(0,
                cholera_base
                + cholera_climate
                + np.random.negative_binomial(3, 0.6, n) * 5).astype(int)

            # ── DENGUE
            # Driven by temperature > 25°C and humidity
            dengue_temp     = np.where(temp > 25, (temp - 25) * 40, 0)
            dengue_humid    = np.clip((humid - 60) / 40, 0, 1) * 80
            dengue_base     = pop / 1_000_000 * 30
            dengue_cases    = np.maximum(0,
                dengue_base
                + dengue_temp + dengue_humid
                + np.random.negative_binomial(2, 0.7, n) * 3).astype(int)

            # ── Outbreak flag (binary: 1 = outbreak month)
            # Outbreak = cases exceed 90th percentile for that country
            malaria_thresh  = np.percentile(malaria_cases, 90)
            cholera_thresh  = np.percentile(cholera_cases, 90)
            dengue_thresh   = np.percentile(dengue_cases,  90)

            outbreak_malaria = (malaria_cases >= malaria_thresh).astype(int)
            outbreak_cholera = (cholera_cases >= cholera_thresh).astype(int)
            outbreak_dengue  = (dengue_cases  >= dengue_thresh).astype(int)

            # ── Combined outbreak risk score (0-1)
            outbreak_risk = np.clip(
                (malaria_cases / (malaria_cases.max() + 1) * 0.5 +
                 cholera_cases / (cholera_cases.max() + 1) * 0.3 +
                 dengue_cases  / (dengue_cases.max()  + 1) * 0.2), 0, 1)

            for i, (_, row) in enumerate(country_df.iterrows()):
                records.append({
                    "date":             row["date"],
                    "country":          country,
                    "malaria_cases":    int(malaria_cases[i]),
                    "cholera_cases":    int(cholera_cases[i]),
                    "dengue_cases":     int(dengue_cases[i]),
                    "outbreak_malaria": int(outbreak_malaria[i]),
                    "outbreak_cholera": int(outbreak_cholera[i]),
                    "outbreak_dengue":  int(outbreak_dengue[i]),
                    "outbreak_risk":    round(float(outbreak_risk[i]), 4),
                    "water_access_pct": round(water_access * 100, 1),
                })

        df = pd.DataFrame(records)
        df.to_csv(DATA_RAW / "disease_raw.csv", index=False)
        print(f"     {len(df):,} disease records generated")
        return df

    def _merge_data(self, climate_df: pd.DataFrame,
                    disease_df: pd.DataFrame) -> pd.DataFrame:
        """Merge climate and disease datasets on date + country."""
        print("  🔗 Merging datasets...")
        df = pd.merge(climate_df, disease_df, on=["date", "country"], how="inner")
        df["year"]  = df["date"].dt.year
        df["month"] = df["date"].dt.month
        df = df.sort_values(["country", "date"]).reset_index(drop=True)
        print(f"     Final shape: {df.shape}")
        return df


if __name__ == "__main__":
    loader = DiseaseDataLoader()
    df = loader.load_all_data(force_regenerate=True)
    print(df.head())
    print(f"\nCountries: {df['country'].unique().tolist()}")
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
