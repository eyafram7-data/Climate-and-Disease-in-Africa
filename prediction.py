"""
=============================================================================
prediction.py — Climate-Disease-Africa Project
=============================================================================
PURPOSE:
    Use trained ML model to forecast future outbreak risk and run
    scenario analysis showing how climate change will affect disease
    burden across Africa.

AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR   = PROJECT_ROOT / "images"

SCENARIOS = {
    "Optimistic": {
        "desc":   "Improved sanitation, stable climate",
        "color":  "#27AE60",
        "temp_delta":   +0.001,
        "precip_mult":  1.002,
        "flood_delta":  -0.003,
    },
    "Baseline": {
        "desc":   "Current trends continue",
        "color":  "#F39C12",
        "temp_delta":   +0.002,
        "precip_mult":  1.000,
        "flood_delta":  +0.000,
    },
    "Pessimistic": {
        "desc":   "Rapid warming, increased flooding",
        "color":  "#E74C3C",
        "temp_delta":   +0.005,
        "precip_mult":  1.004,
        "flood_delta":  +0.008,
    },
}


class OutbreakPredictor:
    """
    Forecasts disease outbreak risk under different climate scenarios.
    """

    def __init__(self, model, feature_names: list, forecast_months: int = 24):
        self.model          = model
        self.feature_names  = feature_names
        self.forecast_months = forecast_months
        print(f"✅ OutbreakPredictor ready — {forecast_months} month horizon")

    def forecast_scenario(self, df_features: pd.DataFrame,
                           scenario: str = "Baseline") -> pd.DataFrame:
        """Generate rolling forecast for a given climate scenario."""
        scen = SCENARIOS[scenario]
        last_row  = df_features[self.feature_names].iloc[-1].copy()
        last_date = df_features["date"].iloc[-1] if "date" in df_features.columns \
                    else pd.Timestamp("2023-12-01")

        records = []
        for step in range(self.forecast_months):
            forecast_date = last_date + pd.DateOffset(months=step+1)
            row = last_row.copy()

            if "temperature_mean" in self.feature_names:
                row["temperature_mean"] = row.get("temperature_mean", 26) \
                                          + scen["temp_delta"] * (step+1)
            if "precipitation" in self.feature_names:
                row["precipitation"] = max(0,
                    row.get("precipitation", 80) * (scen["precip_mult"]**(step+1)))
            if "flood_index" in self.feature_names:
                row["flood_index"] = np.clip(
                    row.get("flood_index", 0.2) + scen["flood_delta"]*(step+1), 0, 1)
            if "month_sin" in self.feature_names:
                m = forecast_date.month
                row["month_sin"] = np.sin(2*np.pi*m/12)
                row["month_cos"] = np.cos(2*np.pi*m/12)

            fv   = np.array([row.get(f, 0) for f in self.feature_names]).reshape(1,-1)
            risk = float(np.clip(self.model.predict(fv)[0], 0, 1))

            records.append({
                "date":         forecast_date,
                "scenario":     scenario,
                "outbreak_risk":risk,
                "step":         step+1,
            })
            last_row = row

        return pd.DataFrame(records)

    def run_all_scenarios(self, df_features: pd.DataFrame) -> pd.DataFrame:
        """Run all three scenarios and return combined DataFrame."""
        print("\n🔮 Running scenario analysis...")
        frames = []
        for s in SCENARIOS:
            frames.append(self.forecast_scenario(df_features, scenario=s))
            print(f"   ✅ {s} scenario complete")
        return pd.concat(frames, ignore_index=True)

    def plot_scenario_forecast(self, df_hist: pd.DataFrame,
                                scenarios: pd.DataFrame,
                                save: bool = True) -> plt.Figure:
        """Plot historical + 3-scenario forecast."""
        print("\n📉 Plotting outbreak risk forecast...")

        fig, ax = plt.subplots(figsize=(14, 6), constrained_layout=True)
        fig.suptitle("🔮  Disease Outbreak Risk Forecast — 3 Climate Scenarios",
                     fontsize=14, fontweight="bold")

        # Historical
        hist = df_hist.groupby("date")["outbreak_risk"].mean().reset_index()
        hist_recent = hist[hist["date"] >= "2021-01-01"]
        ax.plot(hist_recent["date"], hist_recent["outbreak_risk"],
                color="#2C3E50", lw=2.5, label="Historical Risk", zorder=5)

        forecast_start = scenarios["date"].min()
        ax.axvline(forecast_start, color="gray", ls="--", lw=1.5, alpha=0.7)
        ax.text(forecast_start, 0.85, "  Forecast →", color="gray", fontsize=9)

        for s_name, s_info in SCENARIOS.items():
            s_data = scenarios[scenarios["scenario"]==s_name].sort_values("date")
            ax.plot(s_data["date"], s_data["outbreak_risk"],
                    color=s_info["color"], lw=2.5,
                    label=f"{s_name}: {s_info['desc']}")
            ax.fill_between(s_data["date"], s_data["outbreak_risk"],
                            alpha=0.07, color=s_info["color"])

        # Risk threshold lines
        for y, label, color in [
            (0.3, "Low Risk",  "#27AE60"),
            (0.5, "High Risk", "#E74C3C"),
        ]:
            ax.axhline(y, color=color, ls=":", lw=1.2, alpha=0.6)
            ax.text(hist_recent["date"].min(), y+0.01, label,
                    fontsize=7, color=color)

        ax.set_ylim(0, 1)
        ax.set_ylabel("Outbreak Risk Score (0–1)", fontsize=11)
        ax.set_xlabel("Date")
        ax.legend(loc="upper left", fontsize=9)

        if save:
            fig.savefig(IMAGES_DIR/"scenario_forecast.png",
                        bbox_inches="tight", dpi=150)
            print(f"   Saved: scenario_forecast.png")
        return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader   import DiseaseDataLoader
    from preprocessing import DiseasePreprocessor
    from model         import OutbreakModelTrainer

    df      = DiseaseDataLoader().load_all_data()
    result  = DiseasePreprocessor().run_pipeline(df)
    trainer = OutbreakModelTrainer()
    trainer.train_all(result["X_train_scaled"], result["y_train"],
                      result["X_test_scaled"],  result["y_test"],
                      feature_names=result["feature_names"])

    predictor = OutbreakPredictor(
        model=trainer.best_model,
        feature_names=result["feature_names"]
    )
    scenarios = predictor.run_all_scenarios(result["df_features"])
    predictor.plot_scenario_forecast(result["df_features"], scenarios)
    print("\n✅ Prediction complete!")
