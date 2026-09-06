"""
=============================================================================
run_pipeline.py — Climate-Disease-Africa
Master script that runs the entire project end to end.

USAGE:
    python run_pipeline.py

AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
=============================================================================
"""

import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def banner():
    print("\n" + "="*60)
    print("  🦟  CLIMATE-DISEASE AFRICA PIPELINE")
    print("  Predicting Outbreak Risk Using Climate Data & ML")
    print("="*60)
    print("  Author : Emmanuel Yaw Afram (Prestige)")
    print("  Email  : eyafram7@gmail.com")
    print("  GitHub : github.com/eyafram7-data")
    print("="*60 + "\n")


def step(n, title):
    print(f"\n{'─'*55}")
    print(f"  STEP {n}: {title}")
    print(f"{'─'*55}")


def main():
    banner()
    total_start = time.time()

    # ── STEP 1: DATA GENERATION ───────────────────────────────────────────────
    step(1, "DATA LOADING & GENERATION 📦")
    from data_loader import DiseaseDataLoader
    import pandas as pd

    loader = DiseaseDataLoader(random_seed=42)
    df = loader.load_all_data(force_regenerate=True)
    df["date"] = pd.to_datetime(df["date"])
    print(f"\n  ✅ Dataset ready")
    print(f"     Shape    : {df.shape}")
    print(f"     Countries: {df['country'].nunique()}")
    print(f"     Period   : {df['date'].min().date()} → {df['date'].max().date()}")

    # ── STEP 2: PREPROCESSING ────────────────────────────────────────────────
    step(2, "PREPROCESSING & FEATURE ENGINEERING ⚙️")
    from preprocessing import DiseasePreprocessor

    prep   = DiseasePreprocessor()
    result = prep.run_pipeline(df)
    print(f"\n  ✅ Preprocessing complete")
    print(f"     Features : {len(result['feature_names'])}")
    print(f"     Train    : {len(result['X_train']):,} rows")
    print(f"     Test     : {len(result['X_test']):,} rows")

    # ── STEP 3: VISUALIZATIONS ────────────────────────────────────────────────
    step(3, "GENERATING VISUALIZATIONS 📊")
    from visualization import DiseaseVisualizer

    viz = DiseaseVisualizer()
    viz.plot_disease_trends(df)
    viz.plot_climate_disease_correlation(df)
    viz.plot_seasonal_patterns(df)
    viz.plot_country_comparison(df)
    viz.create_outbreak_map(df)
    print(f"\n  ✅ All visualizations saved → images/")

    # ── STEP 4: MODEL TRAINING ────────────────────────────────────────────────
    step(4, "ML MODEL TRAINING 🤖")
    import numpy as np
    from model import OutbreakModelTrainer

    trainer = OutbreakModelTrainer(random_seed=42)
    model_results = trainer.train_all(
        result["X_train_scaled"], result["y_train"],
        result["X_test_scaled"],  result["y_test"],
        feature_names=result["feature_names"]
    )
    trainer.print_leaderboard()
    trainer.save_best_model()

    # Plot model comparison
    viz.plot_model_comparison(model_results)

    # ── STEP 5: FORECASTING ───────────────────────────────────────────────────
    step(5, "SCENARIO FORECASTING 🔮")
    from prediction import OutbreakPredictor

    predictor = OutbreakPredictor(
        model=trainer.best_model,
        feature_names=result["feature_names"],
        forecast_months=24
    )
    df_features = result["df_features"]
    if "date" not in df_features.columns:
        df_features = df_features.copy()
        df_features["date"] = df["date"].values[:len(df_features)]

    scenarios = predictor.run_all_scenarios(df_features)
    predictor.plot_scenario_forecast(df_features, scenarios)

    print(f"\n  📋 Scenario Summary (24-month horizon):")
    for scen in ["Optimistic", "Baseline", "Pessimistic"]:
        s = scenarios[scenarios["scenario"] == scen]
        if len(s):
            print(f"     {scen:12s}: Risk {s['outbreak_risk'].iloc[0]:.3f} "
                  f"→ {s['outbreak_risk'].iloc[-1]:.3f} "
                  f"(net {s['outbreak_risk'].iloc[-1]-s['outbreak_risk'].iloc[0]:+.3f})")

    # ── DONE ──────────────────────────────────────────────────────────────────
    total = time.time() - total_start
    print("\n" + "="*60)
    print("  ✅  PIPELINE COMPLETE")
    print("="*60)
    print(f"  Runtime     : {total:.1f}s ({total/60:.1f} min)")
    print(f"  Dataset     : {df.shape[0]:,} records")
    print(f"  Best model  : {trainer.best_model_name}")
    print(f"  R²          : {model_results[trainer.best_model_name]['r2']:.4f}")
    print(f"\n  📂 Outputs:")
    print(f"     data/raw/             ← Raw generated datasets")
    print(f"     data/processed/       ← Cleaned + feature data")
    print(f"     images/               ← All visualization plots")
    print(f"\n  🚀 Next step:")
    print(f"     streamlit run app.py  ← Launch dashboard")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
