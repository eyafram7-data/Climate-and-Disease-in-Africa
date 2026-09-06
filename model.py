"""
=============================================================================
model.py — Climate-Disease-Africa Project
=============================================================================
PURPOSE:
    Train and evaluate ML models to predict disease outbreak risk.
    We treat this as REGRESSION (predicting a 0-1 risk score) so
    health authorities can see DEGREE of risk, not just yes/no.

AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
=============================================================================
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import (mean_squared_error, mean_absolute_error,
                              r2_score)
from sklearn.model_selection import cross_val_score
import xgboost as xgb
import joblib, time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
DATA_PROC    = PROJECT_ROOT / "data" / "processed"


class OutbreakModelTrainer:
    """
    Trains ML models to predict disease outbreak risk score (0-1).
    """

    def __init__(self, random_seed: int = 42):
        self.random_seed     = random_seed
        self.results         = {}
        self.best_model_name = None
        self.best_model      = None
        self.feature_names   = None

        self.models = {
            "Linear Regression": LinearRegression(),
            "Ridge Regression":  Ridge(alpha=1.0),
            "Random Forest":     RandomForestRegressor(
                n_estimators=200, max_depth=10,
                min_samples_leaf=5, random_state=random_seed, n_jobs=-1),
            "XGBoost":           xgb.XGBRegressor(
                n_estimators=400, learning_rate=0.05, max_depth=5,
                subsample=0.8, colsample_bytree=0.8,
                random_state=random_seed, verbosity=0, n_jobs=-1),
            "SVR":               SVR(kernel="rbf", C=10.0, epsilon=0.01),
        }
        print(f"✅ OutbreakModelTrainer ready — {len(self.models)} models")

    def train_all(self, X_train, y_train, X_test, y_test,
                  feature_names=None) -> dict:
        """Train all models and return results dict."""
        print("\n" + "="*55)
        print("  TRAINING DISEASE OUTBREAK MODELS")
        print("="*55)
        self.feature_names = feature_names

        X_tr = X_train.values if hasattr(X_train, "values") else X_train
        X_te = X_test.values  if hasattr(X_test,  "values") else X_test
        y_tr = np.array(y_train)
        y_te = np.array(y_test)

        for name, model in self.models.items():
            print(f"\n  🔧 Training {name}...")
            t0 = time.time()
            try:
                model.fit(X_tr, y_tr)
                y_pred     = model.predict(X_te)
                y_pred_tr  = model.predict(X_tr)
                y_pred     = np.clip(y_pred, 0, 1)

                rmse = np.sqrt(mean_squared_error(y_te, y_pred))
                mae  = mean_absolute_error(y_te, y_pred)
                r2   = r2_score(y_te, y_pred)
                tr2  = r2_score(y_tr, y_pred_tr)

                self.results[name] = {
                    "model": model, "y_pred": y_pred,
                    "rmse": rmse, "mae": mae, "r2": r2,
                    "train_r2": tr2,
                    "overfit": tr2 - r2,
                    "time_s": round(time.time() - t0, 2),
                }
                if hasattr(model, "feature_importances_") and feature_names:
                    self.results[name]["feature_importances"] = \
                        model.feature_importances_
                    self.results[name]["feature_names"] = feature_names

                print(f"     RMSE={rmse:.5f}  MAE={mae:.5f}  R²={r2:.5f}")
            except Exception as e:
                print(f"     ❌ Error: {e}")

        self.best_model_name = max(self.results, key=lambda k: self.results[k]["r2"])
        self.best_model      = self.results[self.best_model_name]["model"]
        print(f"\n🏆 Best: {self.best_model_name} "
              f"(R²={self.results[self.best_model_name]['r2']:.4f})")
        return self.results

    def print_leaderboard(self) -> pd.DataFrame:
        """Print formatted leaderboard."""
        print("\n" + "="*65)
        print("  OUTBREAK PREDICTION MODEL LEADERBOARD")
        print("="*65)
        rows = []
        for name, res in self.results.items():
            rows.append({
                "Model":    name,
                "RMSE ↓":  f"{res['rmse']:.5f}",
                "MAE ↓":   f"{res['mae']:.5f}",
                "R² ↑":    f"{res['r2']:.5f}",
                "Train R²":f"{res['train_r2']:.5f}",
                "Time(s)": f"{res['time_s']}",
                "🏆":      "BEST" if name == self.best_model_name else "",
            })
        df = pd.DataFrame(rows).sort_values("R² ↑", ascending=False)
        print(df.to_string(index=False))
        print("="*65)
        return df

    def save_best_model(self) -> str:
        """Save best model to disk."""
        path = DATA_PROC / "best_outbreak_model.pkl"
        joblib.dump({
            "model":         self.best_model,
            "model_name":    self.best_model_name,
            "feature_names": self.feature_names,
            "metrics":       {k: self.results[self.best_model_name][k]
                              for k in ["rmse","mae","r2"]},
        }, path)
        print(f"💾 Best model saved → {path}")
        return str(path)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader    import DiseaseDataLoader
    from preprocessing  import DiseasePreprocessor

    df     = DiseaseDataLoader().load_all_data()
    result = DiseasePreprocessor().run_pipeline(df)
    trainer = OutbreakModelTrainer()
    trainer.train_all(result["X_train_scaled"], result["y_train"],
                      result["X_test_scaled"],  result["y_test"],
                      feature_names=result["feature_names"])
    trainer.print_leaderboard()
    trainer.save_best_model()
