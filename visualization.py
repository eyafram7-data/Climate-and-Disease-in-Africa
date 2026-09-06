"""
=============================================================================
visualization.py — Climate-Disease-Africa Project
=============================================================================
AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from folium.plugins import HeatMap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent
IMAGES_DIR   = PROJECT_ROOT / "images"
IMAGES_DIR.mkdir(exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150, "figure.facecolor": "white",
    "axes.facecolor": "#F8F9FA", "axes.grid": True,
    "grid.alpha": 0.35, "font.size": 10,
})

DISEASE_COLORS = {
    "malaria": "#E74C3C",
    "cholera": "#3498DB",
    "dengue":  "#F39C12",
}
COUNTRY_PALETTE = px.colors.qualitative.Set2


class DiseaseVisualizer:
    """Creates all visualizations for the Climate-Disease-Africa project."""

    def __init__(self):
        print("✅ DiseaseVisualizer initialized")

    def plot_disease_trends(self, df: pd.DataFrame, save=True) -> plt.Figure:
        """Plot monthly case trends for all three diseases."""
        print("  📈 Plotting disease trends...")

        monthly = df.groupby("date").agg(
            malaria_cases=("malaria_cases", "sum"),
            cholera_cases=("cholera_cases", "sum"),
            dengue_cases=("dengue_cases",  "sum"),
        ).reset_index()

        fig, axes = plt.subplots(3, 1, figsize=(14, 10),
                                  sharex=True, constrained_layout=True)
        fig.suptitle("🦟  Disease Case Trends Across Africa (2010–2023)",
                     fontsize=14, fontweight="bold")

        for ax, (disease, color) in zip(axes, DISEASE_COLORS.items()):
            col = f"{disease}_cases"
            ax.plot(monthly["date"], monthly[col],
                    color=color, lw=1.5, alpha=0.7, label="Monthly")
            rolling = monthly[col].rolling(12, center=True, min_periods=6).mean()
            ax.plot(monthly["date"], rolling,
                    color="#2C3E50", lw=2.5, label="12-mo trend")
            ax.fill_between(monthly["date"], monthly[col], rolling,
                            alpha=0.1, color=color)
            ax.set_ylabel(f"{disease.title()} Cases")
            ax.legend(fontsize=8)

            # Annotate peak
            peak_idx = monthly[col].idxmax()
            ax.annotate(f"Peak: {monthly[col].max():,.0f}",
                        xy=(monthly["date"].iloc[peak_idx], monthly[col].max()),
                        xytext=(10, -20), textcoords="offset points",
                        fontsize=8, color=color,
                        arrowprops=dict(arrowstyle="->", color=color))

        axes[-1].set_xlabel("Date")
        if save:
            fig.savefig(IMAGES_DIR/"climate_disease_trends.png",
                        bbox_inches="tight", dpi=150)
            print(f"     Saved: climate_disease_trends.png")
        return fig

    def plot_climate_disease_correlation(self, df: pd.DataFrame,
                                          save=True) -> plt.Figure:
        """Plot correlation heatmap between climate and disease variables."""
        print("  🔥 Plotting correlation heatmap...")

        cols = ["malaria_cases","cholera_cases","dengue_cases",
                "temperature_mean","precipitation","humidity",
                "flood_index","ndvi","outbreak_risk"]
        cols = [c for c in cols if c in df.columns]
        corr = df[cols].corr()

        fig, axes = plt.subplots(1, 2, figsize=(16, 6),
                                  constrained_layout=True)
        fig.suptitle("🔥  Climate-Disease Correlation Analysis",
                     fontsize=14, fontweight="bold")

        mask = np.triu(np.ones_like(corr, dtype=bool))
        sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
                    cmap="RdBu_r", center=0, vmin=-1, vmax=1,
                    linewidths=0.5, square=True, ax=axes[0],
                    cbar_kws={"label":"Pearson r","shrink":0.8})
        axes[0].set_title("Full Correlation Matrix")
        axes[0].tick_params(axis="x", rotation=40)

        # Outbreak risk correlations
        risk_corr = corr["outbreak_risk"].drop("outbreak_risk").sort_values()
        colors = ["#E74C3C" if v < 0 else "#27AE60" for v in risk_corr]
        axes[1].barh(risk_corr.index, risk_corr.values,
                     color=colors, edgecolor="white")
        axes[1].axvline(0, color="black", lw=1)
        axes[1].set_title("Correlation with Outbreak Risk Score")
        axes[1].set_xlabel("Pearson r")

        if save:
            fig.savefig(IMAGES_DIR/"correlation_heatmap.png",
                        bbox_inches="tight", dpi=150)
            print(f"     Saved: correlation_heatmap.png")
        return fig

    def plot_seasonal_patterns(self, df: pd.DataFrame,
                                save=True) -> plt.Figure:
        """Plot average disease cases by month (seasonal climatology)."""
        print("  🗓️  Plotting seasonal patterns...")

        df_m = df.copy()
        df_m["month_name"] = df_m["date"].dt.strftime("%b")
        month_order = ["Jan","Feb","Mar","Apr","May","Jun",
                       "Jul","Aug","Sep","Oct","Nov","Dec"]

        seasonal = df_m.groupby("month")[
            ["malaria_cases","cholera_cases","dengue_cases",
             "precipitation","temperature_mean"]].mean()

        fig, axes = plt.subplots(2, 2, figsize=(14, 9),
                                  constrained_layout=True)
        fig.suptitle("🗓️  Seasonal Patterns — Diseases & Climate",
                     fontsize=14, fontweight="bold")

        months = range(1, 13)
        xlabels = ["Jan","Feb","Mar","Apr","May","Jun",
                   "Jul","Aug","Sep","Oct","Nov","Dec"]

        for ax, (disease, color) in zip(axes.flatten(), DISEASE_COLORS.items()):
            ax.bar(months, seasonal[f"{disease}_cases"],
                   color=color, alpha=0.75, edgecolor="white")
            ax.set_xticks(months); ax.set_xticklabels(xlabels, rotation=45)
            ax.set_title(f"{disease.title()} — Monthly Average Cases",
                         fontweight="bold")
            ax.set_ylabel("Average Cases")

        ax = axes[1][1]
        ax.plot(months, seasonal["precipitation"], color="#3498DB",
                lw=2.5, marker="o", markersize=4, label="Precipitation (mm)")
        ax2 = ax.twinx()
        ax2.plot(months, seasonal["temperature_mean"], color="#E74C3C",
                 lw=2.5, marker="s", markersize=4, label="Temperature (°C)")
        ax.set_xticks(months); ax.set_xticklabels(xlabels, rotation=45)
        ax.set_ylabel("Precipitation (mm)", color="#3498DB")
        ax2.set_ylabel("Temperature (°C)", color="#E74C3C")
        ax.set_title("Climate Drivers — Seasonal Pattern", fontweight="bold")
        ax.legend(loc="upper left", fontsize=8)
        ax2.legend(loc="upper right", fontsize=8)

        if save:
            fig.savefig(IMAGES_DIR/"seasonal_patterns.png",
                        bbox_inches="tight", dpi=150)
            print(f"     Saved: seasonal_patterns.png")
        return fig

    def plot_country_comparison(self, df: pd.DataFrame,
                                 save=True) -> plt.Figure:
        """Compare disease burden across African countries."""
        print("  🌍 Plotting country comparison...")

        country_stats = df.groupby("country").agg(
            malaria_total=("malaria_cases","sum"),
            cholera_total=("cholera_cases","sum"),
            dengue_total=("dengue_cases","sum"),
            outbreak_risk=("outbreak_risk","mean"),
        ).reset_index().sort_values("malaria_total", ascending=False)

        fig, axes = plt.subplots(2, 2, figsize=(15, 10),
                                  constrained_layout=True)
        fig.suptitle("🌍  Disease Burden by Country (2010–2023)",
                     fontsize=14, fontweight="bold")

        for ax, (disease, color) in zip(axes.flatten()[:3],
                                         DISEASE_COLORS.items()):
            col = f"{disease}_total"
            bars = ax.barh(country_stats["country"],
                           country_stats[col] / 1000,
                           color=color, alpha=0.8, edgecolor="white")
            ax.set_title(f"Total {disease.title()} Cases (thousands)",
                         fontweight="bold")
            ax.set_xlabel("Cases (thousands)")
            for bar, val in zip(bars, country_stats[col]/1000):
                ax.text(val+max(country_stats[col]/1000)*0.01,
                        bar.get_y()+bar.get_height()/2,
                        f"{val:.0f}k", va="center", fontsize=8)

        ax = axes[1][1]
        bars = ax.barh(country_stats["country"],
                       country_stats["outbreak_risk"],
                       color="#8E44AD", alpha=0.8, edgecolor="white")
        ax.set_title("Mean Outbreak Risk Score (0–1)", fontweight="bold")
        ax.set_xlabel("Risk Score")
        ax.axvline(0.5, color="red", lw=1.5, ls="--", label="High Risk Threshold")
        ax.legend(fontsize=8)

        if save:
            fig.savefig(IMAGES_DIR/"country_comparison.png",
                        bbox_inches="tight", dpi=150)
            print(f"     Saved: country_comparison.png")
        return fig

    def create_outbreak_map(self, df: pd.DataFrame, save=True) -> folium.Map:
        """Create interactive Folium map showing outbreak risk by country."""
        print("  🗺️  Creating interactive outbreak risk map...")

        country_risk = df.groupby("country").agg(
            latitude=("latitude","first"),
            longitude=("longitude","first"),
            outbreak_risk=("outbreak_risk","mean"),
            malaria_cases=("malaria_cases","sum"),
            cholera_cases=("cholera_cases","sum"),
            dengue_cases=("dengue_cases","sum"),
        ).reset_index()

        m = folium.Map(location=[5, 20], zoom_start=4,
                       tiles="CartoDB positron")

        title_html = """
        <div style="position:fixed;top:10px;left:50%;transform:translateX(-50%);
                    z-index:1000;background:white;padding:10px 18px;
                    border-radius:8px;box-shadow:2px 2px 8px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:14px;font-weight:bold;color:#2C3E50;">
            🦟 Africa Disease Outbreak Risk Map
        </div>"""
        m.get_root().html.add_child(folium.Element(title_html))

        def risk_color(risk):
            if risk < 0.3: return "#27AE60"
            elif risk < 0.5: return "#F39C12"
            elif risk < 0.7: return "#E67E22"
            else: return "#E74C3C"

        for _, row in country_risk.iterrows():
            color = risk_color(row["outbreak_risk"])
            popup_html = f"""
            <div style="font-family:Arial;min-width:180px;">
                <h4 style="color:#2C3E50;margin:0 0 8px 0;">
                    🌍 {row['country']}
                </h4>
                <b style="color:{color}">Risk Score: {row['outbreak_risk']:.3f}</b><br>
                🦟 Malaria: {row['malaria_cases']:,.0f} total cases<br>
                💧 Cholera: {row['cholera_cases']:,.0f} total cases<br>
                🌡️ Dengue:  {row['dengue_cases']:,.0f} total cases
            </div>"""
            folium.CircleMarker(
                location=[row["latitude"], row["longitude"]],
                radius=12 + row["outbreak_risk"] * 15,
                color="white", weight=1.5,
                fill=True, fill_color=color, fill_opacity=0.85,
                tooltip=f"{row['country']} | Risk: {row['outbreak_risk']:.3f}",
                popup=folium.Popup(popup_html, max_width=220)
            ).add_to(m)

        legend_html = """
        <div style="position:fixed;bottom:30px;left:30px;z-index:1000;
                    background:white;padding:12px;border-radius:8px;
                    box-shadow:2px 2px 8px rgba(0,0,0,0.3);
                    font-family:Arial;font-size:12px;">
            <b>🦟 Outbreak Risk</b><br>
            <span style="color:#27AE60;">●</span> Low (&lt; 0.3)<br>
            <span style="color:#F39C12;">●</span> Moderate (0.3–0.5)<br>
            <span style="color:#E67E22;">●</span> High (0.5–0.7)<br>
            <span style="color:#E74C3C;">●</span> Critical (&gt; 0.7)
        </div>"""
        m.get_root().html.add_child(folium.Element(legend_html))

        if save:
            path = IMAGES_DIR / "outbreak_map.html"
            m.save(str(path))
            print(f"     Saved: outbreak_map.html")
        return m

    def plot_model_comparison(self, results: dict, save=True) -> plt.Figure:
        """Bar chart comparing all ML model metrics."""
        print("  📊 Plotting model comparison...")

        models = list(results.keys())
        rmse_v = [results[m]["rmse"] for m in models]
        mae_v  = [results[m]["mae"]  for m in models]
        r2_v   = [results[m]["r2"]   for m in models]

        fig, axes = plt.subplots(1, 3, figsize=(15, 5),
                                  constrained_layout=True)
        fig.suptitle("🤖  Model Performance Comparison",
                     fontsize=14, fontweight="bold")

        for ax, vals, label, hi in [
            (axes[0], rmse_v, "RMSE ↓", False),
            (axes[1], mae_v,  "MAE ↓",  False),
            (axes[2], r2_v,   "R² ↑",   True),
        ]:
            best  = min(vals) if not hi else max(vals)
            cols  = ["#27AE60" if v==best else "#BDC3C7" for v in vals]
            bars  = ax.bar(models, vals, color=cols, edgecolor="white")
            ax.set_title(label, fontweight="bold")
            bi = vals.index(best)
            ax.text(bi, best+max(vals)*0.03, "🏆", ha="center", fontsize=14)
            for bar, val in zip(bars, vals):
                ax.text(bar.get_x()+bar.get_width()/2,
                        bar.get_height()+max(vals)*0.01,
                        f"{val:.4f}", ha="center", fontsize=8)
            ax.tick_params(axis="x", rotation=15)

        if save:
            fig.savefig(IMAGES_DIR/"model_comparison.png",
                        bbox_inches="tight", dpi=150)
            print(f"     Saved: model_comparison.png")
        return fig

    def plotly_disease_trends(self, df: pd.DataFrame) -> go.Figure:
        """Interactive Plotly chart for Streamlit dashboard."""
        monthly = df.groupby("date").agg(
            Malaria=("malaria_cases","sum"),
            Cholera=("cholera_cases","sum"),
            Dengue=("dengue_cases","sum"),
        ).reset_index()

        fig = go.Figure()
        for disease, color in [("Malaria","#E74C3C"),
                                ("Cholera","#3498DB"),
                                ("Dengue","#F39C12")]:
            fig.add_trace(go.Scatter(
                x=monthly["date"], y=monthly[disease],
                mode="lines", name=disease,
                line=dict(color=color, width=2),
                fill="tozeroy",
                fillcolor=color.replace(")", ",0.05)").replace("rgb","rgba")
            ))
        fig.update_layout(
            title="Disease Cases Across Africa (2010–2023)",
            xaxis_title="Date", yaxis_title="Total Cases",
            hovermode="x unified", template="plotly_white",
            height=380
        )
        return fig

    def plotly_risk_by_country(self, df: pd.DataFrame) -> go.Figure:
        """Interactive bar chart of outbreak risk by country."""
        risk = df.groupby("country")["outbreak_risk"].mean().sort_values()
        colors = ["#E74C3C" if v > 0.5 else "#F39C12" if v > 0.3
                  else "#27AE60" for v in risk.values]
        fig = go.Figure(go.Bar(
            x=risk.values, y=risk.index,
            orientation="h", marker_color=colors,
            text=risk.values.round(3), textposition="outside"
        ))
        fig.update_layout(
            title="Mean Outbreak Risk Score by Country",
            xaxis_title="Risk Score (0–1)", template="plotly_white",
            height=400, xaxis_range=[0, 1]
        )
        fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                      annotation_text="High Risk Threshold")
        return fig


if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import DiseaseDataLoader
    df  = DiseaseDataLoader().load_all_data()
    viz = DiseaseVisualizer()
    viz.plot_disease_trends(df)
    viz.plot_climate_disease_correlation(df)
    viz.plot_seasonal_patterns(df)
    viz.plot_country_comparison(df)
    viz.create_outbreak_map(df)
    print("\n✅ All visualizations saved to images/")
