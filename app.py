"""
=============================================================================
app.py — Streamlit Dashboard
Climate-Disease-Africa: Outbreak Risk Prediction
=============================================================================
RUN WITH:  streamlit run app.py

AUTHOR:  Emmanuel Yaw Afram (Prestige)
EMAIL:   eyafram7@gmail.com
GITHUB:  github.com/eyafram7-data
LINKEDIN: linkedin.com/in/emmanuel-yaw-afram77
=============================================================================
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import sys
from pathlib import Path
import warnings
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent / "src"))

st.set_page_config(
    page_title="Climate-Disease Africa",
    page_icon="🦟",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
.stApp { background-color: #F0F4F8; }
[data-testid="metric-container"] {
    background: white; border-radius: 12px;
    padding: 16px; border: 1px solid #E0E7FF;
    box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.dashboard-header {
    background: linear-gradient(135deg, #1B4F72, #E74C3C);
    color: white; padding: 24px 32px; border-radius: 12px;
    margin-bottom: 24px; text-align: center;
}
.info-box {
    background: #EBF5FB; border: 1px solid #AED6F1;
    border-radius: 8px; padding: 12px 16px;
    font-size: 14px; color: #1A5276;
}
.footer { text-align:center; color:#7F8C8D;
          font-size:12px; padding:16px; margin-top:32px; }
</style>
""", unsafe_allow_html=True)


@st.cache_data(ttl=3600, show_spinner="Loading disease-climate dataset...")
def load_data():
    from data_loader import DiseaseDataLoader
    return DiseaseDataLoader().load_all_data()


@st.cache_data(show_spinner="Running preprocessing...")
def run_preprocessing(df_json):
    from preprocessing import DiseasePreprocessor
    df = pd.read_json(df_json)
    df["date"] = pd.to_datetime(df["date"])
    return DiseasePreprocessor().run_pipeline(df)


@st.cache_resource(show_spinner="Training ML models...")
def train_models(X_tr, y_tr, X_te, y_te, feat_names):
    from model import OutbreakModelTrainer
    trainer = OutbreakModelTrainer()
    results = trainer.train_all(X_tr, y_tr, X_te, y_te,
                                feature_names=feat_names)
    return trainer, results


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dashboard-header">
    <h1 style="margin:0;font-size:28px;">
        🦟 Climate-Disease Africa
    </h1>
    <p style="margin:6px 0 0;font-size:16px;opacity:0.9;">
        Predicting Malaria, Cholera & Dengue Outbreak Risk Using Climate Data & ML
    </p>
    <p style="margin:4px 0 0;font-size:12px;opacity:0.7;">
        Data: WHO · Africa CDC · ERA5 Climate | 2010–2023 | 10 African Countries
    </p>
</div>
""", unsafe_allow_html=True)

# ── Load data ──────────────────────────────────────────────────────────────────
try:
    df = load_data()
    df["date"] = pd.to_datetime(df["date"])
except Exception as e:
    st.error(f"❌ Error: {e}")
    st.info("Run `python run_pipeline.py` first to generate the dataset.")
    st.stop()

# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:8px 0 16px;">
        <span style="font-size:48px;">🦟</span><br>
        <span style="color:white;font-size:16px;font-weight:bold;">
            Climate-Disease<br>Africa
        </span>
    </div>
    <hr style="border-color:#2E86C1;margin:8px 0;">
    """, unsafe_allow_html=True)

    st.markdown("**📌 FILTERS**")

    countries = sorted(df["country"].unique())
    selected_countries = st.multiselect(
        "🌍 Countries", countries, default=countries)

    years = sorted(df["date"].dt.year.unique())
    start_year = st.slider("Start Year", min(years), max(years), min(years))
    end_year   = st.slider("End Year",   min(years), max(years), max(years))

    disease_focus = st.selectbox(
        "🦟 Disease Focus",
        ["All Diseases", "Malaria", "Cholera", "Dengue"])

    train_toggle = st.checkbox("🤖 Train ML Models", value=True)
    forecast_months = st.slider("Forecast Horizon (months)", 6, 36, 24)

    st.markdown("---")
    st.markdown("""
    <div style="color:#AED6F1;font-size:11px;text-align:center;">
        <b>Author</b><br>
        Emmanuel Yaw Afram<br>
        <a href="https://github.com/eyafram7-data"
           style="color:#5DADE2;">GitHub</a> ·
        <a href="https://www.linkedin.com/in/emmanuel-yaw-afram77"
           style="color:#5DADE2;">LinkedIn</a><br>
        eyafram7@gmail.com
    </div>
    """, unsafe_allow_html=True)

# ── Filter ────────────────────────────────────────────────────────────────────
df_f = df[
    (df["country"].isin(selected_countries)) &
    (df["date"].dt.year >= start_year) &
    (df["date"].dt.year <= end_year)
].copy()

if df_f.empty:
    st.warning("⚠️ No data for selected filters.")
    st.stop()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Overview", "🦟 Disease Trends",
    "🌡️ Climate Drivers", "🤖 ML Models", "🔮 Forecast"])


# ── TAB 1: OVERVIEW ───────────────────────────────────────────────────────────
with tab1:
    st.markdown("#### 📊 Key Statistics")
    c1,c2,c3,c4,c5 = st.columns(5)

    with c1:
        st.metric("🦟 Total Malaria",
                  f"{df_f['malaria_cases'].sum()/1e6:.1f}M cases",
                  delta=f"{df_f['malaria_cases'].mean():.0f} avg/mo")
    with c2:
        st.metric("💧 Total Cholera",
                  f"{df_f['cholera_cases'].sum()/1e3:.0f}K cases",
                  delta=f"{df_f['cholera_cases'].mean():.0f} avg/mo")
    with c3:
        st.metric("🌡️ Total Dengue",
                  f"{df_f['dengue_cases'].sum()/1e3:.0f}K cases",
                  delta=f"{df_f['dengue_cases'].mean():.0f} avg/mo")
    with c4:
        st.metric("🌡️ Avg Temperature",
                  f"{df_f['temperature_mean'].mean():.1f}°C")
    with c5:
        st.metric("☔ Avg Precipitation",
                  f"{df_f['precipitation'].mean():.0f} mm/mo")

    st.markdown("---")
    st.markdown("#### 🌍 Outbreak Risk by Country")

    risk_by_country = df_f.groupby("country")["outbreak_risk"].mean().sort_values()
    colors = ["#E74C3C" if v>0.5 else "#F39C12" if v>0.3
              else "#27AE60" for v in risk_by_country.values]
    fig = go.Figure(go.Bar(
        x=risk_by_country.values, y=risk_by_country.index,
        orientation="h", marker_color=colors,
        text=risk_by_country.values.round(3), textposition="outside"))
    fig.add_vline(x=0.5, line_dash="dash", line_color="red",
                  annotation_text="High Risk")
    fig.update_layout(height=420, template="plotly_white",
                      xaxis_title="Risk Score (0–1)",
                      title="Mean Outbreak Risk Score by Country")
    st.plotly_chart(fig, use_container_width=True)


# ── TAB 2: DISEASE TRENDS ─────────────────────────────────────────────────────
with tab2:
    st.markdown("""
    <div class="info-box">
    <b>💡 About These Diseases:</b>
    <b>Malaria</b> kills 500,000+ Africans/year · driven by rainfall & temperature.<br>
    <b>Cholera</b> spreads after flooding · contaminates water sources.<br>
    <b>Dengue</b> rising with climate change · driven by heat & stagnant water.
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    monthly = df_f.groupby("date").agg(
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
            line=dict(color=color, width=2)))
    fig.update_layout(height=380, template="plotly_white",
                      hovermode="x unified",
                      title="Monthly Disease Cases Across Selected Countries")
    st.plotly_chart(fig, use_container_width=True)

    # Seasonal patterns
    st.markdown("#### 🗓️ Seasonal Pattern by Month")
    df_f["month_name"] = df_f["date"].dt.month
    seasonal = df_f.groupby("month_name")[
        ["malaria_cases","cholera_cases","dengue_cases"]].mean().reset_index()

    fig2 = px.bar(seasonal.melt(id_vars="month_name"),
                  x="month_name", y="value", color="variable",
                  barmode="group",
                  color_discrete_map={"malaria_cases":"#E74C3C",
                                      "cholera_cases":"#3498DB",
                                      "dengue_cases":"#F39C12"},
                  labels={"month_name":"Month","value":"Avg Cases",
                          "variable":"Disease"},
                  template="plotly_white")
    fig2.update_layout(height=340)
    st.plotly_chart(fig2, use_container_width=True)


# ── TAB 3: CLIMATE DRIVERS ───────────────────────────────────────────────────
with tab3:
    st.markdown("#### 🌡️ Climate Variables Over Time")

    clim_var = st.selectbox("Select Variable", [
        "temperature_mean","precipitation","humidity",
        "flood_index","ndvi"])

    fig = px.line(df_f.groupby(["date","country"])[clim_var].mean().reset_index(),
                  x="date", y=clim_var, color="country",
                  template="plotly_white",
                  title=f"{clim_var.replace('_',' ').title()} by Country")
    fig.update_layout(height=370)
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("#### 🔥 Climate–Disease Correlations")
    cols = ["outbreak_risk","malaria_cases","cholera_cases","dengue_cases",
            "temperature_mean","precipitation","humidity","flood_index","ndvi"]
    cols = [c for c in cols if c in df_f.columns]
    corr = df_f[cols].corr()

    fig_h = px.imshow(corr, color_continuous_scale="RdBu_r",
                       zmin=-1, zmax=1, aspect="auto",
                       template="plotly_white",
                       title="Pearson Correlation Matrix")
    fig_h.update_layout(height=450)
    st.plotly_chart(fig_h, use_container_width=True)


# ── TAB 4: ML MODELS ─────────────────────────────────────────────────────────
with tab4:
    if not train_toggle:
        st.info("Enable 'Train ML Models' in the sidebar.")
    else:
        st.markdown("""
        <div class="info-box">
        <b>🤖 Model Training:</b> Predicting outbreak risk score (0–1) from
        climate variables. Temporal split used (no data leakage).
        </div>
        """, unsafe_allow_html=True)

        with st.spinner("Preprocessing..."):
            result = run_preprocessing(df_f.to_json())

        st.success(f"✅ {len(result['feature_names'])} features · "
                   f"{len(result['X_train']):,} training samples")

        with st.spinner("Training models (may take ~60s)..."):
            trainer, model_results = train_models(
                result["X_train_scaled"], np.array(result["y_train"]),
                result["X_test_scaled"],  np.array(result["y_test"]),
                feat_names=result["feature_names"])

        # Leaderboard
        rows = []
        for name, res in model_results.items():
            rows.append({"Model":name,
                         "RMSE":round(res["rmse"],5),
                         "MAE": round(res["mae"],5),
                         "R²":  round(res["r2"],5),
                         "Best":"★" if name==trainer.best_model_name else ""})
        st.dataframe(pd.DataFrame(rows).sort_values("R²",ascending=False),
                     hide_index=True, use_container_width=True)

        # Metric bars
        c1,c2,c3 = st.columns(3)
        for col, metric, hi in [(c1,"RMSE",False),(c2,"MAE",False),(c3,"R²",True)]:
            with col:
                vals   = {n:model_results[n][metric.lower()] for n in model_results}
                best_v = min(vals.values()) if not hi else max(vals.values())
                colors = ["#27AE60" if v==best_v else "#BDC3C7" for v in vals.values()]
                fig_m  = go.Figure(go.Bar(
                    x=list(vals.keys()), y=list(vals.values()),
                    marker_color=colors,
                    text=[f"{v:.4f}" for v in vals.values()],
                    textposition="outside"))
                fig_m.update_layout(title=metric, height=300,
                                    template="plotly_white",
                                    margin=dict(t=40,b=40))
                st.plotly_chart(fig_m, use_container_width=True)

        # Feature importance
        if "XGBoost" in model_results and \
           "feature_importances" in model_results["XGBoost"]:
            st.markdown("#### 🎯 XGBoost Feature Importance")
            fi  = model_results["XGBoost"]["feature_importances"]
            fn  = model_results["XGBoost"]["feature_names"]
            fi_df = pd.DataFrame({"feature":fn,"importance":fi})\
                      .sort_values("importance",ascending=False).head(15)
            fig_fi = px.bar(fi_df, x="importance", y="feature",
                            orientation="h", color="importance",
                            color_continuous_scale="Greens",
                            template="plotly_white")
            fig_fi.update_layout(height=450,
                                  yaxis=dict(autorange="reversed"),
                                  coloraxis_showscale=False)
            st.plotly_chart(fig_fi, use_container_width=True)


# ── TAB 5: FORECAST ───────────────────────────────────────────────────────────
with tab5:
    st.markdown("""
    <div class="info-box">
    <b>🔮 Scenarios:</b><br>
    🟢 <b>Optimistic</b>: Better sanitation, stable climate<br>
    🟡 <b>Baseline</b>: Current trends continue<br>
    🔴 <b>Pessimistic</b>: Rapid warming, increased flooding
    </div>
    """, unsafe_allow_html=True)
    st.markdown("")

    # Simple forecast using historical trend
    last_date  = df_f["date"].max()
    future_d   = pd.date_range(last_date + pd.DateOffset(months=1),
                                periods=forecast_months, freq="MS")
    hist_risk  = df_f.groupby("date")["outbreak_risk"].mean()
    base_risk  = hist_risk.iloc[-6:].mean()
    recent_tr  = np.polyfit(range(24), hist_risk.iloc[-24:].values, 1)[0]

    fc_data = []
    np.random.seed(42)
    for scen, slope, color in [
        ("🟢 Optimistic",  -0.001, "#27AE60"),
        ("🟡 Baseline",    recent_tr, "#F39C12"),
        ("🔴 Pessimistic", +0.002, "#E74C3C"),
    ]:
        for i, d in enumerate(future_d):
            risk = np.clip(
                base_risk + slope*(i+1)
                + 0.02*np.sin(2*np.pi*d.month/12)
                + np.random.normal(0, 0.005), 0, 1)
            fc_data.append({"date":d,"scenario":scen,
                             "outbreak_risk":risk,"color":color})

    fc_df = pd.DataFrame(fc_data)

    fig = go.Figure()
    hist_recent = hist_risk[hist_risk.index >= "2021-01-01"].reset_index()
    fig.add_trace(go.Scatter(
        x=hist_recent["date"], y=hist_recent["outbreak_risk"],
        mode="lines", line=dict(color="#2C3E50",width=3),
        name="Historical Risk"))
    fig.add_vline(x=str(last_date), line_dash="dash",
                  line_color="gray", annotation_text="Forecast Start")

    for scen in fc_df["scenario"].unique():
        s = fc_df[fc_df["scenario"]==scen]
        color = s["color"].iloc[0]
        fig.add_trace(go.Scatter(
            x=s["date"], y=s["outbreak_risk"],
            mode="lines", name=scen,
            line=dict(color=color, width=2.5)))

    fig.add_hline(y=0.5, line_dash="dot", line_color="red",
                  annotation_text="High Risk Threshold")
    fig.update_layout(height=440, template="plotly_white",
                      hovermode="x unified",
                      yaxis_title="Outbreak Risk Score (0–1)",
                      title=f"Outbreak Risk Forecast — {forecast_months} Months")
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("#### 📋 Scenario Summary")
    rows = []
    for scen in fc_df["scenario"].unique():
        s = fc_df[fc_df["scenario"]==scen]
        rows.append({
            "Scenario":scen,
            "Start Risk":f"{s['outbreak_risk'].iloc[0]:.4f}",
            f"End Risk ({forecast_months}mo)":f"{s['outbreak_risk'].iloc[-1]:.4f}",
            "Net Change":f"{s['outbreak_risk'].iloc[-1]-s['outbreak_risk'].iloc[0]:+.4f}",
            "Peak Risk":f"{s['outbreak_risk'].max():.4f}",
        })
    st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)


# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    🦟 Climate-Disease Africa · Built by
    <a href="https://github.com/eyafram7-data">Emmanuel Yaw Afram</a> ·
    <a href="https://www.linkedin.com/in/emmanuel-yaw-afram77">LinkedIn</a> ·
    eyafram7@gmail.com · MIT License · 2024
</div>
""", unsafe_allow_html=True)
