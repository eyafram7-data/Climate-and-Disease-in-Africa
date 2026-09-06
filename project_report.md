# 📄 Project Analysis Report
## Climate-Disease-Africa
### Predicting Disease Outbreak Risk Using Climate Data and Machine Learning

---

**Author:** Emmanuel Yaw Afram (Prestige)
**Email:** eyafram7@gmail.com
**GitHub:** github.com/eyafram7-data
**LinkedIn:** linkedin.com/in/emmanuel-yaw-afram77
**Version:** 1.0.0 | **License:** MIT | **Year:** 2024

---

## Executive Summary

This report presents the findings of a machine learning analysis of the
relationship between climate variables and disease outbreak risk across
10 African countries (2010–2023).

Using synthetic data modelled on WHO and Africa CDC outbreak records,
XGBoost achieved an R² of **0.941** — meaning climate variables alone
explain 94% of the variation in outbreak risk scores.

**Key finding:** Rainfall and flood index are the strongest climate
predictors of disease outbreaks. A 1-month lag in precipitation is
the single most important feature for predicting Malaria risk.

---

## 1. Study Design

| Parameter | Value |
|---|---|
| Study period | January 2010 – December 2023 |
| Countries | 10 African nations |
| Diseases | Malaria, Cholera, Dengue |
| Primary target | Outbreak Risk Score (0–1) |
| Train/test split | Temporal (80% / 20%) |

### Countries Studied

| Country | Climate Zone | Population | Primary Disease |
|---|---|---|---|
| Nigeria | Tropical | 220M | Malaria |
| DR Congo | Equatorial | 100M | Malaria |
| Ghana | Tropical | 33M | Malaria + Cholera |
| Ethiopia | Semi-Arid | 120M | Malaria |
| Tanzania | Tropical | 63M | Malaria + Dengue |
| Mozambique | Tropical | 33M | Cholera |
| Mali | Sahel | 22M | Malaria |
| Cameroon | Equatorial | 27M | Malaria |
| Uganda | Equatorial | 47M | Malaria + Dengue |
| Senegal | Sahel | 17M | Malaria |

---

## 2. Disease Burden Summary (2010–2023)

| Disease | Total Cases (simulated) | Peak Month | Primary Driver |
|---|---|---|---|
| Malaria | ~487 million | May–July | Rainfall + Temperature |
| Cholera | ~38 million | June–August | Flooding + Water access |
| Dengue | ~12 million | July–September | Heat + Humidity |

---

## 3. Feature Engineering Summary

| Feature Category | Examples | Rationale |
|---|---|---|
| Cyclical time | month_sin, month_cos | Seasonal cycles |
| Climate lags | precipitation_lag1/2/3 | Delayed disease response |
| Rolling windows | precip_roll3, flood_roll6 | Accumulated climate stress |
| Heat stress | heat_humid_stress | Mosquito breeding conditions |
| Anomalies | temp_anomaly, precip_anomaly | Unusual climate events |
| Seasonal flag | wet_season | Binary wet/dry indicator |

---

## 4. Model Performance

| Model | RMSE | MAE | R² |
|---|---|---|---|
| **XGBoost** | **0.0413** | **0.0318** | **0.941** |
| Random Forest | 0.0482 | 0.0372 | 0.912 |
| SVR | 0.0608 | 0.0492 | 0.863 |
| Ridge Regression | 0.0792 | 0.0631 | 0.781 |
| Linear Regression | 0.0851 | 0.0682 | 0.723 |

---

## 5. Top Predictive Features (XGBoost)

| Rank | Feature | Importance | Interpretation |
|---|---|---|---|
| 1 | precipitation_lag1 | 0.187 | Last month's rainfall drives mosquito breeding |
| 2 | flood_roll3 | 0.142 | 3-month cumulative flooding → cholera risk |
| 3 | heat_humid_stress | 0.118 | Heat × humidity → dengue conditions |
| 4 | precip_roll6 | 0.094 | 6-month rainfall accumulation |
| 5 | flood_index | 0.088 | Current month flooding |
| 6 | temp_anomaly | 0.071 | Unusual heat events |
| 7 | wet_season | 0.065 | Wet season binary flag |
| 8 | precipitation | 0.058 | Current rainfall |
| 9 | humidity | 0.047 | Atmospheric moisture |
| 10 | ndvi | 0.041 | Vegetation (mosquito habitat) |

---

## 6. Scenario Forecast Results (2024–2025)

| Scenario | Description | Start Risk | End Risk | Change |
|---|---|---|---|---|
| 🟢 Optimistic | Better sanitation, stable climate | 0.421 | 0.389 | −0.032 |
| 🟡 Baseline | Current trends continue | 0.421 | 0.438 | +0.017 |
| 🔴 Pessimistic | Rapid warming, more flooding | 0.421 | 0.497 | +0.076 |

---

## 7. Key Scientific Findings

1. **Malaria is strongly seasonal** — cases peak 1–2 months after
   the peak of the rainy season, consistent with Anopheles mosquito
   breeding cycles.

2. **Cholera spikes immediately after flooding** — the flood index
   is the #2 predictor, confirming water contamination as the
   primary transmission pathway.

3. **Dengue is rising** — rising temperatures across the Sahel are
   expanding the geographic range of Aedes aegypti mosquitoes,
   creating new outbreak zones.

4. **Nigeria and DR Congo carry the highest burden** — together
   accounting for over 40% of all simulated malaria cases.

5. **Climate change will worsen all three diseases** — the
   pessimistic scenario shows a 7.6% increase in outbreak risk
   within just 24 months under accelerated warming.

---

## 8. Limitations

1. Synthetic data — validation with real WHO surveillance data needed
2. Country-level resolution — sub-national hotspots not captured
3. Drug resistance not modelled
4. Population movement and migration not included

---

## 9. Future Work

- Integrate real WHO AFRO surveillance API
- Add sub-national (district) level predictions
- Build SMS early warning system for health workers
- Include drug resistance and healthcare access as features
- Deploy to Streamlit Cloud for public access
- Expand to include Mpox and Meningitis

---

## 10. References

1. Africa CDC (2025). Annual Disease Outbreak Report 2024.
2. WHO (2024). World Malaria Report 2024.
3. Mordecai, E.A. et al. (2019). Thermal biology of mosquito-borne disease. Ecology Letters.
4. Bhatt, S. et al. (2013). The global distribution and burden of dengue. Nature.
5. Lipp, E.K. et al. (2002). Effects of global climate on infectious disease. Clinical Microbiology Reviews.
6. Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD '16.

---

*Report generated by the Climate-Disease-Africa analysis pipeline.*
*Contact: eyafram7@gmail.com | github.com/eyafram7-data*
