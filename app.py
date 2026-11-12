"""
app.py  --  Streamlit interface.  Run with:   streamlit run app.py

Every query typed (or uploaded) here is sent through the SAME trained pipeline that is
described in the paper:  encoding -> scaling -> feature selection -> boosting model,
loaded from carbon_model.pickle.
"""
import os
import pandas as pd
import streamlit as st

import model as M
import preprocessing as prep

st.set_page_config(page_title="AI Carbon Emission Predictor", page_icon="🌍", layout="wide")


@st.cache_resource(show_spinner="Loading trained model...")
def get_bundle():
    try:
        return M.load(M.PICKLE_FILE)
    except Exception:                       # missing / incompatible pickle -> retrain once
        b = M.train(verbose=False)
        try:
            M.save(b)
        except Exception:
            pass
        return b


B = get_bundle()
D = B["defaults"]
CATS = B["encoder"].categories
TOTAL, RENEW, NONRENEW = M.TOTAL, M.RENEW, M.NONRENEW

st.title("🌍 AI-Based Carbon Emission Prediction and Reduction")
st.caption(
    f"Model: **{B['best_algorithm']}** on the **{B['best_feature_set']}** feature set  |  "
    f"features used by the model: {', '.join(f.replace('_', ' ') for f in B['features'])}"
)

tab_pred, tab_whatif, tab_batch, tab_info = st.tabs(
    ["🔮 Predict", "♻️ What-if reduction", "📄 Batch (CSV)", "📊 Model details"]
)


def query_form(key: str) -> dict:
    """Shared input form -> raw query dict."""
    c1, c2 = st.columns(2)
    total = c1.number_input("Total energy consumption (kWh)", 50000.0, 200000.0, D[TOTAL], 1000.0, key=key + "t")
    renew = c2.number_input("Renewable energy consumption (kWh)", 0.0, float(total), min(D[RENEW], float(total)), 1000.0, key=key + "r")
    st.write(f"Non-renewable energy = **{total - renew:,.0f} kWh** (total − renewable)")
    q = {TOTAL: total, RENEW: renew, NONRENEW: total - renew}
    with st.expander("Other operating details (optional - defaults are the dataset medians)"):
        a, b = st.columns(2)
        q["Production_Output_Units"] = a.number_input("Production output (units)", 0.0, 20000.0, D["Production_Output_Units"], key=key + "1")
        q["Supply_Chain_Transport_km"] = b.number_input("Supply-chain transport (km)", 0.0, 10000.0, D["Supply_Chain_Transport_km"], key=key + "2")
        q["Raw_Material_Usage_kg"] = a.number_input("Raw material usage (kg)", 0.0, 200000.0, D["Raw_Material_Usage_kg"], key=key + "3")
        q["Process_Efficiency_Percent"] = b.number_input("Process efficiency (%)", 0.0, 100.0, D["Process_Efficiency_Percent"], key=key + "4")
        q["Employment_Count"] = a.number_input("Employment count", 0.0, 20000.0, D["Employment_Count"], key=key + "5")
        q["Public_Acceptance_Index"] = b.number_input("Public acceptance index", 0.0, 1.0, D["Public_Acceptance_Index"], key=key + "6")
        q["Strategy_Implementation_Cost_USD"] = a.number_input("Strategy implementation cost (USD)", 0.0, 2000000.0, D["Strategy_Implementation_Cost_USD"], key=key + "7")
        q["Expected_Carbon_Reduction_Percent"] = b.number_input("Expected carbon reduction (%)", 0.0, 100.0, D["Expected_Carbon_Reduction_Percent"], key=key + "8")
        q["Social_Impact_Score"] = a.number_input("Social impact score", 0.0, 1.0, D["Social_Impact_Score"], key=key + "9")
        for i, col in enumerate(prep.CATEGORICAL_FEATURES):
            q[col] = (a if i % 2 == 0 else b).selectbox(col.replace("_", " "), CATS[col], index=CATS[col].index(D[col]), key=key + col)
    q["Expected_Renewable_Share_Percent"] = 100 * renew / total
    return q


# ------------------------------------------------------------------ Predict
with tab_pred:
    st.subheader("Ask the model: how much CO₂e will this company-day emit?")
    q = query_form("p")
    if st.button("Predict emission", type="primary"):
        y = float(M.predict_emission(B, q)[0])
        level = "HIGH" if y > B["high_threshold"] else "LOW"
        m1, m2, m3 = st.columns(3)
        m1.metric("Predicted emission", f"{y:.2f} tCO₂e")
        m2.metric("Emission group", level, help=f"High = above the median of {B['high_threshold']:.2f} tCO₂e")
        m3.metric("Renewable share", f"{q['Expected_Renewable_Share_Percent']:.1f} %")
        if level == "HIGH":
            st.warning("High-emission day. Try the **What-if reduction** tab to see the effect of switching to renewable energy.")
        else:
            st.success("Low-emission day (below the dataset median).")

# ------------------------------------------------------------------ What-if
with tab_whatif:
    st.subheader("What if part of the non-renewable energy is replaced by renewable energy?")
    qw = query_form("w")
    pct = st.select_slider("Share of non-renewable energy moved to renewable", [10, 20, 30, 40], value=20, format_func=lambda v: f"{v}%")
    if st.button("Estimate reduction", type="primary"):
        before, after, cut = M.what_if(B, qw, pct / 100)
        a, b, c = st.columns(3)
        a.metric("Before", f"{before:.2f} tCO₂e")
        b.metric("After", f"{after:.2f} tCO₂e", f"-{before - after:.2f}", delta_color="inverse")
        c.metric("Reduction", f"{cut:.2f} %")
        st.bar_chart(pd.DataFrame({"tCO₂e": [before, after]}, index=["Before", "After"]))

# ------------------------------------------------------------------ Batch
with tab_batch:
    st.subheader("Upload a CSV of many queries")
    st.write(
        "Required columns: `Total_Energy_Consumption_kWh` and either `Renewable_Energy_Consumption_kWh` "
        "or `NonRenewable_Energy_Consumption_kWh`. Any other column of the dataset is optional "
        "(missing ones are filled with training-set defaults)."
    )
    up = st.file_uploader("CSV file", type="csv")
    if up is not None:
        try:
            df = pd.read_csv(up)
            if TOTAL not in df or (RENEW not in df and NONRENEW not in df):
                raise ValueError("Need the total energy column and one of renewable / non-renewable energy.")
            if RENEW not in df:
                df[RENEW] = df[TOTAL] - df[NONRENEW]
            if NONRENEW not in df:
                df[NONRENEW] = df[TOTAL] - df[RENEW]
            if "Expected_Renewable_Share_Percent" not in df:
                df["Expected_Renewable_Share_Percent"] = 100 * df[RENEW] / df[TOTAL]
            for c, v in D.items():
                if c not in df:
                    df[c] = v
            out = df.copy()
            out["Predicted_Emission_tCO2e"] = M.predict_emission(B, df)
            out["Group"] = out["Predicted_Emission_tCO2e"].apply(lambda v: "High" if v > B["high_threshold"] else "Low")
            st.dataframe(out, use_container_width=True)
            st.download_button("Download predictions", out.to_csv(index=False).encode(), "predictions.csv", "text/csv")
        except Exception as e:
            st.error(f"Could not process the file: {e}")

# ------------------------------------------------------------------ Info
with tab_info:
    st.subheader("Selected features (paper Table I)")
    st.table(pd.DataFrame({"Selection": list(B["feature_sets"]), "Features kept": [", ".join(v) for v in B["feature_sets"].values()],
                           "n": [len(v) for v in B["feature_sets"].values()]}))
    st.subheader("All 15 models (test-set metrics; CV R² used for selection)")
    res = B["results"].round(2).rename(columns={"R2": "R² (%)", "AUC": "AUC (%)", "CV_R2": "CV R² (%)"})
    st.dataframe(res, use_container_width=True, hide_index=True)
    st.info(f"Chosen model: **{B['best_algorithm']}** with **{B['best_feature_set']}** (highest cross-validated R² on the training data).")
    st.caption("Carbon_Tax_USD and Energy_Cost_USD are excluded from the inputs to avoid target leakage (paper Sec. 3.2).")
