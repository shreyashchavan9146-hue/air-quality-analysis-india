
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Air Quality India",
    page_icon="🌍",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_PATH = BASE_DIR / "data" / "raw" / "data.csv.csv"
AQI_PATH = BASE_DIR / "data" / "processed" / "air_quality_with_aqi.csv"
MODEL_DIR = BASE_DIR / "outputs" / "models"

# ---------------- THEME ----------------
st.markdown("""
<style>
.stApp {
    background: #0B1220;
    color: #E5E7EB;
}
[data-testid="stSidebar"] {
    background: #111C2E;
}
h1, h2, h3 {
    color: #F1F5F9 !important;
}
p, label, span {
    color: #CBD5E1;
}
div[data-testid="stMetric"] {
    background: #16243A;
    border: 1px solid #263B55;
    padding: 20px;
    border-radius: 14px;
}
div[data-testid="stMetricValue"] {
    color: #2DD4BF;
}
.stButton > button {
    background: #0D9488;
    color: white;
    border-radius: 10px;
    border: none;
}
.block-container {
    padding-top: 2rem;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
@st.cache_data
def load_csv(path):
    if path.exists():
        return pd.read_csv(path, encoding="cp1252")
    return pd.DataFrame()

raw_df = load_csv(RAW_PATH)
aqi_df = load_csv(AQI_PATH)

df = aqi_df if not aqi_df.empty else raw_df

# ---------------- HELPERS ----------------
def find_column(data, names):
    normalized = {
        str(c).strip().lower().replace(" ", "").replace("_", ""): c
        for c in data.columns
    }
    for name in names:
        key = name.lower().replace(" ", "").replace("_", "")
        if key in normalized:
            return normalized[key]
    return None

def numeric_series(data, col):
    return pd.to_numeric(data[col], errors="coerce").dropna()

so2_col = find_column(df, ["so2", "sulphur dioxide"])
no2_col = find_column(df, ["no2", "nitrogen dioxide"])
spm_col = find_column(df, ["spm"])
rspm_col = find_column(df, ["rspm", "pm10"])
aqi_col = find_column(df, ["aqi", "estimated_aqi"])
state_col = find_column(df, ["state", "state name", "st_name"])

# ---------------- SIDEBAR ----------------
st.sidebar.title("🌍 AIR QUALITY INDIA")
st.sidebar.caption("Environmental Analytics Platform")
st.sidebar.markdown("---")
st.sidebar.write("📊 Data Analytics")
st.sidebar.write("🧪 Pollutant Monitoring")
st.sidebar.write("🗺️ State Comparison")
st.sidebar.write("🤖 ML Prediction")
st.sidebar.markdown("---")
st.sidebar.info("India Air Quality Analysis Project")

# ---------------- HEADER ----------------
st.markdown("# 🌍 AIR QUALITY ANALYTICS")
st.markdown("### Monitoring India's environmental health")
st.caption("Explore pollutant levels, estimated AQI patterns and machine learning predictions.")
st.markdown("---")

# ---------------- OVERVIEW ----------------
st.header("📊 01 | National Overview")

k1, k2, k3, k4 = st.columns(4)

def mean_metric(col):
    if col is None or df.empty:
        return "N/A"
    values = numeric_series(df, col)
    return f"{values.mean():.2f}" if not values.empty else "N/A"

with k1:
    st.metric("Estimated Average AQI", mean_metric(aqi_col))
with k2:
    st.metric("Average SO₂", mean_metric(so2_col))
with k3:
    st.metric("Average NO₂", mean_metric(no2_col))
with k4:
    st.metric("Average SPM", mean_metric(spm_col))

st.markdown("---")

# ---------------- POLLUTANT ANALYSIS ----------------
st.header("🧪 02 | Pollutant Analysis")

pollutants = {
    "SO2": so2_col,
    "NO2": no2_col,
    "SPM": spm_col,
    "RSPM": rspm_col
}
available = {name: col for name, col in pollutants.items() if col}

if available:
    selected = st.selectbox("Choose pollutant", list(available.keys()))
    values = numeric_series(df, available[selected])
    left, right = st.columns(2)

    with left:
        fig = px.histogram(
            x=values, nbins=30,
            title=f"{selected} Distribution",
            template="plotly_dark",
            color_discrete_sequence=["#2DD4BF"]
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        fig = px.box(
            x=values,
            title=f"{selected} Outlier Analysis",
            template="plotly_dark",
            color_discrete_sequence=["#38BDF8"]
        )
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Pollutant Statistics")
    st.dataframe(values.describe().to_frame("Value"),
                 use_container_width=True)
else:
    st.warning("No supported pollutant columns found.")

st.markdown("---")

# ---------------- STATE ANALYSIS ----------------
st.header("🗺️ 03 | State-wise Air Quality")

if state_col and aqi_col:
    state_data = df[[state_col, aqi_col]].copy()
    state_data[aqi_col] = pd.to_numeric(
        state_data[aqi_col], errors="coerce"
    )
    state_data = state_data.dropna()

    summary = (
        state_data.groupby(state_col)[aqi_col]
        .mean()
        .reset_index()
        .sort_values(aqi_col, ascending=False)
    )

    left, right = st.columns(2)

    with left:
        st.subheader("Highest Average Estimated AQI")
        st.dataframe(summary.head(10),
                     use_container_width=True, hide_index=True)

    with right:
        st.subheader("Lowest Average Estimated AQI")
        st.dataframe(
            summary.tail(10).sort_values(aqi_col),
            use_container_width=True, hide_index=True
        )

    fig = px.bar(
        summary,
        x=state_col,
        y=aqi_col,
        color=aqi_col,
        title="Average Estimated AQI by State",
        template="plotly_dark",
        color_continuous_scale="Teal"
    )
    fig.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig, use_container_width=True)
else:
    st.warning(
        "State or AQI column is unavailable. "
        "Check the processed AQI dataset."
    )

st.markdown("---")

# ---------------- ADVANCED AQI ANALYTICS ----------------
st.markdown("---")
st.header("📈 03A | Advanced AQI Analytics")

if not df.empty:
    if aqi_col:
        aqi_values = numeric_series(df, aqi_col)

        if not aqi_values.empty:
            fig = px.histogram(
                x=aqi_values,
                nbins=40,
                title="Estimated AQI Distribution",
                labels={"x": "Estimated AQI", "y": "Number of Records"},
                template="plotly_dark",
                color_discrete_sequence=["#2DD4BF"]
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No numeric AQI values available.")

    numeric_cols = [
        col for col in [so2_col, no2_col, spm_col, rspm_col, aqi_col]
        if col is not None
    ]

    numeric_data = df[numeric_cols].apply(
        pd.to_numeric, errors="coerce"
    ).dropna()

    if len(numeric_cols) >= 2 and not numeric_data.empty:
        st.subheader("Pollutant Correlation Heatmap")

        correlation = numeric_data.corr()

        fig = px.imshow(
            correlation,
            text_auto=".2f",
            color_continuous_scale="Teal",
            title="Correlation Between Air Quality Variables",
            template="plotly_dark",
            aspect="auto"
        )

        st.plotly_chart(fig, use_container_width=True)
else:
    st.warning("Dataset is not available.")


# ---------------- ML PREDICTION ----------------
st.header("🤖 04 | Machine Learning Prediction")

model_files = {
    "Linear Regression": "linear_regression_aqi_model.pkl",
    "Ridge Regression": "ridge_aqi_model.pkl",
    "Lasso Regression": "lasso_aqi_model.pkl"
}

choice = st.selectbox("Select prediction model", list(model_files))

c1, c2, c3 = st.columns(3)
with c1:
    so2 = st.number_input("SO2 concentration", min_value=0.0)
with c2:
    no2 = st.number_input("NO2 concentration", min_value=0.0)
with c3:
    spm = st.number_input("SPM concentration", min_value=0.0)

if st.button("Predict Estimated AQI", use_container_width=True):
    path = MODEL_DIR / model_files[choice]

    if path.exists():
        try:
            model = joblib.load(path)
            input_df = pd.DataFrame(
                [[so2, no2, spm]],
                columns=["so2", "no2", "spm"]
            )
            prediction = model.predict(input_df)[0]
            st.success(f"Predicted Estimated AQI: {prediction:.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")
    else:
        st.error(f"Model not found: {path}")

st.caption(
    "Note: This is a project-specific estimated AQI prediction, "
    "not an official CPCB AQI reading."
)

st.markdown("---")

# ---------------- DATASET ----------------
st.header("📁 05 | Dataset Explorer")

if not df.empty:
    st.success(f"Dataset loaded: {len(df):,} records")
    st.write("Dataset columns:", df.columns.tolist())
    st.dataframe(df.head(20), use_container_width=True)
else:
    st.warning("Dataset could not be loaded.")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown(
    "<center>🌍 Air Quality Analysis of India | "
    "Python • Pandas • Plotly • Streamlit • Machine Learning</center>",
    unsafe_allow_html=True
)
