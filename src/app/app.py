import math
import os

import mlflow
import pandas as pd
import streamlit as st
from databricks import sql
from databricks.sdk.core import Config

st.set_page_config(page_title="Mumbai House Price Predictor", layout="wide")

CATALOG = os.environ.get("MODEL_CATALOG", "workspace")
SCHEMA = os.environ.get("MODEL_SCHEMA", "harshsinghv")
MODEL_NAME = os.environ.get("MODEL_NAME", "house_price_model")
FULL_MODEL_NAME = f"{CATALOG}.{SCHEMA}.{MODEL_NAME}"
WAREHOUSE_ID = os.environ.get("DATABRICKS_WAREHOUSE_ID")

# Fort, Mumbai — must match the reference point used in the gold layer's feature
# engineering (src/tasks/gold.py) so served predictions match training features.
CITY_CENTER_LAT, CITY_CENTER_LON = 18.9750, 72.8258
FURNISHED_MAP = {"Unfurnished": 0, "Semi-Furnished": 1, "Furnished": 2}
PROPERTY_TYPES = ["Apartment", "Villa", "Independent House", "Independent Floor", "Studio Apartment"]
FEATURE_COLUMNS = [
    "area",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "total_floors",
    "age",
    "bed_bath_ratio",
    "furnished_encoded",
    "locality_listing_count",
    "distance_from_center_km",
    "latitude",
    "longitude",
    "property_type",
]
RAW_INPUT_COLUMNS = [
    "area",
    "bedroom_num",
    "bathroom_num",
    "balcony_num",
    "total_floors",
    "age",
    "property_type",
    "furnished",
    "locality",
    "latitude",
    "longitude",
]


@st.cache_resource
def load_model():
    mlflow.set_registry_uri("databricks-uc")
    return mlflow.pyfunc.load_model(f"models:/{FULL_MODEL_NAME}@champion")


@st.cache_data(ttl=3600)
def load_locality_lookup() -> pd.DataFrame:
    cfg = Config()
    conn = sql.connect(
        server_hostname=cfg.host,
        http_path=f"/sql/1.0/warehouses/{WAREHOUSE_ID}",
        credentials_provider=lambda: cfg.authenticate,
    )
    with conn.cursor() as cursor:
        cursor.execute(
            f"""
            SELECT locality, COUNT(*) AS locality_listing_count,
                   AVG(latitude) AS avg_latitude, AVG(longitude) AS avg_longitude
            FROM {CATALOG}.{SCHEMA}.silver_house_pricing
            GROUP BY locality
            ORDER BY locality
            """
        )
        columns = [c[0] for c in cursor.description]
        rows = cursor.fetchall()
    return pd.DataFrame(rows, columns=columns)


def haversine_km(lat: float, lon: float) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, [CITY_CENTER_LAT, CITY_CENTER_LON, lat, lon])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def engineer_features(raw: pd.DataFrame, locality_lookup: pd.DataFrame) -> pd.DataFrame:
    """Mirrors the feature engineering in src/tasks/gold.py so predictions match training."""
    df = raw.merge(
        locality_lookup[["locality", "locality_listing_count"]], on="locality", how="left"
    )
    median_count = int(locality_lookup["locality_listing_count"].median())
    df["locality_listing_count"] = df["locality_listing_count"].fillna(median_count).astype(int)
    df["furnished_encoded"] = df["furnished"].map(FURNISHED_MAP).fillna(0).astype(int)
    df["bed_bath_ratio"] = (df["bedroom_num"] / df["bathroom_num"].clip(lower=1)).round(2)
    df["distance_from_center_km"] = df.apply(
        lambda r: round(haversine_km(float(r["latitude"]), float(r["longitude"])), 2), axis=1
    )
    return df[FEATURE_COLUMNS]


def read_uploaded_file(uploaded) -> pd.DataFrame:
    name = uploaded.name.lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith((".xlsx", ".xls")):
        return pd.read_excel(uploaded)
    if name.endswith(".json"):
        return pd.read_json(uploaded)
    raise ValueError(f"Unsupported file type: {uploaded.name}")


st.title("Mumbai House Price Predictor")
st.caption(
    f"Backed by `{FULL_MODEL_NAME}@champion` — an XGBoost model trained on 50K+ cleaned "
    f"Mumbai listings (test R² ≈ 0.86)."
)

model = load_model()
locality_lookup = load_locality_lookup()
locality_options = sorted(locality_lookup["locality"].unique().tolist())

tab_single, tab_batch = st.tabs(["Single prediction", "Batch prediction (CSV / Excel / JSON)"])

with tab_single:
    with st.form("single_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            area = st.number_input("Area (sqft)", min_value=100, max_value=20000, value=800)
            bedroom_num = st.number_input("Bedrooms", min_value=0, max_value=10, value=2)
            bathroom_num = st.number_input("Bathrooms", min_value=0, max_value=10, value=2)
        with col2:
            balcony_num = st.number_input("Balconies", min_value=0, max_value=10, value=1)
            total_floors = st.number_input("Total floors in building", min_value=1, max_value=100, value=10)
            age = st.number_input("Age of property (years)", min_value=0, max_value=100, value=5)
        with col3:
            property_type = st.selectbox("Property type", PROPERTY_TYPES)
            furnished = st.selectbox("Furnished", list(FURNISHED_MAP.keys()))
            locality = st.selectbox("Locality", locality_options)

        locality_row = locality_lookup.loc[locality_lookup["locality"] == locality].iloc[0]
        col4, col5 = st.columns(2)
        with col4:
            latitude = st.number_input(
                "Latitude", value=float(locality_row["avg_latitude"]), format="%.6f"
            )
        with col5:
            longitude = st.number_input(
                "Longitude", value=float(locality_row["avg_longitude"]), format="%.6f"
            )

        submitted = st.form_submit_button("Predict price")

    if submitted:
        raw_input = pd.DataFrame(
            [
                {
                    "area": area,
                    "bedroom_num": bedroom_num,
                    "bathroom_num": bathroom_num,
                    "balcony_num": balcony_num,
                    "total_floors": total_floors,
                    "age": age,
                    "property_type": property_type,
                    "furnished": furnished,
                    "locality": locality,
                    "latitude": latitude,
                    "longitude": longitude,
                }
            ]
        )
        features = engineer_features(raw_input, locality_lookup)
        prediction = float(model.predict(features)[0])
        st.success(f"### Predicted price: ₹{prediction:,.0f}")
        st.caption(f"≈ ₹{prediction / area:,.0f} per sqft")

with tab_batch:
    st.write(
        "Upload a CSV, Excel, or JSON file with columns: "
        f"`{', '.join(RAW_INPUT_COLUMNS)}`"
    )
    uploaded = st.file_uploader("Upload file", type=["csv", "xlsx", "xls", "json"])

    if uploaded is not None:
        try:
            batch_df = read_uploaded_file(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {e}")
            batch_df = None

        if batch_df is not None:
            missing = set(RAW_INPUT_COLUMNS) - set(batch_df.columns)
            if missing:
                st.error(f"Missing required columns: {sorted(missing)}")
            else:
                features = engineer_features(batch_df.copy(), locality_lookup)
                predictions = model.predict(features)
                result_df = batch_df.copy()
                result_df["predicted_price"] = predictions
                st.dataframe(result_df, use_container_width=True)
                st.download_button(
                    "Download predictions as CSV",
                    result_df.to_csv(index=False).encode("utf-8"),
                    file_name="predictions.csv",
                    mime="text/csv",
                )
