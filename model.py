import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# Helper functions
# ---------------------------------------------------------
def load_and_prepare_data():
    """
    Load BTS baggage & T100 domestic data, merge, and engineer features.
    Expects:
      - bts_baggage_fees_clean.csv
      - t100_domestic_clean.csv
    """

    try:
        baggage = pd.read_csv("bts_baggage_fees_clean.csv")
        t100 = pd.read_csv("t100_domestic_clean.csv")
    except FileNotFoundError as e:
        st.error(
            f"Could not find required CSV files.\n\n"
            f"Make sure these exist in the same folder as this app:\n"
            f"- bts_baggage_fees_clean.csv\n"
            f"- t100_domestic_clean.csv\n\n"
            f"Details: {e}"
        )
        st.stop()

    baggage.columns = baggage.columns.str.lower()
    t100.columns = t100.columns.str.lower()

    needed_baggage_cols = ["year", "quarter", "carrier", "baggage_fees_usd"]
    needed_t100_cols = [
        "year", "quarter", "carrier",
        "passengers", "departures", "rpm", "asm", "load_factor"
    ]

    for col in needed_baggage_cols:
        if col not in baggage.columns:
            st.error(f"Column '{col}' missing from baggage CSV.")
            st.stop()

    for col in needed_t100_cols:
        if col not in t100.columns:
            st.error(f"Column '{col}' missing from T100 CSV.")
            st.stop()

    df = baggage[needed_baggage_cols].merge(
        t100[needed_t100_cols],
        on=["year", "quarter", "carrier"],
        how="inner"
    )

    # Adjust these aliases to match your carrier naming
    southwest_aliases = ["Southwest Airlines", "Southwest Airlines Co.", "WN"]
    df = df[df["carrier"].isin(southwest_aliases)].copy()

    if df.empty:
        st.error(
            "No Southwest rows found after filtering.\n\n"
            "Check how 'carrier' is named in your CSVs and update 'southwest_aliases' in the code."
        )
        st.stop()

    # Time index from (year, quarter)
    df["period"] = pd.PeriodIndex(
        year=df["year"].astype(int),
        quarter=df["quarter"].astype(int)
    )
    df["date"] = df["period"].dt.to_timestamp(how="end")

    df = df.sort_values("date").reset_index(drop=True)

    # Cyclical encoding for quarter
    df["quarter_sin"] = np.sin(2 * np.pi * (df["quarter"] - 1) / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * (df["quarter"] - 1) / 4)

    # Lag features
    df["baggage_fees_lag1"] = df["baggage_fees_usd"].shift(1)
    df["passengers_lag1"] = df["passengers"].shift(1)

    # Drop first row with NaNs from lag
    df = df.dropna().reset_index(drop=True)

    return df


def train_model(df):
    """
    Train a HistGradientBoosting model to predict baggage_fees_usd.
    Returns the fitted pipeline and (X, y).
    """

    numeric_features = [
        "passengers",
        "departures",
        "rpm",
        "asm",
        "load_factor",
        "quarter_sin",
        "quarter_cos",
        "baggage_fees_lag1",
        "passengers_lag1",
    ]

    categorical_features = ["year"]

    X = df[numeric_features + categorical_features]
    y = df["baggage_fees_usd"]

    numeric_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        max_depth=4,
        learning_rate=0.05,
        max_iter=300,
        random_state=42,
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", regressor),
        ]
    )

    # Simple train/test split: last 20% as test (still chronological)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
    }

    return model, X, y, numeric_features, categorical_features, metrics


def make_future_row(
    year, quarter,
    passengers, departures, rpm, asm,
    load_factor,
    prev_baggage_fees, prev_passengers
):
    quarter_sin = np.sin(2 * np.pi * (quarter - 1) / 4)
    quarter_cos = np.cos(2 * np.pi * (quarter - 1) / 4)

    data = {
        "passengers": [passengers],
        "departures": [departures],
        "rpm": [rpm],
        "asm": [asm],
        "load_factor": [load_factor],
        "quarter_sin": [quarter_sin],
        "quarter_cos": [quarter_cos],
        "baggage_fees_lag1": [prev_baggage_fees],
        "passengers_lag1": [prev_passengers],
        "year": [year],
    }

    return pd.DataFrame(data)


# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Southwest Baggage Revenue Dashboard",
    layout="wide"
)

st.title("✈️ Southwest Baggage Revenue Prediction Dashboard")
st.caption(
    "Aggie Data Science Club × Southwest Airlines — "
    "prototype model using public BTS data"
)

with st.spinner("Loading data and training model..."):
    df = load_and_prepare_data()
    model, X, y, num_feats, cat_feats, metrics = train_model(df)

# ---------------------------------------------------------
# Top-level summary / KPIs
# ---------------------------------------------------------
latest = df.iloc[-1]
if len(df) >= 5:
    # Compare last quarter vs same quarter previous year (approx 4 quarters back)
    prev_year_idx = len(df) - 5
    prev_year_row = df.iloc[prev_year_idx]
    yoy_change = (
        latest["baggage_fees_usd"] - prev_year_row["baggage_fees_usd"]
    ) / prev_year_row["baggage_fees_usd"] * 100
else:
    yoy_change = np.nan

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Quarter Baggage Revenue (USD)",
        f"${latest['baggage_fees_usd']:,.0f}"
    )

with col2:
    st.metric(
        "Passengers (Latest Quarter)",
        f"{latest['passengers']:,.0f}"
    )

with col3:
    yoy_label = (
        f"{yoy_change:,.1f}% vs same quarter last year"
        if not np.isnan(yoy_change)
        else "N/A"
    )
    st.metric(
        "Model Test RMSE (USD)",
        f"${metrics['RMSE']:,.0f}",
        help=yoy_label
    )

st.markdown("---")

# ---------------------------------------------------------
# Layout: left = time series, right = simulator
# ---------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Baggage Revenue Over Time")
    ts_df = df[["date", "baggage_fees_usd"]].set_index("date")
    st.line_chart(ts_df.rename(columns={"baggage_fees_usd": "Baggage Fees (USD)"}))

    with st.expander("Show raw Southwest quarterly data"):
        st.dataframe(
            df[[
                "year", "quarter", "date",
                "baggage_fees_usd", "passengers",
                "departures", "rpm", "asm", "load_factor"
            ]]
        )

with right:
    st.subheader("📈 What-If Revenue Simulator")

    st.markdown(
        "Use this panel to simulate baggage revenue for a future quarter "
        "by adjusting demand and operational inputs."
    )

    latest_year = int(latest["year"])
    latest_quarter = int(latest["quarter"])

    sim_year = st.number_input(
        "Year",
        min_value=2007,
        max_value=2100,
        value=latest_year + 1
    )
    sim_quarter = st.selectbox(
        "Quarter",
        options=[1, 2, 3, 4],
        index=0
    )

    st.markdown("**Demand & Operations**")
    passengers = st.number_input(
        "Passengers",
        min_value=0,
        value=int(latest["passengers"]),
        step=100000
    )
    departures = st.number_input(
        "Departures",
        min_value=0,
        value=int(latest["departures"]),
        step=5000
    )
    rpm = st.number_input(
        "Revenue Passenger Miles (RPM)",
        min_value=0,
        value=int(latest["rpm"]),
        step=1_000_000_000
    )
    asm = st.number_input(
        "Available Seat Miles (ASM)",
        min_value=0,
        value=int(latest["asm"]),
        step=1_000_000_000
    )
    load_factor = st.slider(
        "Load Factor",
        min_value=0.4,
        max_value=1.0,
        value=float(latest["load_factor"]),
        step=0.01
    )

    st.markdown("**Previous Quarter Context**")

    prev_baggage = st.number_input(
        "Previous Quarter Baggage Fees (USD)",
        min_value=0,
        value=int(latest["baggage_fees_usd"])
    )
    prev_passengers = st.number_input(
        "Previous Quarter Passengers",
        min_value=0,
        value=int(latest["passengers"])
    )

    if st.button("Predict Baggage Revenue"):
        future_X = make_future_row(
            year=int(sim_year),
            quarter=int(sim_quarter),
            passengers=passengers,
            departures=departures,
            rpm=rpm,
            asm=asm,
            load_factor=load_factor,
            prev_baggage_fees=prev_baggage,
            prev_passengers=prev_passengers
        )

        pred = model.predict(future_X)[0]
        st.success(f"Predicted baggage revenue: **${pred:,.0f}**")

        st.caption(
            "Model: HistGradientBoostingRegressor on BTS baggage + T100 data, "
            "with seasonality + lag features."
        )

st.markdown("---")

st.subheader("Model Details")

st.markdown(
    """
- **Target:** Quarterly baggage fee revenue for Southwest (USD)  
- **Inputs:** Passengers, departures, RPM, ASM, load factor, seasonality (quarter), and lagged baggage/passenger values  
- **Model:** HistGradientBoostingRegressor (tree-based gradient boosting)  
- **Data Source:** Public U.S. Bureau of Transportation Statistics (BTS) datasets
    """
)
