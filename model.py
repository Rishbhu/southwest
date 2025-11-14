import numpy as np
import pandas as pd
import streamlit as st

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ---------------------------------------------------------
# 1. Dummy data generator
# ---------------------------------------------------------
def generate_dummy_data(start_year=2015, end_year=2024):
    """
    Generate synthetic quarterly data for a single airline (Southwest-like).
    Columns:
      year, quarter, date, passengers, departures, rpm, asm,
      load_factor, baggage_fees_usd
    """

    rows = []
    rng = np.random.default_rng(42)

    for year in range(start_year, end_year + 1):
        for q in range(1, 5):
            years_since_start = year - start_year

            seasonal_boost = {
                1: 0.95,
                2: 1.00,
                3: 1.10,
                4: 1.05,
            }[q]

            base_passengers = 18_000_000
            passengers = (
                base_passengers
                * (1 + 0.04 * years_since_start)
                * seasonal_boost
            )
            passengers += rng.normal(0, 800_000)
            passengers = max(passengers, 5_000_000)

            departures = passengers / 150 + rng.normal(0, 500)
            departures = max(departures, 30_000)

            load_factor = 0.76 + rng.normal(0, 0.03)
            load_factor = float(np.clip(load_factor, 0.7, 0.9))

            asm = passengers / load_factor * 900
            rpm = asm * load_factor

            avg_bag_revenue_per_pax = 3.5 + rng.normal(0, 0.4)
            baggage_fees_usd = passengers * avg_bag_revenue_per_pax
            baggage_fees_usd += rng.normal(0, 5_000_000)
            baggage_fees_usd = max(baggage_fees_usd, 10_000_000)

            # 🔧 FIXED: use string "YYYYQX" instead of year= / quarter=
            period = pd.Period(f"{year}Q{q}")
            date = period.to_timestamp(how="end")

            rows.append(
                {
                    "year": year,
                    "quarter": q,
                    "date": date,
                    "passengers": passengers,
                    "departures": departures,
                    "rpm": rpm,
                    "asm": asm,
                    "load_factor": load_factor,
                    "baggage_fees_usd": baggage_fees_usd,
                }
            )

    df = pd.DataFrame(rows).sort_values("date").reset_index(drop=True)

    df["quarter_sin"] = np.sin(2 * np.pi * (df["quarter"] - 1) / 4)
    df["quarter_cos"] = np.cos(2 * np.pi * (df["quarter"] - 1) / 4)

    df["baggage_fees_lag1"] = df["baggage_fees_usd"].shift(1)
    df["passengers_lag1"] = df["passengers"].shift(1)

    df = df.dropna().reset_index(drop=True)

    return df



# ---------------------------------------------------------
# 2. Model training
# ---------------------------------------------------------
def train_model(df):
    """
    Train a HistGradientBoosting model to predict baggage_fees_usd
    from synthetic quarterly data.
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

    # Chronological train/test split: last 20% is test
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    rmse = mean_squared_error(y_test, y_pred, squared=False)

    metrics = {"MAE": mae, "RMSE": rmse}

    return model, X, y, numeric_features, categorical_features, metrics


# ---------------------------------------------------------
# 3. Helper to build future row for simulator
# ---------------------------------------------------------
def make_future_row(
    year,
    quarter,
    passengers,
    departures,
    rpm,
    asm,
    load_factor,
    prev_baggage_fees,
    prev_passengers,
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
# 4. Streamlit UI
# ---------------------------------------------------------
st.set_page_config(
    page_title="Southwest Baggage Revenue Prototype",
    layout="wide",
)

st.title("✈️ Baggage Revenue Prediction – Prototype Dashboard")
st.caption(
    "Aggie Data Science Club × Southwest Airlines (demo with synthetic data)"
)

with st.spinner("Generating synthetic data and training model..."):
    df = generate_dummy_data()
    model, X, y, num_feats, cat_feats, metrics = train_model(df)

latest = df.iloc[-1]

# Compute YoY approx (4 quarters back)
if len(df) >= 5:
    prev_year_row = df.iloc[-5]
    yoy_change = (
        (latest["baggage_fees_usd"] - prev_year_row["baggage_fees_usd"])
        / prev_year_row["baggage_fees_usd"]
        * 100
    )
else:
    yoy_change = np.nan

# ---------------------------------------------------------
# KPI row
# ---------------------------------------------------------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Quarter Baggage Revenue (USD)",
        f"${latest['baggage_fees_usd']:,.0f}",
    )

with col2:
    st.metric(
        "Passengers (Latest Quarter)",
        f"{latest['passengers']:,.0f}",
    )

with col3:
    st.metric(
        "Model Test RMSE (USD)",
        f"${metrics['RMSE']:,.0f}",
        help=(
            f"Approx. YoY change in revenue: {yoy_change:,.1f}%"
            if not np.isnan(yoy_change)
            else "YoY change not available"
        ),
    )

st.markdown("---")

# ---------------------------------------------------------
# Layout: left = chart, right = simulator
# ---------------------------------------------------------
left, right = st.columns([2, 1])

with left:
    st.subheader("Baggage Revenue Over Time (Synthetic)")
    ts_df = df[["date", "baggage_fees_usd"]].set_index("date")
    st.line_chart(ts_df.rename(columns={"baggage_fees_usd": "Baggage Fees (USD)"}))

    with st.expander("Show raw quarterly data"):
        st.dataframe(
            df[
                [
                "year",
                "quarter",
                "date",
                "baggage_fees_usd",
                "passengers",
                "departures",
                "rpm",
                "asm",
                "load_factor",
                ]
            ]
        )

with right:
    st.subheader("📈 What-If Revenue Simulator")

    st.markdown(
        "Adjust demand and operational inputs to simulate predicted baggage revenue "
        "for a future quarter."
    )

    latest_year = int(latest["year"])
    latest_quarter = int(latest["quarter"])

    sim_year = st.number_input(
        "Year",
        min_value=2010,
        max_value=2100,
        value=latest_year + 1,
    )
    sim_quarter = st.selectbox("Quarter", options=[1, 2, 3, 4], index=0)

    st.markdown("**Demand & Operations**")

    passengers = st.number_input(
        "Passengers",
        min_value=0,
        value=int(latest["passengers"]),
        step=100_000,
    )
    departures = st.number_input(
        "Departures",
        min_value=0,
        value=int(latest["departures"]),
        step=5_000,
    )
    rpm = st.number_input(
        "Revenue Passenger Miles (RPM)",
        min_value=0,
        value=int(latest["rpm"]),
        step=1_000_000_000,
    )
    asm = st.number_input(
        "Available Seat Miles (ASM)",
        min_value=0,
        value=int(latest["asm"]),
        step=1_000_000_000,
    )
    load_factor = st.slider(
        "Load Factor",
        min_value=0.6,
        max_value=0.95,
        value=float(latest["load_factor"]),
        step=0.01,
    )

    st.markdown("**Previous Quarter Context**")

    prev_baggage = st.number_input(
        "Previous Quarter Baggage Fees (USD)",
        min_value=0,
        value=int(latest["baggage_fees_usd"]),
    )
    prev_passengers = st.number_input(
        "Previous Quarter Passengers",
        min_value=0,
        value=int(latest["passengers"]),
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
            prev_passengers=prev_passengers,
        )

        pred = model.predict(future_X)[0]

        st.success(f"Predicted baggage revenue: **${pred:,.0f}**")
        st.caption(
            "Model: gradient-boosted trees on synthetic quarterly data with "
            "seasonality and lag features."
        )

st.markdown("---")

st.subheader("Model Summary")
st.markdown(
    """
- **Target:** Quarterly baggage revenue (synthetic, USD)  
- **Inputs:** Passengers, departures, RPM, ASM, load factor, quarter (seasonality), lagged baggage & passengers  
- **Model:** HistGradientBoostingRegressor (tree-based gradient boosting)  
- **Note:** This is a *prototype* using dummy data; the same pipeline can be fed real Southwest/BTS data.
    """
)
