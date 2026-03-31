import os
import time
from datetime import timedelta

import ccxt
import joblib
import pandas as pd

from dash import Dash, dcc, html, Input, Output
import plotly.graph_objects as go

from common import build_features, HORIZON

DATA_PATH = "data/btcusdt_5m.parquet"
MODEL_PATH = "models/best_model.joblib"

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
LIMIT = 1000
HISTORY_BARS = 200

exchange = ccxt.binance({
    "enableRateLimit": True,
})


def timeframe_to_ms(timeframe: str) -> int:
    mapping = {
        "1m": 60_000,
        "3m": 3 * 60_000,
        "5m": 5 * 60_000,
        "15m": 15 * 60_000,
        "30m": 30 * 60_000,
        "1h": 60 * 60_000,
        "4h": 4 * 60 * 60_000,
        "1d": 24 * 60 * 60_000,
    }
    if timeframe not in mapping:
        raise ValueError(f"Unsupported timeframe: {timeframe}")
    return mapping[timeframe]


def load_local_data() -> pd.DataFrame:
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Missing {DATA_PATH}. Run download_data.py first."
        )
    df = pd.read_parquet(DATA_PATH)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    return df


def save_local_data(df: pd.DataFrame) -> None:
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_parquet(DATA_PATH, index=False)


def update_missing_candles() -> pd.DataFrame:
    df_existing = load_local_data()
    tf_ms = timeframe_to_ms(TIMEFRAME)

    if df_existing.empty:
        raise ValueError("Local data file is empty.")

    last_ts = int(df_existing["timestamp"].iloc[-1])
    since = last_ts + tf_ms
    now_ms = exchange.milliseconds()

    all_new = []

    while since < now_ms:
        candles = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=LIMIT
        )

        if not candles:
            break

        chunk = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        if chunk.empty:
            break

        chunk = chunk.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        all_new.append(chunk)

        new_last_ts = int(chunk["timestamp"].iloc[-1])

        if new_last_ts < since:
            break

        since = new_last_ts + tf_ms
        time.sleep(exchange.rateLimit / 1000)

        if len(chunk) < LIMIT:
            break

    if all_new:
        df_new = pd.concat(all_new, ignore_index=True)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
        df_all = df_all.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        save_local_data(df_all)
        return df_all

    return df_existing


def make_future_times(last_dt: pd.Timestamp, periods: int, freq_minutes: int = 5):
    return [last_dt + timedelta(minutes=freq_minutes * i) for i in range(1, periods + 1)]


app = Dash(__name__)

app.layout = html.Div([
    html.H3("BTC/USDT 5m Live Prediction"),

    html.Button("Update Data", id="update", n_clicks=0),

    html.Pre(id="info", style={"marginTop": "15px"}),

    dcc.Graph(id="chart")
])


@app.callback(
    Output("info", "children"),
    Output("chart", "figure"),
    Input("update", "n_clicks")
)
def update_chart(n_clicks):
    df_raw = update_missing_candles()
    df_feat = build_features(df_raw).reset_index(drop=True)

    if len(df_feat) < HISTORY_BARS + 1:
        raise ValueError("Not enough processed rows to plot live forecast.")

    model = joblib.load(MODEL_PATH)

    df_feat["dt"] = pd.to_datetime(df_feat["timestamp"], unit="ms")

    history = df_feat.iloc[-HISTORY_BARS:].copy()
    last_row = df_feat.iloc[-1]

    X = last_row.drop(["timestamp", "dt"]).values.reshape(1, -1)

    # If your model now predicts relative returns:
    pred_rel = model.predict(X)[0]
    anchor_close = float(last_row["close"])
    pred = anchor_close * (1.0 + pred_rel)

    # Future timestamps for the 48 predicted candles
    last_dt = last_row["dt"]
    pred_times = make_future_times(last_dt, HORIZON, freq_minutes=5)

    divider_time = last_dt

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=history["dt"],
        y=history["close"],
        mode="lines",
        name="current"
    ))

    y_min = min(history["close"].min(), pred.min())
    y_max = max(history["close"].max(), pred.max())

    fig.add_trace(go.Scatter(
        x=[divider_time, divider_time],
        y=[y_min, y_max],
        mode="lines",
        name="prediction start",
        line=dict(dash="dash")
    ))

    fig.add_trace(go.Scatter(
        x=pred_times,
        y=pred,
        mode="lines",
        name="prediction"
    ))

    fig.update_layout(
        title="Live Prediction",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )

    info = (
        f"Last candle time: {divider_time}\n"
        f"Last close: {anchor_close:.2f}\n"
        f"History bars shown: {HISTORY_BARS}\n"
        f"Forecast horizon: {HORIZON} bars ({HORIZON * 5} minutes)\n"
        f"Total local candles: {len(df_raw)}\n"
        f"Usable feature rows: {len(df_feat)}"
    )

    return info, fig


if __name__ == "__main__":
    app.run(port=8051, debug=True)