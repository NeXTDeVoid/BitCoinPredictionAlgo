import random
import pandas as pd
import joblib

from dash import Dash, dcc, html, Input, Output, State
import plotly.graph_objects as go

from common import build_features, HORIZON

MODEL_PATH = "models/best_model.joblib"
DATA_PATH = "data/btcusdt_5m.parquet"

# =========================
# Load data
# =========================
df_raw = pd.read_parquet(DATA_PATH).reset_index(drop=True)
df_feat = build_features(df_raw).reset_index(drop=True)
model = joblib.load(MODEL_PATH)

df_raw["dt"] = pd.to_datetime(df_raw["timestamp"], unit="ms")
df_feat["dt"] = pd.to_datetime(df_feat["timestamp"], unit="ms")

# =========================
# Valid index range
# =========================
HISTORY_BARS = 200
MIN_INDEX = HISTORY_BARS
MAX_INDEX = len(df_feat) - HORIZON - 1

if MAX_INDEX <= MIN_INDEX:
    raise ValueError(
        f"Not enough data for backtesting. "
        f"len(df_feat)={len(df_feat)}, need more than {HISTORY_BARS + HORIZON} usable rows."
    )

DEFAULT_INDEX = min(1000, MAX_INDEX)

app = Dash(__name__)

app.layout = html.Div([
    html.H3("BTC/USDT 5m Backtest"),

    html.Div([
        html.Label(f"Index ({MIN_INDEX} to {MAX_INDEX})"),
        dcc.Input(
            id="index",
            type="number",
            value=DEFAULT_INDEX,
            min=MIN_INDEX,
            max=MAX_INDEX,
            step=1
        ),
        html.Button("Random Date", id="random-btn", n_clicks=0, style={"marginLeft": "10px"}),
        html.Button("Run Backtest", id="run-btn", n_clicks=0, style={"marginLeft": "10px"}),
    ], style={"marginBottom": "20px"}),

    html.Pre(id="info", style={"marginBottom": "20px"}),

    html.Div([
        dcc.Graph(id="pred-chart", style={"width": "49%", "display": "inline-block"}),
        dcc.Graph(id="actual-chart", style={"width": "49%", "display": "inline-block"}),
    ])
])


@app.callback(
    Output("index", "value"),
    Input("random-btn", "n_clicks"),
    prevent_initial_call=True
)
def pick_random_index(n_clicks):
    return random.randint(MIN_INDEX, MAX_INDEX)


@app.callback(
    Output("info", "children"),
    Output("pred-chart", "figure"),
    Output("actual-chart", "figure"),
    Input("run-btn", "n_clicks"),
    State("index", "value")
)
def run_backtest(n_clicks, index):
    if index is None:
        index = DEFAULT_INDEX

    # dcc.Input(type="number") can return float, so force int
    index = int(index)

    # clamp to valid range
    index = max(MIN_INDEX, min(index, MAX_INDEX))

    row = df_feat.iloc[index]
    X = row.drop(["timestamp", "dt"]).values.reshape(1, -1)
    pred_rel = model.predict(X)[0]
    anchor_close = float(row["close"])
    pred = anchor_close * (1.0 + pred_rel)

    history = df_feat.iloc[index - HISTORY_BARS:index].copy()
    future_real = df_feat.iloc[index + 1:index + 1 + HORIZON].copy()

    divider_time = row["dt"]
    pred_times = future_real["dt"].tolist()

    # =========================
    # Left chart: history + prediction
    # =========================
    fig_pred = go.Figure()

    fig_pred.add_trace(go.Scatter(
        x=history["dt"],
        y=history["close"],
        mode="lines",
        name="history"
    ))

    y_min_pred = min(history["close"].min(), pred.min())
    y_max_pred = max(history["close"].max(), pred.max())

    fig_pred.add_trace(go.Scatter(
        x=[divider_time, divider_time],
        y=[y_min_pred, y_max_pred],
        mode="lines",
        name="prediction start",
        line=dict(dash="dash")
    ))

    fig_pred.add_trace(go.Scatter(
        x=pred_times,
        y=pred,
        mode="lines",
        name="prediction"
    ))

    fig_pred.update_layout(
        title="Prediction",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )

    # =========================
    # Right chart: history + actual
    # =========================
    fig_actual = go.Figure()

    fig_actual.add_trace(go.Scatter(
        x=history["dt"],
        y=history["close"],
        mode="lines",
        name="history"
    ))

    y_min_act = min(history["close"].min(), future_real["close"].min())
    y_max_act = max(history["close"].max(), future_real["close"].max())

    fig_actual.add_trace(go.Scatter(
        x=[divider_time, divider_time],
        y=[y_min_act, y_max_act],
        mode="lines",
        name="actual start",
        line=dict(dash="dash")
    ))

    fig_actual.add_trace(go.Scatter(
        x=future_real["dt"],
        y=future_real["close"],
        mode="lines",
        name="actual"
    ))

    fig_actual.update_layout(
        title="Actual Future",
        xaxis_title="Time",
        yaxis_title="Price",
        template="plotly_dark"
    )

    # =========================
    # Metrics
    # =========================
    actual = future_real["close"].values

    mae = float(abs(pred - actual).mean())
    rmse = float(((pred - actual) ** 2).mean() ** 0.5)

    if len(actual) > 1:
        pred_dir = (pred[1:] - pred[:-1]) > 0
        actual_dir = (actual[1:] - actual[:-1]) > 0
        direction_acc = float((pred_dir == actual_dir).mean())
    else:
        direction_acc = 0.0

    info = (
        f"Chosen index: {index}\n"
        f"Timestamp: {divider_time}\n"
        f"Valid range: {MIN_INDEX} to {MAX_INDEX}\n"
        f"History bars shown: {HISTORY_BARS}\n"
        f"Forecast horizon: {HORIZON} bars ({HORIZON * 5} minutes)\n"
        f"MAE: {mae:.2f}\n"
        f"RMSE: {rmse:.2f}\n"
        f"Directional accuracy: {direction_acc:.2%}"
    )

    return info, fig_pred, fig_actual


if __name__ == "__main__":
    app.run(debug=True)