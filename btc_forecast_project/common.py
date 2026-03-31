import numpy as np
import pandas as pd

HORIZON = 48


def candle_features(df):

    df = df.copy()

    df["body"] = df["close"] - df["open"]
    df["range"] = df["high"] - df["low"]

    df["upper_wick"] = df["high"] - df[["open","close"]].max(axis=1)
    df["lower_wick"] = df[["open","close"]].min(axis=1) - df["low"]

    df["body_ratio"] = df["body"] / (df["range"] + 1e-9)
    df["upper_ratio"] = df["upper_wick"] / (df["range"] + 1e-9)
    df["lower_ratio"] = df["lower_wick"] / (df["range"] + 1e-9)

    return df


def rolling_features(df):

    df = df.copy()

    df["ret"] = df["close"].pct_change()

    df["volatility_12"] = df["ret"].rolling(12).std()
    df["volatility_48"] = df["ret"].rolling(48).std()

    df["volume_mean_48"] = df["volume"].rolling(48).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_mean_48"] + 1e-9)

    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_48"] = df["close"].ewm(span=48).mean()

    df["ema_spread"] = df["ema_12"] - df["ema_48"]

    return df


def rsi(series, period=14):

    delta = series.diff()

    up = delta.clip(lower=0)
    down = -1 * delta.clip(upper=0)

    ma_up = up.rolling(period).mean()
    ma_down = down.rolling(period).mean()

    rs = ma_up / (ma_down + 1e-9)

    return 100 - (100 / (1 + rs))


def advanced_features(df):

    df = df.copy()

    df["rsi"] = rsi(df["close"])

    df["vwap"] = (df["volume"] * (df["high"] + df["low"] + df["close"]) / 3).cumsum() / df["volume"].cumsum()

    df["vwap_dev"] = df["close"] - df["vwap"]

    df["compression"] = df["range"].rolling(20).mean()

    return df


def build_features(df):

    df = candle_features(df)
    df = rolling_features(df)
    df = advanced_features(df)

    df = df.dropna()

    return df


def build_targets(df):
    """
    Predict future relative returns from the current close:
    t1 = (close[t+1] / close[t]) - 1
    ...
    t48 = (close[t+48] / close[t]) - 1
    """
    targets = []
    current_close = df["close"]

    for i in range(1, HORIZON + 1):
        future_close = df["close"].shift(-i)
        rel_return = (future_close / current_close) - 1.0
        targets.append(rel_return)

    y = pd.concat(targets, axis=1)
    y.columns = [f"t{i}" for i in range(1, HORIZON + 1)]
    return y