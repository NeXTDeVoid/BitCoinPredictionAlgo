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

    df["ret_1"] = df["close"].pct_change(1)
    df["ret_3"] = df["close"].pct_change(3)
    df["ret_6"] = df["close"].pct_change(6)
    df["ret_12"] = df["close"].pct_change(12)
    df["ret_24"] = df["close"].pct_change(24)
    df["ret_48"] = df["close"].pct_change(48)

    df["volatility_12"] = df["ret"].rolling(12).std()
    df["volatility_48"] = df["ret"].rolling(48).std()

    df["ret_mean_12"] = df["ret"].rolling(12).mean()
    df["ret_mean_48"] = df["ret"].rolling(48).mean()

    df["ret_z_12"] = (
        (df["ret"] - df["ret"].rolling(12).mean()) /
        (df["ret"].rolling(12).std() + 1e-9)
    )
    df["ret_z_48"] = (
        (df["ret"] - df["ret"].rolling(48).mean()) /
        (df["ret"].rolling(48).std() + 1e-9)
    )

    df["volume_mean_48"] = df["volume"].rolling(48).mean()
    df["volume_ratio"] = df["volume"] / (df["volume_mean_48"] + 1e-9)

    df["volume_ma_12"] = df["volume"].rolling(12).mean()
    df["volume_ma_48"] = df["volume"].rolling(48).mean()
    df["volume_ratio_12"] = df["volume"] / (df["volume_ma_12"] + 1e-9)
    df["volume_ratio_48"] = df["volume"] / (df["volume_ma_48"] + 1e-9)

    df["dollar_volume"] = df["close"] * df["volume"]
    df["dollar_volume_ma_48"] = df["dollar_volume"].rolling(48).mean()
    df["dollar_volume_ratio"] = df["dollar_volume"] / (df["dollar_volume_ma_48"] + 1e-9)

    df["ema_12"] = df["close"].ewm(span=12).mean()
    df["ema_48"] = df["close"].ewm(span=48).mean()
    df["ema_spread"] = df["ema_12"] - df["ema_48"]

    df["range_ma_12"] = df["range"].rolling(12).mean()
    df["range_ma_48"] = df["range"].rolling(48).mean()
    df["range_ratio_12"] = df["range"] / (df["range_ma_12"] + 1e-9)
    df["range_ratio_48"] = df["range"] / (df["range_ma_48"] + 1e-9)

    df["body_abs"] = (df["close"] - df["open"]).abs()
    df["body_abs_ma_12"] = df["body_abs"].rolling(12).mean()
    df["body_abs_ratio_12"] = df["body_abs"] / (df["body_abs_ma_12"] + 1e-9)

    df["roll_high_12"] = df["high"].rolling(12).max()
    df["roll_low_12"] = df["low"].rolling(12).min()
    df["roll_high_48"] = df["high"].rolling(48).max()
    df["roll_low_48"] = df["low"].rolling(48).min()

    df["dist_high_12"] = (df["roll_high_12"] - df["close"]) / (df["close"] + 1e-9)
    df["dist_low_12"] = (df["close"] - df["roll_low_12"]) / (df["close"] + 1e-9)
    df["dist_high_48"] = (df["roll_high_48"] - df["close"]) / (df["close"] + 1e-9)
    df["dist_low_48"] = (df["close"] - df["roll_low_48"]) / (df["close"] + 1e-9)

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

    df["vwap"] = (
        df["volume"] * (df["high"] + df["low"] + df["close"]) / 3
    ).cumsum() / df["volume"].cumsum()

    df["vwap_dev"] = df["close"] - df["vwap"]
    df["compression"] = df["range"].rolling(20).mean()

    dt = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    minute_of_day = dt.dt.hour * 60 + dt.dt.minute
    day_of_week = dt.dt.dayofweek

    df["minute_sin"] = np.sin(2 * np.pi * minute_of_day / 1440)
    df["minute_cos"] = np.cos(2 * np.pi * minute_of_day / 1440)
    df["dow_sin"] = np.sin(2 * np.pi * day_of_week / 7)
    df["dow_cos"] = np.cos(2 * np.pi * day_of_week / 7)

    df["ema_12_slope"] = df["ema_12"] - df["ema_12"].shift(3)
    df["ema_48_slope"] = df["ema_48"] - df["ema_48"].shift(3)
    df["price_vs_ema12"] = (df["close"] - df["ema_12"]) / (df["close"] + 1e-9)
    df["price_vs_ema48"] = (df["close"] - df["ema_48"]) / (df["close"] + 1e-9)

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
