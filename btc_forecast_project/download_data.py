import os
import time
from datetime import datetime, timezone

import ccxt
import pandas as pd

DATA_PATH = "data/btcusdt_5m.parquet"

SYMBOL = "BTC/USDT"
TIMEFRAME = "5m"
LIMIT = 1000  # Binance spot max is 1000 per request

# Choose how much history you want on first/full download
START_DATE_UTC = "2022-01-01 00:00:00"

exchange = ccxt.binance({
    "enableRateLimit": True,
})


def ms_from_utc_string(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


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


def load_existing() -> pd.DataFrame:
    if os.path.exists(DATA_PATH):
        df = pd.read_parquet(DATA_PATH)
        df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        return df
    return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def save_df(df: pd.DataFrame) -> None:
    os.makedirs("data", exist_ok=True)
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df.to_parquet(DATA_PATH, index=False)


def fetch_historical(full_refresh: bool = False) -> None:
    tf_ms = timeframe_to_ms(TIMEFRAME)
    now_ms = exchange.milliseconds()

    df_existing = load_existing()

    if full_refresh or df_existing.empty:
        since = ms_from_utc_string(START_DATE_UTC)
        df_all = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        print(f"Starting full download from {START_DATE_UTC} UTC")
    else:
        # Continue from the next expected candle after last saved timestamp
        last_ts = int(df_existing["timestamp"].iloc[-1])
        since = last_ts + tf_ms
        df_all = df_existing.copy()
        print(f"Starting incremental download from {pd.to_datetime(since, unit='ms', utc=True)}")

    total_new = 0

    while since < now_ms:
        candles = exchange.fetch_ohlcv(
            SYMBOL,
            timeframe=TIMEFRAME,
            since=since,
            limit=LIMIT,
        )

        if not candles:
            print("No more candles returned.")
            break

        chunk = pd.DataFrame(
            candles,
            columns=["timestamp", "open", "high", "low", "close", "volume"]
        )

        if chunk.empty:
            print("Empty chunk returned.")
            break

        chunk = chunk.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

        first_ts = int(chunk["timestamp"].iloc[0])
        last_ts = int(chunk["timestamp"].iloc[-1])

        before_len = len(df_all)
        df_all = pd.concat([df_all, chunk], ignore_index=True)
        df_all = df_all.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
        added = len(df_all) - before_len
        total_new += max(0, added)

        save_df(df_all)

        print(
            f"Fetched {len(chunk)} candles | "
            f"added {added} new | "
            f"range {pd.to_datetime(first_ts, unit='ms', utc=True)} -> "
            f"{pd.to_datetime(last_ts, unit='ms', utc=True)} | "
            f"total saved {len(df_all)}"
        )

        # Move to next candle after the last one we received
        since = last_ts + tf_ms

        # Safety: if exchange repeats the same last candle forever, stop
        if len(chunk) == 1 and first_ts >= now_ms - tf_ms:
            break

        time.sleep(exchange.rateLimit / 1000)

    print(f"Done. Total saved candles: {len(df_all)} | New candles added this run: {total_new}")


if __name__ == "__main__":
    os.makedirs("data", exist_ok=True)

    # Set to True once if you want to rebuild from START_DATE_UTC
    FULL_REFRESH = True

    fetch_historical(full_refresh=FULL_REFRESH)