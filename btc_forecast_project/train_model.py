import pandas as pd
import numpy as np
import joblib
import os
import json

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

from common import build_features, build_targets


DATA_PATH = "data/btcusdt_5m.parquet"
MODEL_PATH = "models/best_model.joblib"
META_PATH = "models/best_model_meta.json"

WINDOW = 500
STEP = 100


def composite_score(pred, real):
    mae = np.mean(np.abs(pred - real))
    mae_score = 1.0 / (1.0 + mae)

    direction = np.mean(
        np.sign(pred[:, 1:] - pred[:, :-1]) ==
        np.sign(real[:, 1:] - real[:, :-1])
    )

    pred_diff = pred[:, 1:] - pred[:, :-1]
    real_diff = real[:, 1:] - real[:, :-1]
    movement_match = np.mean(np.abs(pred_diff - real_diff))
    movement_score = 1.0 / (1.0 + movement_match)

    score = 0.50 * movement_score + 0.30 * direction + 0.20 * mae_score
    return score

def load_best_score():
    if os.path.exists(META_PATH):
        try:
            with open(META_PATH, "r") as f:
                meta = json.load(f)
            return float(meta.get("best_score", -1))
        except Exception:
            return -1
    return -1


def save_best_score(score):
    os.makedirs("models", exist_ok=True)
    with open(META_PATH, "w") as f:
        json.dump({"best_score": float(score)}, f)


def train():
    df = pd.read_parquet(DATA_PATH)

    df = build_features(df)

    X = df.drop(columns=["timestamp"])
    print("TRAIN FEATURE COUNT:", X.shape[1])
    print("TRAIN FEATURES:")
    print(list(X.columns))
    y = build_targets(df)

    X = X.iloc[:-48]
    y = y.iloc[:-48]

    best_score = load_best_score()
    print("Starting best score:", best_score)

    for i in range(WINDOW, len(X) - STEP, STEP):
        X_train = X.iloc[i - WINDOW:i]
        y_train = y.iloc[i - WINDOW:i]

        X_test = X.iloc[i:i + STEP]
        y_test = y.iloc[i:i + STEP]

        model = MultiOutputRegressor(
            HistGradientBoostingRegressor(
                max_depth=6,
                max_iter=200
            )
        )

        model.fit(X_train, y_train)

        pred = model.predict(X_test)

        score = composite_score(pred, y_test.values)

        if score > best_score:
            best_score = score

            os.makedirs("models", exist_ok=True)
            joblib.dump(model, MODEL_PATH)
            save_best_score(best_score)

            print("New best model", score)


if __name__ == "__main__":
    train()
