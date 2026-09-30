# ============================================================
# SkyGuard AI - Supervised ML Model
# Hybrid-feature Random Forest
# ============================================================

import os
import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Support models inside backend/models, parent ../models, or via MODEL_DIR env var
if os.environ.get("MODEL_DIR") and os.path.exists(os.environ.get("MODEL_DIR")):
    MODEL_DIR = os.environ.get("MODEL_DIR")
elif os.path.exists(os.path.join(BASE_DIR, "models", "anomaly_model.pkl")):
    MODEL_DIR = os.path.join(BASE_DIR, "models")
elif os.path.exists(os.path.join(BASE_DIR, "..", "models", "anomaly_model.pkl")):
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")
elif os.path.exists(os.path.join(BASE_DIR, "models")):
    MODEL_DIR = os.path.join(BASE_DIR, "models")
else:
    MODEL_DIR = os.path.join(BASE_DIR, "..", "models")

MODEL_FILE = os.path.join(MODEL_DIR, "anomaly_model.pkl")


FEATURES = [
    "change_percent",
    "change_rate",
    "historical_score",
    "spatial_score",
    "regional_agreement",
    "persistence_score",
    "multivariate_score",
    "time_gap_seconds",
]


# ============================================================
# TRAIN
# ============================================================

def train_model():

    os.makedirs(MODEL_DIR, exist_ok=True)

    rng = np.random.default_rng(42)

    n_normal = 2500
    n_anomaly = 1000

    # --------------------------------------------------------
    # NORMAL DATA
    # --------------------------------------------------------

    normal = pd.DataFrame({

        "change_percent":
            rng.uniform(0, 5, n_normal),

        "change_rate":
            rng.uniform(0, 0.15, n_normal),

        "historical_score":
            rng.uniform(0, 0.35, n_normal),

        "spatial_score":
            rng.uniform(0, 0.35, n_normal),

        "regional_agreement":
            rng.uniform(0.45, 1.0, n_normal),

        "persistence_score":
            rng.uniform(0, 0.25, n_normal),

        "multivariate_score":
            rng.uniform(0, 0.25, n_normal),

        "time_gap_seconds":
            rng.uniform(30, 3600, n_normal),

    })

    # --------------------------------------------------------
    # ANOMALY DATA
    # --------------------------------------------------------

    anomaly = pd.DataFrame({

        "change_percent":
            rng.uniform(7, 60, n_anomaly),

        "change_rate":
            rng.uniform(0.15, 5.0, n_anomaly),

        "historical_score":
            rng.uniform(0.45, 1.0, n_anomaly),

        "spatial_score":
            rng.uniform(0.45, 1.0, n_anomaly),

        "regional_agreement":
            rng.uniform(0, 0.45, n_anomaly),

        "persistence_score":
            rng.uniform(0.25, 1.0, n_anomaly),

        "multivariate_score":
            rng.uniform(0, 0.8, n_anomaly),

        "time_gap_seconds":
            rng.uniform(1, 1200, n_anomaly),

    })

    X = pd.concat(
        [normal, anomaly],
        ignore_index=True
    )

    y = np.concatenate([
        np.zeros(n_normal, dtype=int),
        np.ones(n_anomaly, dtype=int)
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X[FEATURES],
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=250,
        max_depth=10,
        min_samples_leaf=4,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    )

    model.fit(
        X_train,
        y_train
    )

    joblib.dump(
        model,
        MODEL_FILE
    )

    print("Supervised anomaly model trained.")
    print(f"Model saved: {MODEL_FILE}")

    return model


# ============================================================
# LOAD
# ============================================================

_MODEL = None


def load_model():

    global _MODEL

    # Already loaded.
    if _MODEL is not None:
        return _MODEL

    # Existing model.
    if os.path.exists(MODEL_FILE):

        try:

            model = joblib.load(
                MODEL_FILE
            )

            # Validate the model.
            if not hasattr(
                model,
                "predict_proba"
            ):
                raise ValueError(
                    "Invalid model"
                )

            _MODEL = model

            return _MODEL

        except Exception as e:

            print(
                "Existing anomaly model is "
                "invalid/corrupted."
            )

            print(
                "Creating a new model..."
            )

            try:
                os.remove(MODEL_FILE)
            except Exception:
                pass

    # Train fresh model.
    _MODEL = train_model()

    return _MODEL


# ============================================================
# PREDICTION
# ============================================================

def predict_probability(features):

    try:

        model = load_model()

        X = pd.DataFrame(
            [features],
            columns=FEATURES
        )

        probability = model.predict_proba(X)

        return float(
            probability[0][1]
        )

    except Exception as e:

        print(
            f"ML prediction skipped: {e}"
        )

        return 0.0



# ============================================================
# FAULT-TYPE CLASSIFIER (SUPPORTING ML ONLY)
# ============================================================

FAULT_MODEL_FILE = os.path.join(MODEL_DIR, "fault_type_model.pkl")
_FAULT_MODEL = None
FAULT_CLASSES = [
    "NORMAL",
    "WEATHER_EVENT",
    "SUDDEN_SPIKE",
    "CALIBRATION_DRIFT",
    "ISOLATED_SENSOR_FAULT",
]


def train_fault_type_model():
    """Train a lightweight supporting 5-class classifier on synthetic feature regimes.

    This classifier is evidence only; anomaly.py remains authoritative for the final status
    and rule-based fault taxonomy.
    """
    os.makedirs(MODEL_DIR, exist_ok=True)
    rng = np.random.default_rng(123)
    per_class = 900

    def frame(**kwargs):
        return pd.DataFrame({k: np.asarray(v) for k, v in kwargs.items()})

    normal = frame(
        change_percent=rng.uniform(0, 5, per_class),
        change_rate=rng.uniform(0, 0.15, per_class),
        historical_score=rng.uniform(0, 0.35, per_class),
        spatial_score=rng.uniform(0, 0.35, per_class),
        regional_agreement=rng.uniform(0.55, 1.0, per_class),
        persistence_score=rng.uniform(0, 0.30, per_class),
        multivariate_score=rng.uniform(0, 0.30, per_class),
        time_gap_seconds=rng.uniform(30, 3600, per_class),
    )
    weather = frame(
        change_percent=rng.uniform(7, 25, per_class),
        change_rate=rng.uniform(0.08, 1.5, per_class),
        historical_score=rng.uniform(0.10, 0.65, per_class),
        spatial_score=rng.uniform(0.00, 0.35, per_class),
        regional_agreement=rng.uniform(0.70, 1.0, per_class),
        persistence_score=rng.uniform(0.15, 0.75, per_class),
        multivariate_score=rng.uniform(0.10, 0.65, per_class),
        time_gap_seconds=rng.uniform(60, 3600, per_class),
    )
    spike = frame(
        change_percent=rng.uniform(8, 60, per_class),
        change_rate=rng.uniform(0.25, 5.0, per_class),
        historical_score=rng.uniform(0.45, 1.0, per_class),
        spatial_score=rng.uniform(0.50, 1.0, per_class),
        regional_agreement=rng.uniform(0.00, 0.35, per_class),
        persistence_score=rng.uniform(0.10, 0.80, per_class),
        multivariate_score=rng.uniform(0.05, 0.70, per_class),
        time_gap_seconds=rng.uniform(1, 300, per_class),
    )
    drift = frame(
        change_percent=rng.uniform(3, 12, per_class),
        change_rate=rng.uniform(0.02, 0.20, per_class),
        historical_score=rng.uniform(0.50, 1.0, per_class),
        spatial_score=rng.uniform(0.25, 0.75, per_class),
        regional_agreement=rng.uniform(0.15, 0.55, per_class),
        persistence_score=rng.uniform(0.70, 1.0, per_class),
        multivariate_score=rng.uniform(0.20, 0.70, per_class),
        time_gap_seconds=rng.uniform(300, 3600, per_class),
    )
    isolated = frame(
        change_percent=rng.uniform(5, 30, per_class),
        change_rate=rng.uniform(0.10, 2.5, per_class),
        historical_score=rng.uniform(0.45, 1.0, per_class),
        spatial_score=rng.uniform(0.45, 1.0, per_class),
        regional_agreement=rng.uniform(0.00, 0.35, per_class),
        persistence_score=rng.uniform(0.25, 0.80, per_class),
        multivariate_score=rng.uniform(0.20, 0.80, per_class),
        time_gap_seconds=rng.uniform(60, 1800, per_class),
    )

    X = pd.concat([normal, weather, spike, drift, isolated], ignore_index=True)
    y = np.array(
        ([FAULT_CLASSES[0]] * per_class)
        + ([FAULT_CLASSES[1]] * per_class)
        + ([FAULT_CLASSES[2]] * per_class)
        + ([FAULT_CLASSES[3]] * per_class)
        + ([FAULT_CLASSES[4]] * per_class)
    )

    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=123,
        n_jobs=-1,
    )
    model.fit(X[FEATURES], y)
    joblib.dump(model, FAULT_MODEL_FILE)
    return model


def load_fault_type_model():
    global _FAULT_MODEL
    if _FAULT_MODEL is not None:
        return _FAULT_MODEL
    if os.path.exists(FAULT_MODEL_FILE):
        try:
            model = joblib.load(FAULT_MODEL_FILE)
            if not hasattr(model, "predict_proba"):
                raise ValueError("Invalid fault-type model")
            _FAULT_MODEL = model
            return _FAULT_MODEL
        except Exception:
            try:
                os.remove(FAULT_MODEL_FILE)
            except Exception:
                pass
    _FAULT_MODEL = train_fault_type_model()
    return _FAULT_MODEL


def predict_fault_type(features):
    """Return supporting ML fault-type evidence for anomaly.py."""
    try:
        model = load_fault_type_model()
        X = pd.DataFrame([features], columns=FEATURES)
        probabilities = model.predict_proba(X)[0]
        classes = list(model.classes_)
        best_idx = int(np.argmax(probabilities))
        return {
            "class": str(classes[best_idx]),
            "confidence": float(probabilities[best_idx]),
            "probabilities": {
                str(cls): float(prob)
                for cls, prob in zip(classes, probabilities)
            },
        }
    except Exception as exc:
        print(f"Fault-type ML prediction skipped: {exc}")
        return None

if __name__ == "__main__":

    train_model()
    