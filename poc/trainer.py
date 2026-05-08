"""Teachable Machine-style action trainer for MotionIQ POC.

Records labelled feature vectors and trains a KNN classifier on demand.
"""

from __future__ import annotations

import numpy as np

try:
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


class ActionTrainer:
    """Collect feature samples per action class and train a live classifier."""

    def __init__(self, n_neighbors: int = 5) -> None:
        self.samples: dict[str, list[np.ndarray]] = {}
        self.risk_levels: dict[str, float] = {}
        self.model: Pipeline | None = None
        self.is_trained = False
        self.class_names: list[str] = []
        self.n_neighbors = n_neighbors
        self.last_train_result: dict = {}

    # ── Sample management ────────────────────────────────────────────────────

    def add_sample(self, class_name: str, feature_vector: np.ndarray) -> None:
        self.samples.setdefault(class_name, []).append(
            np.asarray(feature_vector, dtype=np.float32).copy()
        )

    def set_risk(self, class_name: str, risk: float) -> None:
        self.risk_levels[class_name] = float(np.clip(risk, 0.0, 1.0))

    def sample_counts(self) -> dict[str, int]:
        return {k: len(v) for k, v in self.samples.items()}

    def total_samples(self) -> int:
        return sum(len(v) for v in self.samples.values())

    def can_train(self) -> bool:
        if not SKLEARN_AVAILABLE:
            return False
        return len(self.samples) >= 2 and all(
            len(v) >= 5 for v in self.samples.values()
        )

    def clear_class(self, class_name: str) -> None:
        self.samples.pop(class_name, None)
        self.risk_levels.pop(class_name, None)
        self.is_trained = False
        self.model = None

    def clear_all(self) -> None:
        self.samples.clear()
        self.risk_levels.clear()
        self.model = None
        self.is_trained = False
        self.last_train_result = {}

    # ── Training ─────────────────────────────────────────────────────────────

    def train(self) -> dict:
        if not self.can_train():
            return {"error": "Need ≥ 2 classes with ≥ 5 samples each."}

        X, y = [], []
        for cls, vectors in self.samples.items():
            for vec in vectors:
                X.append(vec)
                y.append(cls)

        X_arr = np.array(X, dtype=np.float32)
        y_arr = np.array(y)
        self.class_names = list(self.samples.keys())

        k = min(self.n_neighbors, min(len(v) for v in self.samples.values()))
        self.model = Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(n_neighbors=k, metric="euclidean")),
        ])
        self.model.fit(X_arr, y_arr)
        self.is_trained = True

        train_acc = float(self.model.score(X_arr, y_arr))
        self.last_train_result = {
            "accuracy": train_acc,
            "n_classes": len(self.class_names),
            "n_samples": len(X_arr),
            "classes": self.class_names,
        }
        return self.last_train_result

    # ── Inference ─────────────────────────────────────────────────────────────

    def predict(self, feature_vector: np.ndarray) -> tuple[str, float, float]:
        """Return (action_class, confidence, risk_score)."""
        if not self.is_trained or self.model is None:
            return "untrained", 0.0, 0.0

        x = np.asarray(feature_vector, dtype=np.float32).reshape(1, -1)
        pred = str(self.model.predict(x)[0])

        try:
            proba = self.model.predict_proba(x)[0]
            conf = float(np.max(proba))
        except Exception:
            conf = 1.0

        risk = self.risk_levels.get(pred, 0.3)
        return pred, conf, risk
