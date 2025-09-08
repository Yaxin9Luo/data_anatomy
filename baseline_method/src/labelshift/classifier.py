from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict

import numpy as np
from sklearn.pipeline import FeatureUnion
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix


def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    ez = np.exp(z)
    return ez / np.sum(ez, axis=axis, keepdims=True)


def _nll_from_logits(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    probs = _softmax(logits / T, axis=1)
    # Gather p(y_i|x_i)
    n = probs.shape[0]
    py = probs[np.arange(n), y]
    py = np.clip(py, 1e-12, 1.0)
    return float(-np.mean(np.log(py)))


def fit_temperature(logits: np.ndarray, y: np.ndarray, Ts: Optional[List[float]] = None) -> float:
    """
    Fit a single temperature T>0 on validation logits by minimizing NLL.
    Uses a simple grid over log-space for robustness (no SciPy dependency).
    """
    if Ts is None:
        # logT in [-2, 2] -> T in [~0.14, ~7.39]
        Ts = [float(np.exp(t)) for t in np.linspace(-2.0, 2.0, 41)]
    best_T = 1.0
    best_nll = float("inf")
    for T in Ts:
        nll = _nll_from_logits(logits, y, T)
        if nll < best_nll:
            best_nll = nll
            best_T = T
    return float(best_T)


@dataclass
class TfidfLogReg:
    vectorizer: FeatureUnion
    clf: LogisticRegression
    temperature: float = 1.0

    @classmethod
    def create(
        cls,
        max_word_features: int = 200_000,
        max_char_features: int = 200_000,
        C: float = 4.0,
        seed: int = 0,
        n_jobs: int = 4,
    ) -> "TfidfLogReg":
        word = TfidfVectorizer(ngram_range=(1, 2), max_features=max_word_features, lowercase=True)
        char = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), max_features=max_char_features, lowercase=True)
        vec = FeatureUnion([("word", word), ("char", char)])
        clf = LogisticRegression(
            max_iter=500,
            solver="saga",
            multi_class="multinomial",
            C=C,
            random_state=seed,
            n_jobs=n_jobs,
        )
        return cls(vectorizer=vec, clf=clf, temperature=1.0)

    def fit(self, X: List[str], y: List[int]) -> None:
        Xv = self.vectorizer.fit_transform(X)
        self.clf.fit(Xv, y)

    def logits(self, X: List[str]) -> np.ndarray:
        Xv = self.vectorizer.transform(X)
        # decision_function returns shape [n, K]
        return self.clf.decision_function(Xv)

    def predict_proba(self, X: List[str]) -> np.ndarray:
        logits = self.logits(X)
        probs = _softmax(logits / self.temperature, axis=1)
        return probs

    def predict(self, X: List[str]) -> np.ndarray:
        probs = self.predict_proba(X)
        return np.argmax(probs, axis=1)


def train_tfidf_classifier(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    seed: int = 0,
    n_jobs: int = 4,
) -> Tuple[TfidfLogReg, Dict[str, float], np.ndarray]:
    """
    Train TF-IDF + LogisticRegression classifier and calibrate temperature on validation set.
    Returns: (model, metrics, confusion_matrix)
    """
    model = TfidfLogReg.create(seed=seed, n_jobs=n_jobs)
    model.fit(train_texts, train_labels)

    # Pre-calibration metrics
    val_logits = model.logits(val_texts)
    y_val = np.array(val_labels)
    # Fit temperature
    T = fit_temperature(val_logits, y_val)
    model.temperature = T

    # Post-calibration evaluation
    val_probs = _softmax(val_logits / T, axis=1)
    y_pred = np.argmax(val_probs, axis=1)
    acc = accuracy_score(y_val, y_pred)
    cm = confusion_matrix(y_val, y_pred)
    # Normalize rows to get P(y_hat=j|y=i)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    C = cm / row_sums

    metrics = {
        "val_acc": float(acc),
        "temperature": float(T),
    }
    return model, metrics, C

