"""Feature builders shared by the training scripts."""
from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from .text import TOKEN_PATTERN, for_tfidf, linguistic_features

LING_CLIP = 5.0


def make_tfidf(max_features: int = 25000) -> TfidfVectorizer:
    # Settings intentionally simple so they can be re-implemented in JS.
    return TfidfVectorizer(
        preprocessor=for_tfidf,
        token_pattern=TOKEN_PATTERN,
        lowercase=False,          # handled by for_tfidf
        ngram_range=(1, 2),
        min_df=3,
        max_df=0.9,
        sublinear_tf=True,
        max_features=max_features,
        dtype=np.float64,
    )


def ling_matrix(clean_texts) -> np.ndarray:
    return np.array([linguistic_features(t) for t in clean_texts], dtype=np.float64)


class LingScaler:
    """StandardScaler + clipping, exportable as plain mean/std arrays."""

    def fit(self, X):
        self.sc = StandardScaler().fit(X)
        self.mean_ = self.sc.mean_
        self.scale_ = self.sc.scale_
        return self

    def transform(self, X):
        return np.clip((X - self.mean_) / self.scale_, -LING_CLIP, LING_CLIP)


def hybrid(X_tfidf, Z_ling, ling_weight: float = 1.0):
    return sparse.hstack([X_tfidf, sparse.csr_matrix(Z_ling * ling_weight)]).tocsr()
