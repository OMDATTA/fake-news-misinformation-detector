"""Transformer sentence-embedding baseline (all-MiniLM-L6-v2) vs TF-IDF.

Encoding 38k long articles on CPU is slow, so we embed a stratified 10k-article
training subsample and the full test split. For a fair comparison the TF-IDF
and linguistic models are re-trained on the *same* 10k subsample.
"""
import json
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from _common import DATA, REPORTS, SEED
from fnd.features import LingScaler, hybrid, ling_matrix, make_tfidf
from fnd.metrics import scores

MODEL = "sentence-transformers/all-MiniLM-L6-v2"
N_TRAIN = 10000


def embed(texts, cache):
    if cache.exists():
        return np.load(cache)
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(MODEL)
    m.max_seq_length = 256   # first ~200 words: headline + lede
    t = time.time()
    E = m.encode(list(texts), batch_size=32, show_progress_bar=False, normalize_embeddings=True)
    print(f"encoded {len(texts)} in {time.time()-t:.0f}s")
    np.save(cache, E)
    return E


def main():
    df = pd.read_parquet(DATA / "processed.parquet")
    trn = df[df.split == "train"]
    tr = pd.concat([trn[trn.y == c].sample(N_TRAIN // 2, random_state=SEED) for c in (0, 1)])
    te = df[df.split == "test"]
    Etr = embed(tr.clean, DATA / "emb_train10k.npy")
    Ete = embed(te.clean, DATA / "emb_test.npy")
    ytr, yte = tr.y.values, te.y.values
    res = {}

    lr = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    res["MiniLM embeddings + LogReg"] = scores(yte, lr.fit(Etr, ytr).predict_proba(Ete)[:, 1])

    vec = make_tfidf(); Xtr = vec.fit_transform(tr.clean); Xte = vec.transform(te.clean)
    lr2 = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced")
    res["TF-IDF + LogReg (same 10k)"] = scores(yte, lr2.fit(Xtr, ytr).predict_proba(Xte)[:, 1])

    sc = LingScaler().fit(ling_matrix(tr.clean))
    Ztr, Zte = sc.transform(ling_matrix(tr.clean)), sc.transform(ling_matrix(te.clean))
    Htr = np.hstack([Etr, Ztr * 0.2]); Hte = np.hstack([Ete, Zte * 0.2])
    lr3 = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced")
    res["MiniLM + linguistic + LogReg"] = scores(yte, lr3.fit(Htr, ytr).predict_proba(Hte)[:, 1])

    out = {"model": MODEL, "n_train": int(len(tr)), "n_test": int(len(te)),
           "max_seq_length": 256, "results": res}
    (REPORTS / "embeddings.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
