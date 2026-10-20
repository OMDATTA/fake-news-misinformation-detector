"""Do transformer embeddings transfer better out-of-distribution than TF-IDF?

Requires scripts 04 (embedding cache) and 06 (WELFake sample) to have run.
All three models are trained on the same 10k ISOT subsample.
"""
import json

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from _common import DATA, REPORTS, SEED
from fnd.features import LingScaler, ling_matrix, make_tfidf
from fnd.metrics import scores

import importlib
emb = importlib.import_module("04_transformer_embeddings")


def main():
    df = pd.read_parquet(DATA / "processed.parquet")
    trn = df[df.split == "train"]
    tr = pd.concat([trn[trn.y == c].sample(emb.N_TRAIN // 2, random_state=SEED) for c in (0, 1)])
    w = pd.read_parquet(DATA / "welfake_ood.parquet")
    Etr = emb.embed(tr.clean, DATA / "emb_train10k.npy")
    Ew = emb.embed(w.clean, DATA / "emb_welfake.npy")
    res = {}
    m = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(Etr, tr.y)
    res["MiniLM embeddings + LogReg"] = scores(w.y, m.predict_proba(Ew)[:, 1])
    vec = make_tfidf(); Xtr = vec.fit_transform(tr.clean)
    m2 = LogisticRegression(C=8.0, max_iter=3000, class_weight="balanced").fit(Xtr, tr.y)
    res["TF-IDF + LogReg (same 10k)"] = scores(w.y, m2.predict_proba(vec.transform(w.clean))[:, 1])
    sc = LingScaler().fit(ling_matrix(tr.clean))
    H = lambda E, d: np.hstack([E, sc.transform(ling_matrix(d.clean)) * 0.2])
    m3 = LogisticRegression(C=4.0, max_iter=3000, class_weight="balanced").fit(H(Etr, tr), tr.y)
    res["MiniLM + linguistic + LogReg"] = scores(w.y, m3.predict_proba(H(Ew, w))[:, 1])
    d = json.loads((REPORTS / "embeddings.json").read_text())
    d["ood_welfake"] = res
    (REPORTS / "embeddings.json").write_text(json.dumps(d, indent=2))
    for k, v in res.items():
        print(k, round(v["accuracy"], 4), round(v["roc_auc"], 4), round(v["ece"], 4))


if __name__ == "__main__":
    main()
