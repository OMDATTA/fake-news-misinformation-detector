"""Out-of-distribution check on WELFake (articles not present in ISOT).

ISOT is stylistically easy: reliable = Reuters wire copy, misleading = a set of
flagged websites. WELFake mixes several other corpora (Kaggle, McIntire,
BuzzFeed). Testing the ISOT-trained model there shows how much of the
in-distribution score transfers. Spoiler: much less - which is exactly why the
app presents its output as "resemblance", not as a verdict.
"""
import json

import numpy as np
import pandas as pd
from datasets import load_dataset
from sklearn.linear_model import LogisticRegression

from _common import DATA, REPORTS, SEED
from fnd.features import LingScaler, hybrid, ling_matrix, make_tfidf
from fnd.metrics import reliability, scores
from fnd.text import scrub

N = 4000


def key(s):
    return s.str.lower().str.replace(r"[^a-z0-9]", "", regex=True).str[:400]


def main():
    hp = json.loads((REPORTS / "hparams.json").read_text())
    cal_p = json.loads((REPORTS / "calibration.json").read_text())
    df = pd.read_parquet(DATA / "processed.parquet")
    tr = df[df.split == "train"]

    w = load_dataset("davanstrien/WELFake", split="train").to_pandas()
    w = w.dropna(subset=["text"])
    # NB: despite the dataset card, label 1 = fake here (verified by inspecting titles).
    w["y"] = (w["label"] == 1).astype(int)
    w["raw"] = (w["title"].fillna("") + ". " + w["text"].fillna("")).str.strip()
    w["clean"] = w["raw"].map(scrub)
    w = w[w["clean"].str.split().str.len() >= 30]
    isot_keys = set(key(df["clean"]))
    isot_titles = set(df["title"].str.strip().str.lower())
    w = w[~key(w["clean"]).isin(isot_keys) & ~w["title"].fillna("").str.strip().str.lower().isin(isot_titles)]
    w = w.drop_duplicates(subset=["clean"])
    w = pd.concat([w[w.y == c].sample(N // 2, random_state=SEED) for c in (0, 1)])

    w[["title", "raw", "clean", "y"]].to_parquet(DATA / "welfake_ood.parquet", index=False)

    vec = make_tfidf(); Xtr = vec.fit_transform(tr.clean)
    sc = LingScaler().fit(ling_matrix(tr.clean))
    W = hp["ling_weight"]
    clf = LogisticRegression(C=hp["C"], max_iter=5000, class_weight="balanced")
    clf.fit(hybrid(Xtr, sc.transform(ling_matrix(tr.clean)), W), tr.y)
    z = clf.decision_function(hybrid(vec.transform(w.clean), sc.transform(ling_matrix(w.clean)), W))
    p = 1 / (1 + np.exp(-(cal_p["a"] * z + cal_p["b"])))

    out = {"dataset": "https://huggingface.co/datasets/davanstrien/WELFake", "n": int(len(w)),
           "note": "Sampled after removing any article whose text or title also appears in ISOT.",
           "scores": scores(w.y.values, p),
           "reliability": reliability(w.y.values, p, 10),
           "share_confident": float(np.mean((p > 0.9) | (p < 0.1)))}
    (REPORTS / "ood_welfake.json").write_text(json.dumps(out, indent=2))
    print(json.dumps({k: out[k] for k in ("n", "scores", "share_confident")}, indent=2))


if __name__ == "__main__":
    main()
