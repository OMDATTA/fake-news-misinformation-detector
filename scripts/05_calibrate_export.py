"""Train the deployed model, calibrate it, and export it to JSON for the website.

Deployed model:  [TF-IDF(1-2 grams, 25k)  |  15 standardised linguistic features]
                 -> class-weighted logistic regression
                 -> Platt scaling fitted on a held-out calibration split.
"""
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from _common import DATA, FIG, REPORTS, WEB_DATA
from _plot import GREY, PAPER, RED, plt
from fnd.features import LING_CLIP, LingScaler, hybrid, ling_matrix, make_tfidf
from fnd.metrics import reliability, scores
from fnd.text import LING_FEATURES


def sig(v, n=5):
    return float(f"{v:.{n}g}")


def main():
    hp = json.loads((REPORTS / "hparams.json").read_text())
    C, W = hp["C"], hp["ling_weight"]
    df = pd.read_parquet(DATA / "processed.parquet")
    tr, cal, te = (df[df.split == s] for s in ("train", "cal", "test"))

    vec = make_tfidf()
    Xtr = vec.fit_transform(tr.clean)
    sc = LingScaler().fit(ling_matrix(tr.clean))
    H = lambda d: hybrid(vec.transform(d.clean), sc.transform(ling_matrix(d.clean)), W)
    Htr, Hcal, Hte = hybrid(Xtr, sc.transform(ling_matrix(tr.clean)), W), H(cal), H(te)

    clf = LogisticRegression(C=C, max_iter=5000, class_weight="balanced").fit(Htr, tr.y)
    z_cal, z_te = clf.decision_function(Hcal), clf.decision_function(Hte)

    # Platt scaling: p = sigmoid(a*z + b), fitted on the calibration split
    platt = LogisticRegression(C=1e6, max_iter=1000).fit(z_cal.reshape(-1, 1), cal.y)
    a, b = float(platt.coef_[0, 0]), float(platt.intercept_[0])
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.001, y_max=0.999).fit(z_cal, cal.y)

    p_raw = 1 / (1 + np.exp(-z_te))
    p_platt = 1 / (1 + np.exp(-(a * z_te + b)))
    p_iso = iso.predict(z_te)
    cal_report = {
        "method": "platt", "a": a, "b": b,
        "uncalibrated": scores(te.y, p_raw), "platt": scores(te.y, p_platt),
        "isotonic": scores(te.y, p_iso),
        "reliability": {"uncalibrated": reliability(te.y, p_raw, 10),
                        "platt": reliability(te.y, p_platt, 10)},
    }
    (REPORTS / "calibration.json").write_text(json.dumps(cal_report, indent=2))
    print("uncal", {k: round(v, 4) for k, v in cal_report["uncalibrated"].items() if k in ("brier", "ece")})
    print("platt", {k: round(v, 4) for k, v in cal_report["platt"].items() if k in ("brier", "ece")})
    print("iso  ", {k: round(v, 4) for k, v in cal_report["isotonic"].items() if k in ("brier", "ece")})

    fig, ax = plt.subplots(figsize=(5.2, 5))
    ax.plot([0, 1], [0, 1], ls="--", color=GREY, lw=1, label="Perfect calibration")
    for key, col, lab in (("uncalibrated", GREY, "Raw logistic output"), ("platt", RED, "Platt-calibrated")):
        rb = cal_report["reliability"][key]
        ax.plot([r["mean_pred"] for r in rb], [r["frac_pos"] for r in rb], marker="o", color=col,
                lw=2 if key == "platt" else 1.2, label=lab)
    ax.set_xlabel("Predicted P(misleading-like)"); ax.set_ylabel("Observed fraction misleading")
    ax.set_title("Reliability diagram (test set)"); ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(FIG / "calibration.png", dpi=160); plt.close(fig)

    # ---- export for the browser ---------------------------------------------
    n_t = len(vec.vocabulary_)
    terms = [None] * n_t
    for t, i in vec.vocabulary_.items():
        terms[i] = t
    coef = clf.coef_[0]
    model = {
        "version": "1.0.0",
        "trained_on": "ISOT Fake News Dataset (de-duplicated, source-scrubbed)",
        "positive_class": "misleading-like",
        "tfidf": {"ngram_range": [1, 2], "sublinear_tf": True, "norm": "l2",
                  "terms": terms, "idf": [sig(v) for v in vec.idf_],
                  "coef": [sig(v) for v in coef[:n_t]]},
        "ling": {"names": [k for k, _ in LING_FEATURES], "labels": [l for _, l in LING_FEATURES],
                 "mean": [sig(v, 7) for v in sc.mean_], "scale": [sig(v, 7) for v in sc.scale_],
                 "clip": LING_CLIP, "weight": W,
                 "coef": [sig(v) for v in coef[n_t:]]},
        "intercept": float(clf.intercept_[0]),
        "calibration": {"method": "platt", "a": a, "b": b},
        "train_vocab_coverage_p05": None,
    }
    # typical vocabulary coverage (share of unigram tokens in vocab) for OOD warnings
    from fnd.text import tokenize
    vocab = vec.vocabulary_
    cov = [np.mean([t in vocab for t in tokenize(x)]) for x in te.clean.sample(2000, random_state=0)]
    model["train_vocab_coverage_p05"] = float(np.percentile(cov, 5))
    (WEB_DATA / "model.json").write_text(json.dumps(model, separators=(",", ":")))

    # parity fixtures: python probabilities for a handful of test docs
    fx = te.sample(25, random_state=7)
    zf = clf.decision_function(H(fx))
    fixtures = [{"text": r.raw, "p": float(1 / (1 + np.exp(-(a * z + b)))), "logit": float(z)}
                for r, z in zip(fx.itertuples(), zf)]
    (ROOT_T := REPORTS.parent / "tests" / "fixtures.json").write_text(json.dumps(fixtures))
    print("exported", (WEB_DATA / "model.json").stat().st_size // 1024, "KB")

    # top global features for the website
    order = np.argsort(coef[:n_t])
    top = {"misleading": [{"term": terms[i], "coef": sig(coef[i], 3)} for i in order[::-1][:20]],
           "reliable": [{"term": terms[i], "coef": sig(coef[i], 3)} for i in order[:20]],
           "linguistic": [{"name": l, "coef": sig(c, 3)} for (_, l), c in zip(LING_FEATURES, coef[n_t:])]}
    (REPORTS / "top_features.json").write_text(json.dumps(top, indent=2))


if __name__ == "__main__":
    main()
