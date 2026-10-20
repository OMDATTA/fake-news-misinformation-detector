"""Model comparison, leakage check, topic hold-out and class-imbalance study."""
import json
import time

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.naive_bayes import ComplementNB
from sklearn.svm import LinearSVC

from _common import DATA, FIG, REPORTS, SEED
from _plot import DARK_RED, GREY, PAPER, RED, plt
from fnd.features import LingScaler, hybrid, ling_matrix, make_tfidf
from fnd.metrics import scores

rng = np.random.default_rng(SEED)


def main():
    df = pd.read_parquet(DATA / "processed.parquet")
    df["topic"] = pd.read_parquet(DATA / "topics.parquet")["topic"].values
    tr, cal, te = (df[df.split == s] for s in ("train", "cal", "test"))
    ytr, ycal, yte = tr.y.values, cal.y.values, te.y.values

    t0 = time.time()
    vec = make_tfidf()
    Xtr = vec.fit_transform(tr.clean); Xcal = vec.transform(cal.clean); Xte = vec.transform(te.clean)
    Ltr, Lcal, Lte = ling_matrix(tr.clean), ling_matrix(cal.clean), ling_matrix(te.clean)
    sc = LingScaler().fit(Ltr)
    Ztr, Zcal, Zte = sc.transform(Ltr), sc.transform(Lcal), sc.transform(Lte)
    print(f"features in {time.time()-t0:.0f}s, vocab={len(vec.vocabulary_)}")

    # ---- small hyper-parameter search for the hybrid on the calibration split
    best = None
    for C in (2.0, 8.0, 32.0):
        for w in (0.5, 1.0):
            m = LogisticRegression(C=C, max_iter=4000, class_weight="balanced")
            m.fit(hybrid(Xtr, Ztr, w), ytr)
            f = f1_score(ycal, m.predict(hybrid(Xcal, Zcal, w)), average="macro")
            print(f"  hybrid C={C} w={w} cal macroF1={f:.4f}")
            if best is None or f > best[0]:
                best = (f, C, w)
    _, C_best, W_best = best
    (REPORTS / "hparams.json").write_text(json.dumps({"C": C_best, "ling_weight": W_best}))

    results = {}

    def add(name, family, p):
        results[name] = {"family": family, **scores(yte, p)}
        print(f"{name:45s} acc={results[name]['accuracy']:.4f} auc={results[name]['roc_auc']:.4f}")

    add("Majority class", "Baseline",
        DummyClassifier(strategy="prior").fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    add("Linguistic features + LogReg", "Linguistic",
        LogisticRegression(max_iter=3000, class_weight="balanced").fit(Ztr, ytr).predict_proba(Zte)[:, 1])
    add("Linguistic features + Gradient Boosting", "Linguistic",
        HistGradientBoostingClassifier(random_state=SEED).fit(Ltr, ytr).predict_proba(Lte)[:, 1])
    add("TF-IDF + Complement Naive Bayes", "TF-IDF",
        ComplementNB(alpha=0.3).fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    add("TF-IDF + Logistic Regression", "TF-IDF",
        LogisticRegression(C=C_best, max_iter=4000, class_weight="balanced").fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    svm = CalibratedClassifierCV(LinearSVC(C=0.5, class_weight="balanced"), cv=3, method="sigmoid")
    add("TF-IDF + Linear SVM (Platt)", "TF-IDF", svm.fit(Xtr, ytr).predict_proba(Xte)[:, 1])
    hyb = LogisticRegression(C=C_best, max_iter=4000, class_weight="balanced")
    hyb.fit(hybrid(Xtr, Ztr, W_best), ytr)
    add("TF-IDF + linguistic + LogReg (deployed)", "Hybrid",
        hyb.predict_proba(hybrid(Xte, Zte, W_best))[:, 1])

    # ---- leakage check: same model on RAW text (publisher artifacts kept)
    vec_raw = make_tfidf()
    Rtr = vec_raw.fit_transform(tr.raw); Rte = vec_raw.transform(te.raw)
    p_raw = LogisticRegression(C=C_best, max_iter=4000, class_weight="balanced").fit(Rtr, ytr).predict_proba(Rte)[:, 1]
    leakage = {"raw_text": scores(yte, p_raw),
               "scrubbed_text": {k: results["TF-IDF + Logistic Regression"][k] for k in scores(yte, p_raw)}}
    raw_coef = LogisticRegression(C=C_best, max_iter=4000, class_weight="balanced").fit(Rtr, ytr).coef_[0]
    names = vec_raw.get_feature_names_out()
    leakage["raw_top_reliable_terms"] = names[np.argsort(raw_coef)[:12]].tolist()
    leakage["raw_top_misleading_terms"] = names[np.argsort(raw_coef)[::-1][:12]].tolist()

    # ---- topic hold-out generalisation (hybrid model)
    topic_info = {t["topic"]: t for t in json.loads((REPORTS / "source_topic.json").read_text())["topics"]}
    holdout = []
    for k in sorted(df.topic.unique()):
        mtr = tr.topic.values != k
        mte = te.topic.values == k
        if len(np.unique(yte[mte])) < 2:
            continue
        m = LogisticRegression(C=C_best, max_iter=4000, class_weight="balanced")
        m.fit(hybrid(Xtr[mtr], Ztr[mtr], W_best), ytr[mtr])
        p = m.predict_proba(hybrid(Xte[mte], Zte[mte], W_best))[:, 1]
        in_dist = hyb.predict_proba(hybrid(Xte[mte], Zte[mte], W_best))[:, 1]
        s_out, s_in = scores(yte[mte], p), scores(yte[mte], in_dist)
        holdout.append({"topic": int(k), "label": ", ".join(topic_info[k]["top_words"][:4]),
                        "n_test": int(mte.sum()),
                        "acc_seen": s_in["accuracy"], "acc_held_out": s_out["accuracy"],
                        "auc_held_out": s_out["roc_auc"], "ece_held_out": s_out["ece"]})
        print("holdout", holdout[-1])

    # ---- class imbalance: 10% misleading prevalence in train AND test
    def subsample(idx_df, prev):
        pos = idx_df[idx_df.y == 1]; neg = idx_df[idx_df.y == 0]
        n_pos = int(len(neg) * prev / (1 - prev))
        return pd.concat([neg, pos.sample(min(n_pos, len(pos)), random_state=SEED)]).index

    prev = 0.10
    itr = tr.index.get_indexer(subsample(tr, prev))
    ical = cal.index.get_indexer(subsample(cal, prev))
    ite = te.index.get_indexer(subsample(te, prev))
    Htr, Hcal, Hte = hybrid(Xtr, Ztr, W_best), hybrid(Xcal, Zcal, W_best), hybrid(Xte, Zte, W_best)
    imb = {}
    m_plain = LogisticRegression(C=C_best, max_iter=4000).fit(Htr[itr], ytr[itr])
    m_bal = LogisticRegression(C=C_best, max_iter=4000, class_weight="balanced").fit(Htr[itr], ytr[itr])
    p_plain, p_bal = m_plain.predict_proba(Hte[ite])[:, 1], m_bal.predict_proba(Hte[ite])[:, 1]
    imb["Unweighted, threshold 0.5"] = scores(yte[ite], p_plain)
    imb["Class-weighted (balanced), threshold 0.5"] = scores(yte[ite], p_bal)
    pc = m_plain.predict_proba(Hcal[ical])[:, 1]
    ths = np.linspace(0.05, 0.95, 91)
    t_best = float(ths[np.argmax([f1_score(ycal[ical], pc >= t) for t in ths])])
    imb[f"Unweighted, F1-tuned threshold ({t_best:.2f})"] = scores(yte[ite], p_plain, t_best)
    imbalance = {"prevalence": prev, "n_train": int(len(itr)), "n_test": int(len(ite)),
                 "results": imb}

    out = {"hparams": {"C": C_best, "ling_weight": W_best}, "n_train": int(len(tr)),
           "n_test": int(len(te)), "results": results, "leakage": leakage,
           "topic_holdout": holdout, "imbalance": imbalance}
    (REPORTS / "model_comparison.json").write_text(json.dumps(out, indent=2))

    # ---- figures
    r = pd.DataFrame(results).T.sort_values("macro_f1")
    fig, ax = plt.subplots(figsize=(8, 4.2))
    colors = [RED if "deployed" in n else (DARK_RED if f != "Baseline" else GREY) for n, f in zip(r.index, r.family)]
    ax.barh(r.index, r.macro_f1, color=colors)
    for i, v in enumerate(r.macro_f1):
        ax.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=8, color=PAPER)
    ax.set_xlim(0, 1.08); ax.set_xlabel("Macro-F1 on held-out test set (scrubbed text)")
    ax.set_title("Model comparison"); fig.tight_layout()
    fig.savefig(FIG / "model_comparison.png", dpi=160); plt.close(fig)

    h = pd.DataFrame(holdout)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    y = np.arange(len(h))
    ax.barh(y + 0.2, h.acc_seen, 0.4, color=GREY, label="Topic seen in training")
    ax.barh(y - 0.2, h.acc_held_out, 0.4, color=RED, label="Topic held out")
    ax.set_yticks(y, h.label); ax.set_xlim(0.5, 1.0); ax.set_xlabel("Accuracy")
    ax.set_title("Generalisation to unseen topics"); ax.legend(frameon=False, loc="lower left")
    fig.tight_layout(); fig.savefig(FIG / "topic_holdout.png", dpi=160); plt.close(fig)


if __name__ == "__main__":
    main()
