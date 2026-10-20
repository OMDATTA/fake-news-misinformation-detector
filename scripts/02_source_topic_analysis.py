"""Source-artifact audit and topic analysis.

1. Source artifacts: how far can a model get using ONLY publisher fingerprints
   (Reuters mentions, datelines, image credits, embedded tweets, curly quotes...)?
   If the answer is "very far", raw-text accuracy is inflated by leakage.
2. Topic analysis: NMF topics on cleaned text, class mix per topic, plus the
   dataset's own `subject` field and publication-year distribution.
"""
import json
import re

import numpy as np
import pandas as pd
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

from _common import DATA, FIG, REPORTS, SEED
from _plot import DARK_RED, GREY, PAPER, RED, plt

MARKERS = {
    "Mentions 'Reuters'": r"reuters",
    "Image credit ('Featured image', 'Photo by')": r"featured image|photo by|image via|getty images",
    "Embedded tweet / handle": r"pic\.twitter\.com|@\w{2,}",
    "Contains URL": r"https?://|www\.",
    "Curly quotes / apostrophes": "[\u2018\u2019\u201c\u201d]",
    "Title in ALL CAPS words (3+)": None,
    "Title tag like [VIDEO] / (IMAGES)": r"^[^.]*[\[(](?:video|image|images|tweet|tweets|watch)s?[\])]",
}


def title_caps(t):
    return sum(1 for w in re.findall(r"[A-Za-z]{2,}", t or "") if w.isupper()) >= 3


def main():
    df = pd.read_parquet(DATA / "processed.parquet")
    out = {}

    # ---- 1. Source artifacts ----------------------------------------------
    M = pd.DataFrame(index=df.index)
    for name, rx in MARKERS.items():
        if rx is None:
            M[name] = df.title.map(title_caps)
        else:
            flags = re.IGNORECASE if "Curly" not in name and "dateline" not in name else 0
            M[name] = df.raw.str.contains(rx, regex=True, flags=flags)
    M = M.astype(int)
    rates = []
    for name in MARKERS:
        rates.append({"marker": name,
                      "reliable_pct": float(M.loc[df.y == 0, name].mean() * 100),
                      "misleading_pct": float(M.loc[df.y == 1, name].mean() * 100)})
    acc = cross_val_score(LogisticRegression(max_iter=1000), M.values, df.y.values,
                          cv=5, scoring="accuracy").mean()
    out["source_markers"] = rates
    out["marker_only_cv_accuracy"] = float(acc)
    out["subject_crosstab"] = (pd.crosstab(df.subject, df.y.map({0: "reliable", 1: "misleading"}))
                               .reset_index().to_dict(orient="records"))
    out["subject_note"] = ("The `subject` column perfectly separates the classes (each subject "
                           "appears in only one class), so it is excluded from all models.")

    yr = df.dropna(subset=["year"]).groupby(["year", "y"]).size().unstack(fill_value=0)
    out["year_counts"] = [{"year": int(i), "reliable": int(r.get(0, 0)), "misleading": int(r.get(1, 0))}
                          for i, r in yr.iterrows()]

    # ---- 2. NMF topics ------------------------------------------------------
    tr = df[df.split == "train"]
    tv = TfidfVectorizer(stop_words="english", max_features=8000, min_df=5, max_df=0.5,
                         token_pattern=r"[a-z]{3,}", lowercase=True)
    X = tv.fit_transform(tr.clean)
    nmf = NMF(n_components=10, random_state=SEED, init="nndsvda", max_iter=400)
    W = nmf.fit_transform(X)
    vocab = np.array(tv.get_feature_names_out())
    Xall = tv.transform(df.clean)
    df["topic"] = nmf.transform(Xall).argmax(1)
    topics = []
    for k in range(nmf.n_components):
        top = vocab[np.argsort(nmf.components_[k])[::-1][:8]].tolist()
        sub = df[df.topic == k]
        topics.append({"topic": k, "top_words": top, "n": int(len(sub)),
                       "misleading_share": float(sub.y.mean())})
    out["topics"] = topics
    df[["topic"]].to_parquet(DATA / "topics.parquet")

    (REPORTS / "source_topic.json").write_text(json.dumps(out, indent=2))

    # ---- figures ------------------------------------------------------------
    r = pd.DataFrame(rates)
    fig, ax = plt.subplots(figsize=(8, 4.2))
    yy = np.arange(len(r))
    ax.barh(yy + 0.2, r.reliable_pct, 0.4, color=GREY, label="Reliable (Reuters)")
    ax.barh(yy - 0.2, r.misleading_pct, 0.4, color=RED, label="Misleading")
    ax.set_yticks(yy, r.marker); ax.invert_yaxis(); ax.set_xlabel("% of articles")
    ax.set_title(f"Publisher fingerprints alone reach {acc*100:.1f}% CV accuracy")
    ax.legend(frameon=False, loc="lower right"); fig.tight_layout()
    fig.savefig(FIG / "source_markers.png", dpi=160); plt.close(fig)

    t = pd.DataFrame(topics).sort_values("misleading_share")
    fig, ax = plt.subplots(figsize=(8, 4.6))
    labels = [", ".join(w[:4]) for w in t.top_words]
    ax.barh(labels, t.misleading_share * 100, color=[RED if s > .5 else DARK_RED for s in t.misleading_share])
    for i, (s, n) in enumerate(zip(t.misleading_share, t.n)):
        ax.text(s * 100 + 1, i, f"{s*100:.0f}%  (n={n:,})", va="center", color=PAPER, fontsize=8)
    ax.set_xlim(0, 115); ax.set_xlabel("% misleading within topic")
    ax.set_title("NMF topics: class mix varies strongly by topic"); fig.tight_layout()
    fig.savefig(FIG / "topics.png", dpi=160); plt.close(fig)
    print(json.dumps({k: out[k] for k in ("marker_only_cv_accuracy",)}, indent=2))
    for tpc in topics: print(tpc)
    print(pd.DataFrame(rates))


if __name__ == "__main__":
    main()
