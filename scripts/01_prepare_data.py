"""Download ISOT (via Hugging Face mirror), clean, de-duplicate and split.

Label convention used everywhere in this repo:
    y = 1  -> resembles the *misleading* examples (ISOT "fake")
    y = 0  -> resembles the *reliable* examples   (ISOT "true", Reuters)
"""
import json

import pandas as pd
from datasets import load_dataset
from sklearn.model_selection import train_test_split

from _common import DATA, REPORTS, SEED
from fnd.text import scrub

HF_ID = "ErfanMoosaviMonazzah/fake-news-detection-dataset-English"


def main():
    ds = load_dataset(HF_ID)
    df = pd.concat([ds[s].to_pandas() for s in ds], ignore_index=True)
    df = df.drop(columns=[c for c in df.columns if c.startswith("Unnamed")])
    n_raw = len(df)
    df["y"] = (df["label"] == 0).astype(int)       # ISOT: 0 = fake, 1 = true
    df["raw"] = (df["title"].fillna("") + ". " + df["text"].fillna("")).str.strip()
    df["clean"] = df["raw"].map(scrub)
    df["n_words"] = df["clean"].str.split().str.len()
    df["date_parsed"] = pd.to_datetime(df["date"].str.strip(), errors="coerce", format="mixed")
    df["year"] = df["date_parsed"].dt.year

    # Exact + near-exact duplicates (same cleaned body) are common in ISOT.
    before = len(df)
    df = df[df["n_words"] >= 30]
    short_removed = before - len(df)
    df["dedup_key"] = df["clean"].str.lower().str.replace(r"[^a-z0-9]", "", regex=True).str[:400]
    dup_mask = df.duplicated("dedup_key")
    df = df[~dup_mask].reset_index(drop=True)

    train, rest = train_test_split(df, test_size=0.3, stratify=df["y"], random_state=SEED)
    cal, test = train_test_split(rest, test_size=2 / 3, stratify=rest["y"], random_state=SEED)
    df["split"] = "train"
    df.loc[cal.index, "split"] = "cal"
    df.loc[test.index, "split"] = "test"

    cols = ["title", "text", "subject", "date", "year", "y", "raw", "clean", "n_words", "split"]
    df[cols].to_parquet(DATA / "processed.parquet", index=False)

    summary = {
        "source": f"https://huggingface.co/datasets/{HF_ID}",
        "original": "ISOT Fake News Dataset (University of Victoria)",
        "rows_raw": int(n_raw),
        "rows_short_removed": int(short_removed),
        "rows_duplicates_removed": int(dup_mask.sum()),
        "rows_final": int(len(df)),
        "class_counts": {"reliable": int((df.y == 0).sum()), "misleading": int((df.y == 1).sum())},
        "split_counts": df["split"].value_counts().to_dict(),
    }
    (REPORTS / "data_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
