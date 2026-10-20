"""Collect every report into one compact JSON consumed by the website."""
import json

from _common import REPORTS, WEB_DATA

KEEP = ("accuracy", "macro_f1", "roc_auc", "pr_auc", "brier", "ece", "precision", "recall", "f1",
        "balanced_accuracy")


def slim(d):
    return {k: round(v, 4) for k, v in d.items() if k in KEEP}


def main():
    r = lambda n: json.loads((REPORTS / n).read_text())
    data, st, mc, emb, cal, ood, top = (r("data_summary.json"), r("source_topic.json"),
                                        r("model_comparison.json"), r("embeddings.json"),
                                        r("calibration.json"), r("ood_welfake.json"),
                                        r("top_features.json"))
    site = {
        "data": data,
        "source_markers": st["source_markers"],
        "marker_only_cv_accuracy": round(st["marker_only_cv_accuracy"], 4),
        "topics": st["topics"],
        "year_counts": st["year_counts"],
        "models": [{"name": k, "family": v["family"], **slim(v)} for k, v in mc["results"].items()],
        "embeddings": {"model": emb["model"], "n_train": emb["n_train"],
                       "in_dist": {k: slim(v) for k, v in emb["results"].items()},
                       "ood": {k: slim(v) for k, v in emb.get("ood_welfake", {}).items()}},
        "leakage": {"raw": slim(mc["leakage"]["raw_text"]), "scrubbed": slim(mc["leakage"]["scrubbed_text"]),
                    "raw_top_reliable_terms": mc["leakage"]["raw_top_reliable_terms"],
                    "raw_top_misleading_terms": mc["leakage"]["raw_top_misleading_terms"]},
        "topic_holdout": mc["topic_holdout"],
        "imbalance": {**{k: v for k, v in mc["imbalance"].items() if k != "results"},
                      "results": {k: slim(v) for k, v in mc["imbalance"]["results"].items()}},
        "calibration": {"a": cal["a"], "b": cal["b"],
                        "uncalibrated": slim(cal["uncalibrated"]), "platt": slim(cal["platt"]),
                        "isotonic": slim(cal["isotonic"]), "reliability": cal["reliability"]},
        "ood": {"n": ood["n"], "scores": slim(ood["scores"]), "reliability": ood["reliability"],
                "share_confident": round(ood["share_confident"], 4)},
        "top_features": top,
        "hparams": mc["hparams"],
    }
    (WEB_DATA / "report.json").write_text(json.dumps(site, indent=1))
    print("report.json", (WEB_DATA / "report.json").stat().st_size // 1024, "KB")


if __name__ == "__main__":
    main()
