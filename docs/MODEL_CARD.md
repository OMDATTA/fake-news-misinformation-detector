# Model card: Misinformation-style resemblance classifier v1.0

## Summary
A class-weighted logistic regression over **25,000 TF-IDF word 1-2-grams** and **15 standardised linguistic features**, followed by **Platt calibration**. It outputs a calibrated probability that a news text belongs with the *misleading* examples of the ISOT corpus rather than the *reliable* (Reuters) examples.

## Intended use
* Education and research on NLP, explainability and calibration.
* A prompt for critical reading: *why* does this text read like misleading news?

## Out-of-scope uses
* Deciding whether a claim, article, person or outlet is truthful.
* Content moderation, ranking or takedown decisions.
* Non-English text, or news outside roughly 2015–2018 US politics and world affairs.

## Training data
ISOT Fake News Dataset: "true" articles are Reuters wire copy, and "fake" articles come from websites flagged by PolitiFact and Wikipedia. Labels are assigned **by source**, not by checking individual claims. After de-duplication there are 38k articles, split 70/10/20 into train, calibration and test.

## Pre-processing
Publisher fingerprints are removed before any feature extraction: Reuters mentions, wire datelines, image credits, URLs, @handles, embedded-tweet links, "[VIDEO]"-style tags, timestamps, "Factbox", "Read more" and "via". These markers alone reach 96% cross-validated accuracy (see `reports/source_topic.json`).

## Evaluation
See `reports/*.json`. Headline results:
* In-distribution (ISOT test): about 99% accuracy, ECE about 0.2% after Platt scaling.
* Out-of-distribution (WELFake, de-duplicated against ISOT): about 73% accuracy, ROC-AUC about 0.79, ECE about 18%. **The model is overconfident on unfamiliar sources.**
* Topic hold-out: accuracy stays at 98–99% when a whole NMF topic is unseen.
* Transformer embeddings (MiniLM) do not transfer better than TF-IDF out-of-distribution.

## Explanations
Because the model is linear, per-phrase contributions (`coef x tfidf`) and per-feature contributions (`coef x z`) are exact decompositions of the logit before calibration.

## Ethical considerations
* A false story written in a sober style will look "reliable", and an accurate story written in an excited style will look "misleading".
* Associating a writing style with misinformation can unfairly penalise informal, activist or non-native writing.
* The UI therefore uses resemblance language, shows calibrated probabilities instead of verdicts, never displays 0% or 100%, warns on short or unfamiliar text, and links to the limitations.
