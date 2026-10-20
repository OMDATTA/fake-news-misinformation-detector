#!/usr/bin/env bash
# Reproduce every number in the README and on the website.
set -euo pipefail
cd "$(dirname "$0")/scripts"
python3 01_prepare_data.py          # download ISOT, scrub, de-duplicate, split
python3 02_source_topic_analysis.py # publisher-artifact audit + NMF topics
python3 03_train_compare.py         # model comparison, leakage, topic hold-out, imbalance
python3 05_calibrate_export.py      # final model + Platt calibration -> web/assets/data/model.json
python3 06_out_of_distribution.py   # WELFake out-of-distribution test
python3 04_transformer_embeddings.py  # MiniLM embeddings (slow on CPU: ~15 min)
python3 07_embeddings_ood.py        # embeddings vs TF-IDF out-of-distribution
python3 08_build_site_data.py       # bundle reports -> web/assets/data/report.json
node ../tests/parity.mjs           # browser model == Python model
python3 09_render_readme.py         # README.md from template + reports
