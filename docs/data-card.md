# Dataset card

Source: [GroupLens MovieLens 100K](https://grouplens.org/datasets/movielens/100k/).
100,000 ratings, 943 users, 1,682 movies collected in 1997–1998. This dataset
supports a low-cost recommendation-methodology study. It does not contain
video pixels, impressions, dwell time, watch completion, or randomized exposure.

Read [the original terms](https://files.grouplens.org/datasets/movielens/ml-100k-README.txt)
before downloading. Research use requires acknowledgment. Redistribution and
commercial/revenue-bearing uses require separate permission under those terms.
The ZIP ships code and aggregate results only: no ratings, item metadata,
user demographics, model weights, or raw training records are redistributed.
No GroupLens, University of Minnesota, or Meta endorsement is implied.

Downloaded bytes are checked against a known SHA256. Canonical interaction,
partition and genre digests are recorded. Only u.data, u.item and the original
README are extracted; ZIP paths are never extracted wholesale. User demographic
files are not extracted. All artifacts derived at row or model level stay local.

Temporal splitting excludes cold-start observations, so coverage is reported
alongside metrics. Self-selected explicit ratings introduce exposure and selection
bias. A real video expansion should use properly licensed exposure logs with
user/time splits, propensity-aware analysis and watch-time-specific labels.
