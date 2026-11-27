# Archive

The original 2022 version of this project: an ANN predicting `lag_price` on a Kaggle
(Olist-derived) monthly dataset. It was kept for history and replaced by the RGM pipeline in `src/`.
It was retired because it predicted last month's price instead of recommending one. Its features also leaked the target (`unit_price`), and it used a random rather than time-based train/test split.
