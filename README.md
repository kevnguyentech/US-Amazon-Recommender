# US Amazon Recommender

A neural recommendation system trained on Amazon Electronics reviews, using a two-tower embedding model with bias terms. Built end-to-end: data pipeline, model training, evaluation, and a deployed FastAPI service.

## Stack
- Python, PyTorch (model)
- Pandas, scikit-learn (data pipeline)
- FastAPI, Uvicorn (deployment)
- Hugging Face `datasets` (Amazon Reviews 2023 by McAuley Lab)

## How it works
Each user and item gets a learned embedding vector. Predicted rating = dot product of user and item embeddings + user bias + item bias + global bias. Trained with MSE loss on 78k user-item-rating triples.

## Project structure
```
US_Amazon_Recommender/
├── data/
│   ├── load_amazon.py    # pulls Amazon Reviews 2023 via HF datasets
│   ├── prepare.py        # filters sparse users/items, splits train/test
│   └── check_data.py     # quick null/describe check on the raw CSV
├── model/
│   ├── dataset.py         # PyTorch Dataset wrapper
│   ├── two_tower.py       # model architecture
│   ├── train.py           # training loop
│   └── evaluate.py        # precision@k eval with random baseline
├── api/
│   └── main.py            # FastAPI predict + recommend endpoints
├── tests/                 # pytest suite for prepare/model/api
└── requirements.txt
```

## Running it
```bash
pip install -r requirements.txt
python data/load_amazon.py
python data/prepare.py
python model/train.py
python model/evaluate.py
python -m pytest tests/ -v
uvicorn api.main:app --reload
```
API docs at `http://127.0.0.1:8000/docs`. Two endpoints: `/predict` (single user-item rating prediction) and `/recommend` (top-k items for a user).

## Results & Limitations

**Setup:** 300k Amazon Electronics reviews, filtered to users/items with 5+ ratings (97,646 rows), 55,108 users, 9,809 items, avg 9.95 ratings/item.

> **Metrics below are stale and pending a re-run.** They were measured before
> two evaluation fixes and neither is trustworthy as written. Re-run
> `python data/prepare.py && python model/train.py && python model/evaluate.py`
> and replace them.

**RMSE:** 1.15 (vs 1.24 baseline of predicting the global average rating) *(stale: optimistic)*

**Precision@5:** 0.858 (vs 0.846 random baseline) *(stale: mostly measuring nothing)*

Why the old numbers can't be trusted:

- **RMSE was optimistic.** The train/test split was a uniform random split, so the
  model trained on a user's later reviews and was tested on their earlier ones.
  The split is now per-user temporal (each user's most recent 20% is held out),
  which removes the lookahead. Expect the honest RMSE to be higher than 1.15.
- **Precision@5 was mostly dead weight.** Users with exactly 5 test items were
  included, and for them `topk(preds, 5)` and `random.sample(range(5), 5)` both
  select the entire candidate set, so model and baseline precision are *identical
  by construction*. On a synthetic run with the same shape, every one of those
  users had a model-minus-random gap of exactly 0.0, and they were 55.6% of the
  evaluated population. The reported gap was therefore an average over a majority
  that could not register any difference. Note this does not bias the aggregate in
  a predictable direction: it depends on where that subgroup's precision level
  sits relative to the rest. Those users are now excluded.

The original reading still looks right and is worth re-testing once the numbers
are honest: ratings skew hard toward 4-5 stars (mean 4.26), so almost any ranking
looks "precise" against this metric. The real limiter probably isn't item sparsity
(9.95 ratings/item is workable) but that the model only sees user_id and item_id,
no content signal to distinguish *why* someone likes an item.

One caveat that still stands: Precision@5 ranks each user's *test items only*,
which are all items they actually rated. That is a known-positives-only protocol
and it inflates both the model and the baseline toward the base rate of 4+ ratings.
Ranking over the full catalog would be the standard protocol and would drop the
number substantially.

**Next step:** add item content features (title/description embeddings) so the model has signal beyond just IDs.