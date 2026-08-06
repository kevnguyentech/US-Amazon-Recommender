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

**RMSE:** 1.15 (vs 1.24 baseline of predicting the global average rating)

**Precision@5:** 0.858 (vs 0.846 random baseline)

Model beats baseline on RMSE but barely beats random on Precision@5. Ratings skew hard toward 4-5 stars (mean 4.26), so almost any ranking looks "precise" against this metric, including random. The real limiter isn't item sparsity (9.95 ratings/item is workable) but that the model only sees user_id and item_id, no content signal to distinguish *why* someone likes an item.

**Next step:** add item content features (title/description embeddings) so the model has signal beyond just IDs.