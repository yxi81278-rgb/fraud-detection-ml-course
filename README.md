# Fraud Detection with Advanced Sequence Models

欺诈通话检测 — 基于 Mamba / xLSTM / Transformer 的社会工程对抗鲁棒性研究

## Project Structure

```
fraud_detection_project/
├── data/
│   ├── raw/fraud_calls.xlsx        # 2677 fraud call transcripts
│   ├── raw/thucnews/               # Download THUCNews here
│   ├── processed/                  # Generated: prepared_data.pkl
│   └── attacked/                   # Generated: rewritten_test_samples.pkl
├── src/
│   ├── utils.py                    # Tokenization, datasets, metrics
│   ├── models.py                   # Mamba, xLSTM (mLSTM+sLSTM), Transformer
│   ├── 01_prepare_data.py          # Data loading & preprocessing
│   ├── 03_train.py                 # Train all classifiers
│   ├── 04_rewrite.py               # Fraud-R1 attack strategies
│   └── 05_evaluate.py              # Full evaluation & visualization
├── models/                         # Saved model checkpoints
├── results/
│   ├── tables/all_results.csv      # All evaluation metrics
│   └── figures/                    # Generated charts
└── README.md
```

## Models

| Model | Architecture | Paper | Year |
|-------|-------------|-------|------|
| Mamba | Selective State Space Model (S6) | Gu & Dao | 2024 |
| xLSTM | Extended LSTM (Matrix + Scalar) | Beck & Hochreiter | NeurIPS 2024 |
| Transformer | Lightweight encoder from scratch | Vaswani et al. | 2017/2024 |
| XGBoost | TF-IDF + Gradient Boosting | Chen & Guestrin | 2016 |

## Attack Strategies

1. Trust Building (trust) — Insert trust-establishing phrases
2. Urgency (urgency) — Add time pressure
3. Emotional (emotion) — Appeal to emotions
4. Authority (authority) — Impersonate authorities
5. Comprehensive (comprehensive) — Multi-turn combination

## Requirements

```
torch>=1.12  transformers  scikit-learn  xgboost  jieba
openpyxl  pandas  numpy  matplotlib  tqdm
```

## Quick Start

### 1. Prepare Data
```bash
# Download THUCNews: http://thuctc.thunlp.org/
# Extract to: data/raw/thucnews/
python src/01_prepare_data.py
```

### 2. Train Models
```bash
python src/03_train.py --models all --epochs 30 --batch_size 32
```

### 3. Generate Attack Variants
```bash
python src/04_rewrite.py
```

### 4. Evaluate
```bash
python src/05_evaluate.py
```

Results appear in `results/tables/all_results.csv` and `results/figures/`.
