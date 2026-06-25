# Fraud Detection with Social Engineering Attack Robustness Analysis

虚假通话检测 — 基于 BiLSTM+Attention / Transformer 的 Fraud-R1 社会工程对抗鲁棒性研究

## Project Structure

```
fraud_detection_project/
├── data/
│   ├── raw/fraud_calls.xlsx        # 2677 fraud call transcripts
│   ├── raw/thucnews/               # THUCNews news corpus
│   ├── processed/                  # Generated: prepared_data.pkl
│   └── attacked/                   # Generated: rewritten_test_samples.pkl
├── src/
│   ├── utils.py                    # Tokenization, datasets, metrics
│   ├── models.py                   # BiLSTM+Attention, Transformer
│   ├── 01_prepare_data.py          # Data loading & preprocessing
│   ├── 03_train.py                 # Train all classifiers
│   ├── 04_rewrite.py               # Fraud-R1 attack strategies
│   └── 05_evaluate.py              # Full evaluation & visualization
├── models/                         # Saved model checkpoints
├── results/                        # Experiment results (customer service negatives)
├── results_exp1/                   # Experiment results (news negatives)
├── results_exp2/                   # Experiment results (customer service, seed=123)
├── results_exp3/                   # Experiment results (news, adjusted attack params)
└── README.md
```

## Models

| Model | Architecture | Paper | Year |
|-------|-------------|-------|------|
| BiLSTM+Attention | Bidirectional LSTM + Self-Attention | Hochreiter & Schmidhuber / Bahdanau et al. | 1997 / 2015 |
| Transformer | Lightweight encoder from scratch | Vaswani et al. | 2017 |
| XGBoost | TF-IDF + Gradient Boosting | Chen & Guestrin | 2016 |
| Logistic Regression | TF-IDF + Linear Classifier | — | — |

## Attack Strategies (Fraud-R1 Inspired)

1. Trust Building (trust) — Insert trust-establishing phrases
2. Urgency (urgency) — Add time pressure
3. Emotion (emotion) — Appeal to emotions
4. Authority (authority) — Impersonate authorities
5. Comprehensive (comprehensive) — Multi-turn combination

## Requirements

```
torch>=1.12  scikit-learn  xgboost  jieba
openpyxl  pandas  numpy  matplotlib  tqdm
```

## Quick Start

### 1. Prepare Data
```bash
python src/01_prepare_data.py
```

### 2. Train Models
```bash
python src/03_train.py --models all --epochs 30 --batch_size 32
```

### 3. Generate Attack Variants (Fraud-R1 strategies)
```bash
python src/04_rewrite.py
```

### 4. Evaluate
```bash
python src/05_evaluate.py
```

Results appear in `results/tables/all_results.csv` and `results/figures/`.

## Experiment Results Summary

Four experiments were conducted with different negative sample sources and configurations:

| Experiment | Negative Samples | Key Finding |
|-----------|-----------------|-------------|
| Exp 1 (results_exp1/) | THUCNews news (1,838) | BiLSTM F1=0.9988 on Original/Attacked/Trust; Transformer F1=1.0000 everywhere |
| Exp 2 (results/) | Customer service dialogs (2,976) | All models F1=1.0000 across all 7 conditions |
| Exp 3 (results_exp2/) | Customer service dialogs, seed=123 | Identical to Exp 2 — verified reproducibility |
| Exp 4 (results_exp3/) | THUCNews news, adjusted attack params | Identical pattern to Exp 1 — attack param changes did not alter behavior |

All detailed metrics (Accuracy, Precision, Recall, F1) are available in `all_results.csv` files under each results directory.

