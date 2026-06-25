
# -*- coding: utf-8 -*-
# utils.py — 公共工具函数

import re
import os
import sys
import random
import pickle
import joblib
import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Tuple, Dict, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import jieba

# ========== 配置 ==========
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW = os.path.join(PROJECT_ROOT, 'data', 'raw')
DATA_PROCESSED = os.path.join(PROJECT_ROOT, 'data', 'processed')
MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
RESULTS_DIR = os.path.join(PROJECT_ROOT, 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
TABLES_DIR = os.path.join(RESULTS_DIR, 'tables')
DATA_ATTACKED = os.path.join(PROJECT_ROOT, 'data', 'attacked')


# ========== 中文文本预处理 ==========
def clean_text(text: str) -> str:
    # 去除多余空白和特殊字符
    text = re.sub(r"[-]", "", text)
    text = re.sub(r'[-]', '', text)
    return text.strip()

def tokenize(text: str) -> List[str]:
    text = clean_text(text)
    tokens = jieba.lcut(text)
    tokens = [t.strip() for t in tokens if t.strip()]
    return tokens

def texts_to_sequences(texts: List[str], vocab: Dict[str, int], max_len: int = 512) -> np.ndarray:
    seqs = np.zeros((len(texts), max_len), dtype=np.int64)
    for i, text in enumerate(texts):
        tokens = tokenize(text)[:max_len]
        for j, token in enumerate(tokens):
            seqs[i, j] = vocab.get(token, 1)  # 1 = UNK
    return seqs

def build_vocab(texts: List[str], max_size: int = 30000, min_freq: int = 2) -> Dict[str, int]:
    counter = Counter()
    for text in texts:
        tokens = tokenize(text)
        counter.update(tokens)
    vocab = {'[PAD]': 0, '[UNK]': 1}
    for word, cnt in counter.most_common(max_size - 2):
        if cnt >= min_freq:
            vocab[word] = len(vocab)
    return vocab


# ========== 数据集类 ==========
class TextDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Dict[str, int] = None,
                 max_len: int = 512, tfidf_vectorizer: TfidfVectorizer = None,
                 fit_vectorizer: bool = False):
        self.texts = [clean_text(t) for t in texts]
        self.labels = np.array(labels, dtype=np.int64)
        self.max_len = max_len
        # Token sequences
        if vocab is not None:
            self.sequences = texts_to_sequences(self.texts, vocab, max_len)
        else:
            self.sequences = None
        # TF-IDF features
        if tfidf_vectorizer is not None:
            if fit_vectorizer:
                self.tfidf = tfidf_vectorizer.fit_transform(self.texts).toarray().astype(np.float32)
            else:
                self.tfidf = tfidf_vectorizer.transform(self.texts).toarray().astype(np.float32)
        else:
            self.tfidf = None

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        item = {'label': torch.tensor(self.labels[idx], dtype=torch.long)}
        if self.sequences is not None:
            item['input_ids'] = torch.tensor(self.sequences[idx], dtype=torch.long)
        if self.tfidf is not None:
            item['tfidf'] = torch.tensor(self.tfidf[idx], dtype=torch.float32)
        return item


# ========== 训练工具 ==========
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device() -> torch.device:
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    acc = accuracy_score(labels, preds)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary', zero_division=0)
    return {'accuracy': acc, 'precision': precision, 'recall': recall, 'f1': f1}

def format_metrics(metrics: Dict[str, float]) -> str:
    return ' | '.join(f'{k}: {v:.4f}' for k, v in metrics.items())


# ========== 保存/加载 ==========
def save_pickle(obj, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(obj, path)

def load_pickle(path: str):
    return joblib.load(path)

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save(model.state_dict(), path)

def load_model(model: nn.Module, path: str, device: torch.device = None):
    model.load_state_dict(torch.load(path, map_location=device or get_device()))
    return model

print('utils.py OK')
