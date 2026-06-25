# -*- coding: utf-8 -*-
# 01_prepare_data.py — 数据加载、负样本构造、数据集划分

import os, sys, glob, random
import numpy as np
import pandas as pd
import openpyxl
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import jieba

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    PROJECT_ROOT, DATA_RAW, DATA_PROCESSED, clean_text,
    build_vocab, save_pickle, load_pickle, set_seed
)

set_seed(42)


def load_fraud_data(xlsx_path: str):
    wb = openpyxl.load_workbook(xlsx_path)
    ws = wb.active
    texts_orig, texts_attacked = [], []
    for row in ws.iter_rows(min_row=2, values_only=True):
        orig, attacked = row[0], row[1]
        if orig and str(orig).strip():
            texts_orig.append(str(orig).strip())
            texts_attacked.append(str(attacked).strip() if attacked else '')
    print(f'[Fraud] Loaded {len(texts_orig)} fraud call transcripts')
    return texts_orig, texts_attacked


def load_thucnews(data_dir: str, n_samples: int = 3000):
    texts = []
    # THUCNews 结构: thucnews/<category>/<file>.txt
    categories = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    if not categories:
        # 尝试其他结构: 根目录下直接是txt文件
        txt_files = glob.glob(os.path.join(data_dir, '**', '*.txt'), recursive=True)
        if txt_files:
            print(f'[THUCNews] Found {len(txt_files)} txt files (flat structure)')
            random.shuffle(txt_files)
            for f in txt_files[:n_samples]:
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                        text = fh.read().strip()
                        if len(text) > 30:
                            texts.append(text)
                except:
                    pass
        else:
            raise FileNotFoundError(
                f'THUCNews data not found at {data_dir}. '
                'Please download from http://thuctc.thunlp.org/ and extract to data/raw/thucnews/'
            )
    else:
        per_cat = max(n_samples // len(categories), 1)
        print(f'[THUCNews] Loading from {len(categories)} categories, ~{per_cat} per category')
        for cat in categories:
            cat_dir = os.path.join(data_dir, cat)
            txt_files = glob.glob(os.path.join(cat_dir, '*.txt'))
            random.shuffle(txt_files)
            for f in txt_files[:per_cat]:
                try:
                    with open(f, 'r', encoding='utf-8', errors='ignore') as fh:
                        text = fh.read().strip()
                        if len(text) > 30:
                            texts.append(text)
                except:
                    pass
    texts = texts[:n_samples]
    print(f'[THUCNews] Loaded {len(texts)} normal texts as negative samples')
    return texts


def prepare_data(fraud_xlsx: str, thucnews_dir: str = None):
    print('=' * 60)
    print('Step 1: Loading fraud call data...')
    texts_fraud_orig, texts_fraud_attacked = load_fraud_data(fraud_xlsx)

    # 负样本
    if thucnews_dir and os.path.exists(thucnews_dir):
        print(f'\nStep 2: Loading negative samples from {thucnews_dir}...')
        texts_normal = load_thucnews(thucnews_dir, n_samples=len(texts_fraud_orig) + 300)
    else:
        print(f'\n[WARNING] THUCNews not found at {thucnews_dir}!')
        print('Using TextFooler-attacked fraud calls as temporary negative samples for test only.')
        print('Please download THUCNews and place in data/raw/thucnews/ for proper training.')
        # 临时方案: 用 TextFooler 攻击版作为负样本 + 追加一些随机噪声文本
        texts_normal = []
        for t in texts_fraud_attacked[:len(texts_fraud_orig)]:
            texts_normal.append(t)
        # 追加一些中文模板生成的负样本
        templates = [
            '您好，请问有什么可以帮您的？',
            '感谢您的来电，祝您生活愉快。',
            '您的快递已送达，请及时取件。',
            '今天天气不错，适合出去走走。',
            '明天开会请准时参加。',
            '公司季度财报已发布，请查看邮件。',
        ]
        while len(texts_normal) < len(texts_fraud_orig):
            texts_normal.append(random.choice(templates))
        print(f'[FALLBACK] Using {len(texts_normal)} fake negative samples (FOR TESTING ONLY)')

    # 构建完整数据集
    all_texts = texts_fraud_orig + texts_normal
    all_labels = [1] * len(texts_fraud_orig) + [0] * len(texts_normal)

    print(f'\nStep 3: Dataset statistics')
    print(f'  Fraud (positive): {len(texts_fraud_orig)}')
    print(f'  Normal (negative): {len(texts_normal)}')
    print(f'  Total: {len(all_texts)}')

    # 划分训练集/验证集/测试集
    print(f'\nStep 4: Splitting into train/val/test (70/15/15)...')
    X_temp, X_test, y_temp, y_test = train_test_split(
        all_texts, all_labels, test_size=0.15, random_state=42, stratify=all_labels
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.15/0.85, random_state=42, stratify=y_temp
    )
    # 测试集中只保留欺诈样本的索引用于后续改写
    test_fraud_indices = [i for i, l in enumerate(y_test) if l == 1]
    X_test_fraud_orig = [X_test[i] for i in test_fraud_indices]
    X_test_fraud_attacked = [texts_fraud_attacked[i] for i in range(len(texts_fraud_orig))
                              if i < len(texts_fraud_attacked) and all_texts[i] in X_test]

    print(f'  Train: {len(X_train)} ({sum(y_train)} fraud / {len(y_train)-sum(y_train)} normal)')
    print(f'  Val:   {len(X_val)} ({sum(y_val)} fraud / {len(y_val)-sum(y_val)} normal)')
    print(f'  Test:  {len(X_test)} ({sum(y_test)} fraud / {len(y_test)-sum(y_test)} normal)')

    # 构建词表
    print(f'\nStep 5: Building vocabulary...')
    vocab = build_vocab(X_train, max_size=20000, min_freq=2)
    print(f'  Vocabulary size: {len(vocab)}')

    # TF-IDF 向量化
    print(f'\nStep 6: Building TF-IDF vectorizer...')
    tfidf_vec = TfidfVectorizer(
        max_features=5000, ngram_range=(1, 2),
        tokenizer=jieba.lcut,
        token_pattern=None
    )
    tfidf_vec.fit(X_train)
    print(f'  TF-IDF features: {tfidf_vec.max_features}')

    # 保存 (tfidf_vectorizer 单独存，避免 pickle 报错)
    print(f'\nStep 7: Saving processed data...')
    os.makedirs(DATA_PROCESSED, exist_ok=True)
    # 先存 TF-IDF 矩阵
    X_train_tfidf = tfidf_vec.transform(X_train).toarray().astype('float32')
    X_val_tfidf = tfidf_vec.transform(X_val).toarray().astype('float32')
    X_test_tfidf = tfidf_vec.transform(X_test).toarray().astype('float32')
    data = {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
        'X_test_fraud_orig': X_test_fraud_orig,
        'X_test_fraud_attacked': X_test_fraud_attacked,
        'texts_fraud_orig': texts_fraud_orig,
        'texts_fraud_attacked': texts_fraud_attacked,
        'texts_normal': texts_normal,
        'vocab': vocab,
        'X_train_tfidf': X_train_tfidf,
        'X_val_tfidf': X_val_tfidf,
        'X_test_tfidf': X_test_tfidf,
    }
    save_pickle(data, os.path.join(DATA_PROCESSED, 'prepared_data.pkl'))
    print(f'\nAll done! Data saved to {DATA_PROCESSED}/prepared_data.pkl')
    return data


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--fraud_xlsx', type=str,
                        default='data/raw/fraud_calls.xlsx',
                        help='Path to fraud call xlsx file')
    parser.add_argument('--thucnews_dir', type=str,
                        default=os.path.join(DATA_RAW, 'thucnews'),
                        help='Path to THUCNews directory')
    args = parser.parse_args()
    prepare_data(args.fraud_xlsx, args.thucnews_dir)
