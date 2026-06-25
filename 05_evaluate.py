# -*- coding: utf-8 -*-
# 05_evaluate.py - Full evaluation: XGBoost/Logistic/BiLSTM/Transformer vs 7 conditions

import os, sys, json, numpy as np, pandas as pd, torch, torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import *
from models import create_model
set_seed(42)
matplotlib.rcParams['font.sans-serif']=['DejaVu Sans']
matplotlib.rcParams['axes.unicode_minus']=False


def evaluate_sklearn_model(model, X_tfidf, labels):
    preds = model.predict(X_tfidf)
    return compute_metrics(labels, preds)


@torch.no_grad()
def evaluate_dl_model(model, texts, labels, vocab, device, max_len=512, batch_size=64):
    dataset = TextDataset(texts, labels, vocab=vocab, max_len=max_len)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    model.eval()
    all_preds, all_labels = [], []
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        logits = model(input_ids)
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(batch["label"].numpy())
    return compute_metrics(np.array(all_labels), np.array(all_preds))


def evaluate_all(data, rewritten_data, models_info, vocab, device):
    X_test = data["X_test"]
    y_test = np.array(data["y_test"])
    test_texts = X_test
    test_labels = y_test.tolist()
    strategies = ["original","attacked","trust","urgency","emotion","authority","comprehensive"]
    results = []
    print("Evaluating on all test conditions...")
    for strategy in strategies:
        print(f"  Strategy: {strategy}")
        if strategy == "original":
            eval_texts = test_texts
            eval_labels = test_labels
        else:
            strategy_texts = rewritten_data.get(strategy, rewritten_data.get("original",[]))
            eval_texts = []
            eval_labels = []
            fi = 0
            for i, (text, label) in enumerate(zip(test_texts, test_labels)):
                if label == 1:
                    eval_texts.append(strategy_texts[fi] if fi < len(strategy_texts) else text)
                    fi += 1
                else:
                    eval_texts.append(text)
                eval_labels.append(label)
        for model_name, info in models_info.items():
            if info["type"] == "sklearn":
                if strategy != "original":
                    continue
                m = evaluate_sklearn_model(info["model"], data["X_test_tfidf"], np.array(eval_labels))
            else:
                m = evaluate_dl_model(info["model"], eval_texts, eval_labels, vocab, device)
            results.append({"model":model_name,"strategy":strategy,**m})
            print(f"    {model_name:15s} | {format_metrics(m)}")
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(TABLES_DIR,"all_results.csv"),index=False)
    return df


def plot_results(df, save_dir):
    print("Generating figures...")
    models = df["model"].unique()
    strategies = df["strategy"].unique()
    colors = ["#2196F3","#4CAF50","#FF9800","#9C27B0","#F44336","#00BCD4"]
    # Figure 1: F1 bar chart
    fig, ax = plt.subplots(figsize=(14,6))
    x = np.arange(len(strategies))
    width = 0.15
    for i, model in enumerate(models):
        mdf = df[df["model"]==model].set_index("strategy")
        f1s = [mdf.loc[s,"f1"] if s in mdf.index else 0 for s in strategies]
        offset = (i - (len(models)-1)/2) * width
        ax.bar(x + offset, f1s, width, label=model, color=colors[i%len(colors)])
    ax.set_ylabel("F1 Score")
    ax.set_title("F1 Score Across Models and Attack Strategies")
    ax.set_xticks(x)
    ax.set_xticklabels(strategies, rotation=15)
    ax.legend(loc="lower left")
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir,"f1_comparison.png"), dpi=150)
    plt.close()
    # Figure 2: F1 drop
    fig, ax = plt.subplots(figsize=(12,6))
    for i, model in enumerate(models):
        mdf = df[df["model"]==model].set_index("strategy")
        if "original" in mdf.index:
            baseline = mdf.loc["original","f1"]
            drops = []
            labels = []
            for s in strategies:
                if s != "original" and s in mdf.index:
                    drops.append(baseline - mdf.loc[s,"f1"])
                    labels.append(s)
            ax.plot(range(len(drops)), drops, "o-", label=model, color=colors[i%len(colors)], linewidth=2, markersize=8)
    ax.set_ylabel("F1 Drop")
    ax.set_title("F1 Score Degradation per Attack Strategy")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(os.path.join(save_dir,"robustness_drop.png"), dpi=150)
    plt.close()
    print(f"  Figures saved to {save_dir}/")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=os.path.join(DATA_PROCESSED,"prepared_data.pkl"))
    p.add_argument("--rewritten", default=os.path.join(DATA_ATTACKED,"rewritten_test_samples.pkl"))
    p.add_argument("--model_dir", default=MODELS_DIR)
    p.add_argument("--device_id", type=int, default=0)
    args = p.parse_args()
    device = torch.device(f"cuda:{args.device_id}" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    data = load_pickle(args.data)
    vocab = data["vocab"]
    rewritten_data = load_pickle(args.rewritten) if os.path.exists(args.rewritten) else {}
    print(f"Rewritten strategies: {list(rewritten_data.keys())}")
    models_info = {}
    for path, name, mtype in [
        (os.path.join(args.model_dir,"xgboost_model.pkl"),"XGBoost","sklearn"),
        (os.path.join(args.model_dir,"logistic_model.pkl"),"Logistic","sklearn")
    ]:
        if os.path.exists(path):
            models_info[name] = {"type":mtype,"model":load_pickle(path)}
    for mt in ["bilstm", "transformer"]:
        path = os.path.join(args.model_dir, f"{mt}_best.pt")
        if os.path.exists(path):
            model = create_model(mt, len(vocab), n_layers=4, d_model=128)
            model.load_state_dict(torch.load(path, map_location=device))
            model = model.to(device)
            models_info[mt.upper()] = {"type":"dl","model":model}
    print(f"Loaded {len(models_info)} models")
    if not models_info:
        print("ERROR: No models found!")
        return
    df = evaluate_all(data, rewritten_data, models_info, vocab, device)
    plot_results(df, FIGURES_DIR)
    best = df[df["strategy"]=="original"].sort_values("f1",ascending=False).iloc[0]
    print(f"Best model: {best['model']} F1={best['f1']:.4f}")
    print(f"Results: {TABLES_DIR}/all_results.csv | Figures: {FIGURES_DIR}/")


if __name__ == "__main__":
    main()