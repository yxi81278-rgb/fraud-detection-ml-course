import os, sys, time, json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
import xgboost as xgb
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from utils import (
    PROJECT_ROOT, DATA_PROCESSED, MODELS_DIR, TextDataset,
    set_seed, get_device, compute_metrics, format_metrics,
    save_model, load_pickle, save_pickle
)
from models import create_model

set_seed(42)


def train_xgboost(X_train_tfidf, y_train, X_val_tfidf, y_val, model_dir):
    print('\n' + '=' * 60)
    print('Training XGBoost...')
    params = {
        'n_estimators': 200, 'max_depth': 6, 'learning_rate': 0.1,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'objective': 'binary:logistic', 'eval_metric': 'logloss',
        'random_state': 42, 'verbosity': 0,
    }
    model = xgb.XGBClassifier(**params)
    model.fit(X_train_tfidf, y_train,
              eval_set=[(X_val_tfidf, y_val)], verbose=False)
    train_preds = model.predict(X_train_tfidf)
    val_preds = model.predict(X_val_tfidf)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    print(f'  Train: {format_metrics(train_m)}')
    print(f'  Val:   {format_metrics(val_m)}')
    save_pickle(model, os.path.join(model_dir, 'xgboost_model.pkl'))
    return model, val_m


def train_logistic(X_train_tfidf, y_train, X_val_tfidf, y_val, model_dir):
    print('\n' + '=' * 60)
    print('Training Logistic Regression (baseline)...')
    model = LogisticRegression(C=1.0, max_iter=2000, random_state=42)
    model.fit(X_train_tfidf, y_train)
    train_preds = model.predict(X_train_tfidf)
    val_preds = model.predict(X_val_tfidf)
    train_m = compute_metrics(y_train, train_preds)
    val_m = compute_metrics(y_val, val_preds)
    print(f'  Train: {format_metrics(train_m)}')
    print(f'  Val:   {format_metrics(val_m)}')
    save_pickle(model, os.path.join(model_dir, 'logistic_model.pkl'))
    return model, val_m


def train_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        optimizer.zero_grad()
        logits = model(input_ids)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def evaluate_epoch(model, dataloader, criterion, device):
    model.eval()
    total_loss, all_preds, all_labels = 0, [], []
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        logits = model(input_ids)
        loss = criterion(logits, labels)
        total_loss += loss.item()
        preds = logits.argmax(dim=-1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    metrics = compute_metrics(np.array(all_labels), np.array(all_preds))
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def train_dl_model(model_type, train_dataset, val_dataset, vocab, device, model_dir, **kwargs):
    print(f'\n{"="*60}')
    print(f'Training {model_type.upper()}...')
    batch_size = kwargs.get('batch_size', 32)
    lr = kwargs.get('lr', 1e-3)
    epochs = kwargs.get('epochs', 30)
    patience = kwargs.get('patience', 5)
    d_model = kwargs.get('d_model', 128)
    n_layers = kwargs.get('n_layers', 4)
    max_len = kwargs.get('max_len', 512)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = create_model(model_type, len(vocab), d_model=d_model,
                          n_layers=n_layers, max_len=max_len, dropout=0.1)
    model = model.to(device)
    print(f'  Parameters: {sum(p.numel() for p in model.parameters()):,}')

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    best_val_f1 = 0; best_epoch = 0; no_improve = 0

    for epoch in range(1, epochs + 1):
        train_m = train_epoch(model, train_loader, optimizer, criterion, device)
        val_m = evaluate_epoch(model, val_loader, criterion, device)
        scheduler.step()

        if val_m['f1'] > best_val_f1:
            best_val_f1 = val_m['f1']
            best_epoch = epoch
            no_improve = 0
            save_model(model, os.path.join(model_dir, f'{model_type}_best.pt'))
        else:
            no_improve += 1

        if epoch % 3 == 0 or epoch == 1:
            print(f'  Epoch {epoch:3d} | Train {format_metrics(train_m)} | Val {format_metrics(val_m)}')

        if no_improve >= patience:
            print(f'  Early stopping at epoch {epoch}')
            break

    print(f'  Best: Epoch {best_epoch}, Val F1={best_val_f1:.4f}')
    model.load_state_dict(torch.load(os.path.join(model_dir, f'{model_type}_best.pt'), map_location=device))
    model = model.to(device)
    return model


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', type=str, default=os.path.join(DATA_PROCESSED, 'prepared_data.pkl'))
    parser.add_argument('--model_dir', type=str, default=MODELS_DIR)
    parser.add_argument('--models', type=str, nargs='+', default=['all'])
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--max_len', type=int, default=512)
    parser.add_argument('--device_id', type=int, default=0)
    args = parser.parse_args()

    print('Loading prepared data...')
    data = load_pickle(args.data)
    X_train = data['X_train']; y_train = data['y_train']
    X_val = data['X_val']; y_val = data['y_val']
    vocab = data['vocab']

    device = torch.device(f'cuda:{args.device_id}' if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    results = {}

    X_train_tfidf = data["X_train_tfidf"]
    X_val_tfidf = data["X_val_tfidf"]

    if 'all' in args.models or 'xgboost' in args.models:
        model, val_m = train_xgboost(X_train_tfidf, y_train, X_val_tfidf, y_val, args.model_dir)
        results['xgboost'] = val_m

    if 'all' in args.models or 'logistic' in args.models:
        model, val_m = train_logistic(X_train_tfidf, y_train, X_val_tfidf, y_val, args.model_dir)
        results['logistic'] = val_m

    train_ds = TextDataset(X_train, y_train, vocab=vocab, max_len=args.max_len)
    val_ds = TextDataset(X_val, y_val, vocab=vocab, max_len=args.max_len)

    dl_kwargs = dict(batch_size=args.batch_size, lr=args.lr, epochs=args.epochs,
                     d_model=args.d_model, n_layers=args.n_layers, max_len=args.max_len)

    for model_type in ['bilstm', 'transformer']:
        if 'all' in args.models or model_type in args.models:
            model = train_dl_model(model_type, train_ds, val_ds, vocab, device,
                                   args.model_dir, **dl_kwargs)
            val_m = evaluate_epoch(model, DataLoader(val_ds, batch_size=args.batch_size),
                                   nn.CrossEntropyLoss(), device)
            results[model_type] = val_m

    print('\n' + '=' * 60)
    print('FINAL RESULTS (Validation Set)')
    print('=' * 60)
    for name, metrics in results.items():
        print(f'  {name:15s} | {format_metrics(metrics)}')

    save_pickle(results, os.path.join(args.model_dir, 'validation_results.pkl'))
    print(f'\nResults saved to {args.model_dir}/validation_results.pkl')


if __name__ == '__main__':
    main()