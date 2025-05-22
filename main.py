import argparse
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torchinfo import summary

from src.data import load_data
from src.utils import normalize_fn
from src.methods.deep_network import MLP, Trainer
from src.utils import accuracy_fn, macrof1_fn, get_n_classes


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def k_fold_indices(n_samples, k=5, seed=42):
    rng = np.random.default_rng(seed)
    indices = rng.permutation(n_samples)
    fold_sizes = [n_samples // k] * k
    for i in range(n_samples % k):
        fold_sizes[i] += 1
    current = 0
    folds = []
    for fold_size in fold_sizes:
        start, stop = current, current + fold_size
        val_idx = indices[start:stop]
        train_idx = np.concatenate((indices[:start], indices[stop:]))
        folds.append((train_idx, val_idx))
        current = stop
    return folds


def main(args):
    set_seed(42)

    # Load and reshape data
    xtrain, xtest, ytrain, y_test = load_data()
    xtrain = xtrain.reshape(xtrain.shape[0], -1).astype(np.float32) / 255.0
    xtest  = xtest.reshape(xtest.shape[0], -1).astype(np.float32) / 255.0
    n_classes = get_n_classes(ytrain)

    # 5-Fold Cross-validation
    folds = k_fold_indices(len(xtrain), k=5)
    accs, f1s = [], []

    for i, (tr_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {i+1}/5 ---")
        x_tr, y_tr = xtrain[tr_idx], ytrain[tr_idx]
        x_val, y_val = xtrain[val_idx], ytrain[val_idx]

        # Normalize
        mu = x_tr.mean(axis=0, keepdims=True)
        sigma = x_tr.std(axis=0, keepdims=True) + 1e-8
        x_tr = normalize_fn(x_tr, mu, sigma)
        x_val = normalize_fn(x_val, mu, sigma)

        # Reshape if CNN, otherwise use MLP
        if args.nn_type == "cnn":
            from src.methods.deep_network import CNN
            xtrain_img = x_tr.reshape(-1, 28, 28, 3)
            xval_img = x_val.reshape(-1, 28, 28, 3)
            x_tr = np.transpose(xtrain_img, (0, 3, 1, 2))
            x_val = np.transpose(xval_img, (0, 3, 1, 2))
            model = CNN(input_channels=3, n_classes=n_classes)
        elif args.nn_type == "mlp":
            model = MLP(input_size=2352, n_classes=n_classes, hidden_layers=[256], dropout_prob=0.5)
        else:
            raise ValueError("nn_type must be 'mlp' or 'cnn'")

        summary(model)

        trainer = Trainer(model, lr=args.lr, epochs=args.epochs, batch_size=args.nn_batch_size)
        preds_train = trainer.fit(x_tr, y_tr)
        preds_val = trainer.predict(x_val)

        acc_tr = accuracy_fn(preds_train, y_tr)
        f1_tr = macrof1_fn(preds_train, y_tr)
        acc_val = accuracy_fn(preds_val, y_val)
        f1_val = macrof1_fn(preds_val, y_val)

        accs.append(acc_val)
        f1s.append(f1_val)

        print(f"Train : accuracy = {acc_tr:.3f}% | F1 = {f1_tr:.6f}")
        print(f"Validation : accuracy = {acc_val:.3f}% | F1 = {f1_val:.6f}")

    # Best fold results
    best_idx = int(np.argmax(accs))
    print("\n==== Best fold result ====")
    print(f"Best Accuracy : {accs[best_idx]:.3f}%")
    print(f"Best F1-score : {f1s[best_idx]:.6f}")
    print(f"Found at fold {best_idx+1}/5")

    # Test mode
    if args.test:
        mu = xtrain.mean(axis=0, keepdims=True)
        sigma = xtrain.std(axis=0, keepdims=True) + 1e-8
        xtest = normalize_fn(xtest, mu, sigma)

        if args.nn_type == "cnn":
            from src.methods.deep_network import CNN
            xtest = np.transpose(xtest.reshape(-1, 28, 28, 3), (0, 3, 1, 2))
            model = CNN(input_channels=3, n_classes=n_classes)
        else:
            model = MLP(input_size=2352, n_classes=n_classes, hidden_layers=[256], dropout_prob=0.5)

        summary(model)

        trainer = Trainer(model, lr=args.lr, epochs=args.epochs, batch_size=args.nn_batch_size)
        trainer.fit(xtrain, ytrain)
        preds_val = trainer.predict(xtest)

        np.save("dermamnist_test_preds.npy", preds_val)
        print("Test predictions saved to dermamnist_test_preds.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str)
    parser.add_argument('--nn_type', default="mlp", choices=["mlp", "cnn"])
    parser.add_argument('--nn_batch_size', type=int, default=128)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--test', action="store_true")

    args = parser.parse_args()
    main(args)
