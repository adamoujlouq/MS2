import argparse
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torchvision import transforms

from torchinfo import summary
from src.data import load_data
from src.utils import normalize_fn, accuracy_fn, macrof1_fn, get_n_classes
from src.methods.deep_network import MLP, Trainer


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


def apply_augmentation(x, transform):
    x_aug = []
    for img in x:
        img = img.reshape(28, 28, 3).astype(np.uint8)
        img_tensor = torch.from_numpy(img).permute(2, 0, 1)
        img_tensor = transform(img_tensor)
        x_aug.append(img_tensor.permute(1, 2, 0).numpy().reshape(-1))
    return np.array(x_aug, dtype=np.float32)


def main(args):
    set_seed(42)

    xtrain, xtest, ytrain, y_test = load_data()
    xtrain = xtrain.reshape(xtrain.shape[0], -1).astype(np.float32) / 255.0
    xtest  = xtest.reshape(xtest.shape[0], -1).astype(np.float32) / 255.0
    n_classes = get_n_classes(ytrain)

    # Data augmentation setup
    augment = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.1, contrast=0.1)
    ])

    folds = k_fold_indices(len(xtrain), k=3)
    accs, f1s = [], []

    for i, (tr_idx, val_idx) in enumerate(folds):
        print(f"\n--- Fold {i+1}/3 ---")
        x_tr, y_tr = xtrain[tr_idx], ytrain[tr_idx]
        x_val, y_val = xtrain[val_idx], ytrain[val_idx]

        # Normalisation
        mu = x_tr.mean(axis=0, keepdims=True)
        sigma = x_tr.std(axis=0, keepdims=True) + 1e-8
        x_tr = normalize_fn(x_tr, mu, sigma)
        x_val = normalize_fn(x_val, mu, sigma)

        if args.use_aug:
            x_tr = apply_augmentation(x_tr, augment)

        if args.nn_type == "cnn":
            from src.methods.deep_network import CNN
            x_tr = x_tr.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2)
            x_val = x_val.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2)
            model = CNN(input_channels=3, n_classes=n_classes)
        else:
            model = MLP(input_size=2352, n_classes=n_classes)

        summary(model)
        trainer = Trainer(model, lr=args.lr, epochs=args.epochs, batch_size=args.nn_batch_size, device=args.device)
        preds_train = trainer.fit(x_tr, y_tr)
        preds_val = trainer.predict(x_val)

        acc_tr = accuracy_fn(preds_train, y_tr)
        f1_tr = macrof1_fn(preds_train, y_tr)
        acc_val = accuracy_fn(preds_val, y_val)
        f1_val = macrof1_fn(preds_val, y_val)

        accs.append(acc_val)
        f1s.append(f1_val)

        print(f"Train      : accuracy = {acc_tr:.3f}% | F1 = {f1_tr:.6f}")
        print(f"Validation : accuracy = {acc_val:.3f}% | F1 = {f1_val:.6f}")

    best_idx = int(np.argmax(accs))
    print("\n==== Best fold result ====")
    print(f"Best Accuracy : {accs[best_idx]:.3f}%")
    print(f"Best F1-score : {f1s[best_idx]:.6f}")
    print(f"Found at fold {best_idx+1}/3")

    if args.test:
        mu = xtrain.mean(axis=0, keepdims=True)
        sigma = xtrain.std(axis=0, keepdims=True) + 1e-8
        xtest = normalize_fn(xtest, mu, sigma)

        if args.nn_type == "cnn":
            from src.methods.deep_network import CNN
            xtest = xtest.reshape(-1, 28, 28, 3).transpose(0, 3, 1, 2)
            model = CNN(input_channels=3, n_classes=n_classes)
        else:
            model = MLP(input_size=2352, n_classes=n_classes)

        summary(model)
        trainer = Trainer(model, lr=args.lr, epochs=args.epochs, batch_size=args.nn_batch_size, device=args.device)
        trainer.fit(xtrain, ytrain)
        preds_val = trainer.predict(xtest)

        np.save("dermamnist_test_preds.npy", preds_val)
        print("Test predictions saved to dermamnist_test_preds.npy")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default="dataset", type=str)
    parser.add_argument('--nn_type', default="mlp", choices=["mlp", "cnn"])
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--nn_batch_size', type=int, default=64)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--use_aug', action="store_true", help="Use data augmentation")
    parser.add_argument('--weight_decay', type=float, default=1e-3, help="Weight decay (L2 regularization)")


    args = parser.parse_args()
    main(args)