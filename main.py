import argparse
import numpy as np
import random
import torch
import matplotlib.pyplot as plt
from torchinfo import summary

from src.data import load_data
from src.utils import normalize_fn
from src.methods.deep_network import Trainer
from src.utils import accuracy_fn, macrof1_fn, get_n_classes
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, input_size, n_classes, hidden_layers=[128]):
        super().__init__()
        self.layers = nn.ModuleList()
        inp = input_size
        for dim in hidden_layers:
            self.layers.append(nn.Linear(inp, dim))
            inp = dim
        self.output_layer = nn.Linear(inp, n_classes)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.output_layer(x)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def plot_heatmap(data, title, xticks, yticks):
    plt.figure(figsize=(8, 6))
    im = plt.imshow(data, cmap='viridis')
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            plt.text(j, i, f"{data[i,j]:.1f}", ha='center', va='center', color='w', fontsize=8)
    plt.xticks(np.arange(len(xticks)), xticks)
    plt.yticks(np.arange(len(yticks)), yticks)
    plt.xlabel("Hidden Dim")
    plt.ylabel("Epochs")
    plt.title(title)
    plt.colorbar(im)
    plt.tight_layout()
    plt.show()

def main(args):
    set_seed(42)

    xtrain, xtest, ytrain, y_test = load_data()
    xtrain = xtrain.reshape(xtrain.shape[0], -1).astype(np.float32) / 255.0
    xtest = xtest.reshape(xtest.shape[0], -1).astype(np.float32) / 255.0

    rng = np.random.default_rng(42)
    idx = rng.permutation(len(xtrain))
    split = int(0.9 * len(idx))
    tr_idx, val_idx = idx[:split], idx[split:]
    x_val, y_val = xtrain[val_idx], ytrain[val_idx]
    xtrain, ytrain = xtrain[tr_idx], ytrain[tr_idx]

    mu = xtrain.mean(axis=0, keepdims=True)
    sigma = xtrain.std(axis=0, keepdims=True) + 1e-8
    xtrain = normalize_fn(xtrain, mu, sigma)
    x_val = normalize_fn(x_val, mu, sigma)

    xtrain_img = xtrain.reshape(-1, 28, 28, 3)
    x_val_img  = x_val.reshape(-1, 28, 28, 3)

    n_classes = get_n_classes(ytrain)

    if args.nn_type == "cnn":
        from src.methods.deep_network import CNN
        xtrain = np.transpose(xtrain_img, (0, 3, 1, 2))
        x_val  = np.transpose(x_val_img, (0, 3, 1, 2))
        model  = CNN(input_channels=3, n_classes=n_classes)
        summary(model)
        trainer = Trainer(model, lr=args.lr, epochs=args.epochs, batch_size=args.nn_batch_size)
        preds_train = trainer.fit(xtrain, ytrain)
        preds_val   = trainer.predict(x_val)
        acc_tr = accuracy_fn(preds_train, ytrain)
        f1_tr  = macrof1_fn(preds_train, ytrain)
        acc_val = accuracy_fn(preds_val, y_val)
        f1_val  = macrof1_fn(preds_val, y_val)
        print(f"\nTrain : accuracy = {acc_tr:.3f}% | F1 = {f1_tr:.6f}")
        tag = "Test" if args.test else "Validation"
        print(f"{tag} : accuracy = {acc_val:.3f}% | F1 = {f1_val:.6f}")
        if args.test:
            np.save("dermamnist_test_preds.npy", preds_val)
            print("Test predictions saved to dermamnist_test_preds.npy")
        return

    # GRID SEARCH (MLP only)
    epochs_list = [50, 100, 200, 300]
    hidden_dims = [64, 128, 256, 512]
    batch_size = args.nn_batch_size
    lr = args.lr

    train_accs = np.zeros((len(epochs_list), len(hidden_dims)))
    val_accs = np.zeros((len(epochs_list), len(hidden_dims)))
    train_f1s = np.zeros((len(epochs_list), len(hidden_dims)))
    val_f1s = np.zeros((len(epochs_list), len(hidden_dims)))

    for i, epoch in enumerate(epochs_list):
        for j, hdim in enumerate(hidden_dims):
            print(f"\n[INFO] Training MLP with hidden_layers=[{hdim}], epochs={epoch}")
            model = MLP(input_size=2352, n_classes=n_classes, hidden_layers=[hdim])
            trainer = Trainer(model, lr=lr, epochs=epoch, batch_size=batch_size)

            preds_train = trainer.fit(xtrain, ytrain)
            preds_val = trainer.predict(x_val)

            train_accs[i, j] = accuracy_fn(preds_train, ytrain)
            val_accs[i, j] = accuracy_fn(preds_val, y_val)
            train_f1s[i, j] = macrof1_fn(preds_train, ytrain)
            val_f1s[i, j] = macrof1_fn(preds_val, y_val)

    plot_heatmap(train_accs, "Train Accuracy (%)", hidden_dims, epochs_list)
    plot_heatmap(val_accs, "Validation Accuracy (%)", hidden_dims, epochs_list)
    plot_heatmap(train_f1s, "Train F1-score (%)", hidden_dims, epochs_list)
    plot_heatmap(val_f1s, "Validation F1-score (%)", hidden_dims, epochs_list)

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
