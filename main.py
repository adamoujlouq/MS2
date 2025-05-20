import argparse
import numpy as np
from torchinfo import summary

from src.data import load_data
from src.utils import normalize_fn
from src.methods.deep_network import MLP, CNN, Trainer
from src.utils import accuracy_fn, macrof1_fn, get_n_classes


def main(args):
    # 1. Load data  ────────────────────────────────────────────────────────
    xtrain, xtest, ytrain, y_test = load_data()          # (N,28,28,3) uint8
    xtrain = xtrain.reshape(xtrain.shape[0], -1)         # (N,2352)
    xtest  = xtest.reshape(xtest.shape[0], -1)

    # 2-a. Validation split  ──────────────────────────────────────────────
    if not args.test:
        # === WRITE YOUR CODE HERE (split 90 / 10) =======================
        rng = np.random.default_rng(42)
        idx = rng.permutation(len(xtrain))
        split = int(0.9 * len(idx))
        tr_idx, val_idx = idx[:split], idx[split:]

        x_val, y_val   = xtrain[val_idx], ytrain[val_idx]
        xtrain, ytrain = xtrain[tr_idx], ytrain[tr_idx]
        # ================================================================
    else:                              # test mode → val = test (pour métriques)
        x_val, y_val = xtest, y_test

    # 2-b. Normalisation  ────────────────────────────────────────────────
    # === WRITE YOUR CODE HERE (normalise features vectorisés) ============
    xtrain = xtrain.astype(np.float32) / 255.0
    x_val  = x_val.astype(np.float32)  / 255.0

    mu    = xtrain.mean(axis=0, keepdims=True)
    sigma = xtrain.std(axis=0, keepdims=True) + 1e-8

    xtrain = normalize_fn(xtrain, mu, sigma)
    x_val  = normalize_fn(x_val, mu, sigma)

    # on garde aussi une version (N,28,28,3) pour la CNN
    xtrain_img = xtrain.reshape(-1, 28, 28, 3)
    x_val_img  = x_val.reshape( -1, 28, 28, 3)
    # =====================================================================

    # 3. Initialise le modèle  ────────────────────────────────────────────
    n_classes = get_n_classes(ytrain)
    if args.nn_type == "mlp":
        # === WRITE YOUR CODE HERE (MLP instanciation) ===================
        model = MLP(input_size=2352, n_classes=n_classes, hidden_dim=256)
        # =================================================================
    elif args.nn_type == "cnn":
        # === WRITE YOUR CODE HERE (CNN: remet en CHW) ===================
        xtrain = np.transpose(xtrain_img, (0, 3, 1, 2))
        x_val  = np.transpose(x_val_img,  (0, 3, 1, 2))
        model  = CNN(input_channels=3, n_classes=n_classes)
        # =================================================================
    else:
        raise ValueError("nn_type must be 'mlp' or 'cnn'")

    summary(model)

    # 4. Trainer & apprentissage  ────────────────────────────────────────
    trainer = Trainer(model, lr=args.lr,
                      epochs=args.max_iters,
                      batch_size=args.nn_batch_size)

    preds_train = trainer.fit(xtrain, ytrain)
    preds_val   = trainer.predict(x_val)

    # 5. Métriques  ───────────────────────────────────────────────────────
    acc_tr  = accuracy_fn(preds_train, ytrain)
    f1_tr   = macrof1_fn(preds_train, ytrain)

    acc_val = accuracy_fn(preds_val, y_val)
    f1_val  = macrof1_fn(preds_val, y_val)

    print(f"\nTrain : accuracy = {acc_tr:.3f}% | F1 = {f1_tr:.6f}")
    tag = "Test" if args.test else "Validation"
    print(f"{tag} : accuracy = {acc_val:.3f}% | F1 = {f1_val:.6f}")

    # (option) sauvegarde prédictions test
    if args.test:
        np.save("dermamnist_test_preds.npy", preds_val)
        print("Test predictions saved to dermamnist_test_preds.npy")


# ──────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--data', default="dataset", type=str)
    parser.add_argument('--nn_type', default="mlp", choices=["mlp", "cnn"])
    parser.add_argument('--nn_batch_size', type=int, default=64)
    parser.add_argument('--device', default="cpu")
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--max_iters', type=int, default=20)
    parser.add_argument('--test', action="store_true")

    args = parser.parse_args()
    main(args)
