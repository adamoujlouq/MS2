import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np


class MLP(nn.Module):
    def __init__(self, input_size, n_classes, hidden_layers=None, dropout_prob=0.3):
        super().__init__()
        if hidden_layers is None:
            hidden_layers = [512, 256, 128, 64]  # Plus profond mais plus régulier

        self.layers = nn.ModuleList()
        self.batchnorms = nn.ModuleList()
        self.dropout = nn.Dropout(dropout_prob)

        inp = input_size
        for dim in hidden_layers:
            self.layers.append(nn.Linear(inp, dim))
            self.batchnorms.append(nn.BatchNorm1d(dim))
            inp = dim

        self.output_layer = nn.Linear(inp, n_classes)

    def forward(self, x):
        for layer, bn in zip(self.layers, self.batchnorms):
            x = F.relu(bn(layer(x)))
            x = self.dropout(x)
        return self.output_layer(x)

class CNN(nn.Module):
   

    def __init__(self, input_channels, n_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.3)

        self.flattened_size = 64 * 7 * 7
        
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)
        x = self.dropout(x)


        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool2(x)
        x = self.dropout(x)


        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        return self.fc2(x)

class Trainer:
    def __init__(self, model, lr=2e-4, epochs=100, batch_size=64, device="cpu", class_weights=None):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.batch_size = batch_size
        self.device = device

        if class_weights is not None:
            self.criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=self.lr,
            weight_decay=1e-3  # forte régularisation
        )

        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=20, gamma=0.5)

    def train_all(self, dataloader):
        self.model.train()
        for epoch in range(1, self.epochs + 1):
            total_loss = 0.0
            for x_batch, y_batch in dataloader:
                x_batch, y_batch = x_batch.to(self.device), y_batch.to(self.device)
                self.optimizer.zero_grad()
                logits = self.model(x_batch)
                loss = self.criterion(logits, y_batch)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()

    def fit(self, training_data, training_labels):
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels).long())
        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        self.train_all(train_loader)
        return self.predict(training_data)

    def predict(self, test_data):
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)
        return self.predict_torch(test_loader)

    def predict_torch(self, dataloader):
        self.model.eval()
        all_preds = []
        with torch.no_grad():
            for x_batch in dataloader:
                if isinstance(x_batch, (list, tuple)):
                    x_batch = x_batch[0]
                x_batch = x_batch.to(self.device)
                logits = self.model(x_batch)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds.cpu())
        return torch.cat(all_preds).numpy()