import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

## MS2


class MLP(nn.Module):
    """
    An MLP network which does classification.

    It should not use any convolutional layers.
    """

    def __init__(self, input_size, n_classes, hidden_dim=128):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_size, n_classes, my_arg=32)

        Arguments:
            input_size (int): size of the input
            n_classes (int): number of classes to predict
            hidden_dim (int, default=128): size of the hidden layer
        """
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, n_classes)

        

       

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, D)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
        """
        x = F.relu(self.fc1(x))
        preds = self.fc2(x)
        return preds


class CNN(nn.Module):
    """
    A CNN which does classification.

    It should use at least one convolutional layer.
    """

    def __init__(self, input_channels, n_classes):
        """
        Initialize the network.

        You can add arguments if you want, but WITH a default value, e.g.:
            __init__(self, input_channels, n_classes, my_arg=32)

        Arguments:
            input_channels (int): number of channels in the input
            n_classes (int): number of classes to predict
        """
        super().__init__()
    
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)  
        self.pool1 = nn.MaxPool2d(2, 2)                                       

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)             
        self.pool2 = nn.MaxPool2d(2, 2)                                       

        self.flattened_size = 64 * 7 * 7
        self.fc1 = nn.Linear(self.flattened_size, 128)
        self.fc2 = nn.Linear(128, n_classes)

    def forward(self, x):
        """
        Predict the class of a batch of samples with the model.

        Arguments:
            x (tensor): input batch of shape (N, Ch, H, W)
        Returns:
            preds (tensor): logits of predictions of shape (N, C)
                Reminder: logits are value pre-softmax.
        """
        x = F.relu(self.conv1(x))     
        x = self.pool1(x)            

        x = F.relu(self.conv2(x))    
        x = self.pool2(x)             

        x = x.view(x.size(0), -1)    
        x = F.relu(self.fc1(x))       
        preds = self.fc2(x)           
        return preds


class Trainer(object):
    """
    Trainer class for the deep networks.
    It will also serve as an interface between numpy and pytorch.
    """

    def __init__(self, model, lr, epochs, batch_size):
        self.lr = lr
        self.epochs = epochs
        self.model = model
        self.batch_size = batch_size

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

    def train_all(self, dataloader):
        for ep in range(self.epochs):
            self.train_one_epoch(dataloader)

    def train_one_epoch(self, dataloader):
        self.model.train()
        for x_batch, y_batch in dataloader:
            self.optimizer.zero_grad()
            logits = self.model(x_batch)
            loss = self.criterion(logits, y_batch.long())
            loss.backward()
            self.optimizer.step()

    def predict_torch(self, dataloader):
        self.model.eval()
        all_preds = []

        with torch.no_grad():
            for batch in dataloader:
                x = batch[0]
                logits = self.model(x)
                preds = torch.argmax(logits, dim=1)
                all_preds.append(preds)

        pred_labels = torch.cat(all_preds)
        return pred_labels

    def fit(self, training_data, training_labels):
        train_dataset = TensorDataset(torch.from_numpy(training_data).float(),
                                      torch.from_numpy(training_labels))
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        self.train_all(train_dataloader)

        return self.predict(training_data)

    def predict(self, test_data):
        test_dataset = TensorDataset(torch.from_numpy(test_data).float())
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False)

        pred_labels = self.predict_torch(test_dataloader)
        return pred_labels.cpu().numpy()