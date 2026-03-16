import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from tqdm import tqdm
import pandas as pd
import wandb

class DigitsClassifier(nn.Module):
    def __init__(self, in_ch, out_ch, kernel_size, stride, padding):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, 32, kernel_size, stride, padding),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(32),
            nn.Flatten(),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        )

    def forward(self, input) -> torch.Tensor:
        y = self.net(input)
        return y.logit()
    
class Trainer():
    def __init__(self, in_ch:int, out_ch:int, criterion=None, metrics=None, lr:float=0.003, batch_size:int=64, epochs:int=1000, max_grad_norm:float=1.0, kernel_size:int=3, stride:int=1, padding:int=1):
        self.model = DigitsClassifier(in_ch, out_ch, kernel_size, stride, padding)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)

        self.batch_size = batch_size
        self.epochs = epochs
        self.max_grad_norm = max_grad_norm
        if criterion is None:
            self.criterion = nn.CrossEntropyLoss()
        else:
            self.criterion = criterion()
        if metrics is None:
            self.metrics = []


        wandb.init(project="mnist classifier", config={
            "epochs": epochs,
            "learning_rate": lr,
            "criterion": criterion,
            "batch_size": batch_size,
            "in_channels": in_ch,
            "out_channels": out_ch,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "max_grad_norm": max_grad_norm,
        })
    
    def learn(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.train()
        X_train = X.to_numpy()
        y_train = y.to_numpy()
        for epoch in tqdm(range(self.epochs)):
            idx = np.random.choice(X_train.shape[1], self.batch_size)
            X_batch = X_train[idx]
            y_batch = y_train[idx]
            y_pred = self.model.forward(X_batch)
            loss = self.criterion(y_pred, y_batch)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
            self.optimizer.step()

            accuracy = accuracy_score(y_train, y_pred)
            recall = recall_score(y_train, y_pred)
            f1 = f1_score(y_train, y_pred)
            precision = precision_score(y_train, y_pred)

            wandb.log(data={
                "loss": loss.item(),
                "accuracy": accuracy,
                "recall": recall,
                "f1_score": f1,
                "precision": precision,
            }, step=epoch)

    def evaluate(self, X: pd.DataFrame, y: pd.DataFrame):
        self.model.eval()
        X_test = X.to_numpy()
        y_test = y.to_numpy()
        y_pred = self.model.forward(X_test)

        loss = self.criterion(y_pred, y_test)
        
        report = classification_report(y_test, y_pred)
        return loss, report

    def save(self):
        torch.save(self.model.state_dict(), "model/model.pth")
        wandb.save("model/model.pth")

    def load(self, path):
        model = torch.load(path)
        return model
    
