# trainer/base_trainer.py

import torch
from tqdm import tqdm

class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu'):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc="Training")

        for batch_data, batch_labels in pbar:
            # Move to device
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            batch_labels = batch_labels.to(self.device)

            # Forward + loss
            outputs = self.model(**batch_data)
            loss_dict = self.loss_fn(outputs, batch_labels)

            # Backprop
            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            self.optimizer.step()

            total_loss += loss_dict['total'].item()
            pbar.set_postfix(loss=loss_dict['total'].item())

        return total_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for batch_data, batch_labels in tqdm(dataloader, desc="Validation"):
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(**batch_data)
                loss_dict = self.loss_fn(outputs, batch_labels)
                total_loss += loss_dict['total'].item()

        return total_loss / len(dataloader)

    def fit(self, train_loader, val_loader=None, epochs=10):
        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_one_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Val Loss:   {val_loss:.4f}")

    def predict(self, dataloader):
        self.model.eval()
        predictions, targets = [], []

        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = {
                    "methyl": batch_data["methyl"].to(self.device),
                    "mirna": batch_data["mirna"].to(self.device),
                    "rna": batch_data["rna"].to(self.device)
                }
                batch_labels = batch_labels.to(self.device)

                outputs = self.model(**batch_data)
                logits = outputs['logits']
                pred_classes = torch.argmax(logits, dim=1)

                predictions.append(pred_classes.cpu())
                targets.append(batch_labels.cpu())

        predictions = torch.cat(predictions)
        targets = torch.cat(targets)

        return predictions, targets
