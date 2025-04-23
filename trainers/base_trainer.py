# trainer/base_trainer.py

import torch
from tqdm import tqdm
from utils import set_seed  

class BaseTrainer:
    def __init__(self, model, optimizer, loss_fn, device='cpu', seed=None):
        if seed is not None:
            set_seed(seed)
        self.model = model.to(device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.device = device
        

    def train_one_epoch(self, dataloader):
        self.model.train()
        total_loss = 0
        pbar = tqdm(dataloader, desc="Training")

        for batch_data, batch_labels in pbar:
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            batch_labels = batch_labels.to(self.device)

            outputs = self.model(**batch_data)
            loss_dict = self.loss_fn(outputs, batch_labels)

            self.optimizer.zero_grad()
            loss_dict['total'].backward()
            self.optimizer.step()

            self.optimizer.step()

            if self.loss_fn.current_step.item() % 10 == 0:
                print(f"step {self.loss_fn.current_step.item():5d} | "
                      f"loss: {loss_dict['total'].item():.4f} | "
                      f"ce: {loss_dict['ce'].item():.4f} | "
                      f"kl: {loss_dict['kl_total'].item():.4f} | "
                      f"beta: {loss_dict['beta'].item():.4f}")


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

    def fit(self, train_loader, val_loader=None, epochs=10, log_interval=1):
        for epoch in range(1, epochs + 1):
            train_loss = self.train_one_epoch(train_loader)
    
            if val_loader:
                val_loss = self.evaluate(val_loader)
            else:
                val_loss = None
    
            # ✅ Print train/val loss every `log_interval` epochs
            if epoch % log_interval == 0 or epoch == 1 or epoch == epochs:
                print(f"\n📆 Epoch {epoch}/{epochs}")
                print(f"🧮 Train Loss: {train_loss:.4f}")
                if val_loader:
                    print(f"🧪 Val Loss:   {val_loss:.4f}")


    def predict(self, dataloader, return_logits=False):
        self.model.eval()
        predictions, targets = [], []
        logits_all = [] if return_logits else None
    
        with torch.no_grad():
            for batch_data, batch_labels in dataloader:
                batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
                batch_labels = batch_labels.to(self.device)
    
                outputs = self.model(**batch_data)
                logits = outputs['logits']
                pred_classes = torch.argmax(logits, dim=1)
    
                predictions.append(pred_classes.cpu())
                targets.append(batch_labels.cpu())
    
                if return_logits:
                    logits_all.append(logits.cpu())
    
        predictions = torch.cat(predictions)
        targets = torch.cat(targets)
    
        if return_logits:
            return predictions, targets, torch.cat(logits_all)
    
        return predictions, targets


