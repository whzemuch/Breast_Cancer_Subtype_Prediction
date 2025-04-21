import time
import torch
from pathlib import Path
from typing import Dict, Optional
from utils import AverageMeter
from .registry import TrainerRegistry

@TrainerRegistry.register('multiomics_trainer')
class MultiOmicsTrainer:
    def __init__(
            self,
            model: torch.nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: torch.nn.Module,
            device: torch.device,
            config: Dict,
            checkpoint_dir: str = "./checkpoints"
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config

        # Initialize metrics trackers
        self.train_loss = AverageMeter()
        self.val_loss = AverageMeter()
        self.val_acc = AverageMeter()

        # Setup checkpointing
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_val_acc = 0.0

    def train(
            self,
            train_loader: torch.utils.data.DataLoader,
            val_loader: Optional[torch.utils.data.DataLoader] = None,
            num_epochs: int = 10
    ):
        for epoch in range(num_epochs):
            # Train phase
            self._train_epoch(train_loader)

            # Validation phase
            if val_loader:
                self._validate(val_loader)

            # Print progress
            self._log_epoch(epoch)

            # Save checkpoints
            if (epoch + 1) % self.config.get("checkpoint_interval", 1) == 0:
                self._save_checkpoint(epoch)

    def _train_epoch(self, loader: torch.utils.data.DataLoader):
        self.model.train()
        self.train_loss.reset()

        for batch in loader:
            # Move data to device
            methy = batch['methylation'].to(self.device)
            mirna = batch['mirna'].to(self.device)
            rna = batch['rna'].to(self.device)
            labels = batch['label'].to(self.device)

            # Forward pass
            outputs = self.model(methy, mirna, rna)
            loss = self.criterion(outputs, labels)

            # Backward pass and optimize
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # Update metrics
            self.train_loss.update(loss.item(), labels.size(0))

    def _validate(self, loader: torch.utils.data.DataLoader):
        self.model.eval()
        self.val_loss.reset()
        self.val_acc.reset()

        with torch.no_grad():
            for batch in loader:
                methy = batch['methylation'].to(self.device)
                mirna = batch['mirna'].to(self.device)
                rna = batch['rna'].to(self.device)
                labels = batch['label'].to(self.device)

                # Forward pass
                outputs = self.model(methy, mirna, rna)
                loss = self.criterion(outputs, labels)

                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct = (preds == labels).sum().item()

                # Update metrics
                batch_size = labels.size(0)
                self.val_loss.update(loss.item(), batch_size)
                self.val_acc.update(correct / batch_size, batch_size)

        # Update best validation accuracy
        if self.val_acc.avg > self.best_val_acc:
            self.best_val_acc = self.val_acc.avg

    def _log_epoch(self, epoch: int):
        log_str = f"Epoch {epoch + 1}/{self.config['n_epochs']} | "
        log_str += f"Train Loss: {self.train_loss.avg:.4f}"

        if self.val_loss.count > 0:
            log_str += f" | Val Loss: {self.val_loss.avg:.4f}"
            log_str += f" | Val Acc: {self.val_acc.avg:.2%}"

        print(log_str)

    def _save_checkpoint(self, epoch: int):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'train_loss': self.train_loss.avg,
            'val_loss': self.val_loss.avg,
            'val_acc': self.val_acc.avg,
            'config': self.config
        }

        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"checkpoint_epoch{epoch + 1}_{timestamp}.pth"
        path = self.checkpoint_dir / filename
        torch.save(checkpoint, path)
        print(f"Saved checkpoint to {path}")

    def test(self, test_loader: torch.utils.data.DataLoader) -> float:
        self.model.eval()
        test_acc = AverageMeter()

        with torch.no_grad():
            for batch in test_loader:
                methy = batch['methylation'].to(self.device)
                mirna = batch['mirna'].to(self.device)
                rna = batch['rna'].to(self.device)
                labels = batch['label'].to(self.device)

                outputs = self.model(methy, mirna, rna)
                _, preds = torch.max(outputs, 1)
                correct = (preds == labels).sum().item()
                test_acc.update(correct / labels.size(0), labels.size(0))

                print(f"\nTest Accuracy: {test_acc.avg:.2%}")
        return test_acc.avg