from pathlib import Path
import time
from typing import Dict, Optional, Tuple
import torch
from torch.utils.data import DataLoader

class TrainerCore:
    """Handles core training functionality and state management"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        device: torch.device,
        config: Dict
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.config = config
        self.current_epoch = 0

class Checkpointer:
    """Handles model checkpoint saving/loading"""
    
    def __init__(self, checkpoint_dir: str = "./checkpoints"):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.best_metric = 0.0

    def save_checkpoint(self, trainer: TrainerCore, metric: float):
        timestamp = time.strftime("%Y%m%d-%H%M%S")
        filename = f"checkpoint_epoch{trainer.current_epoch}_{timestamp}.pth"
        path = self.checkpoint_dir / filename
        
        torch.save({
            'epoch': trainer.current_epoch,
            'model_state_dict': trainer.model.state_dict(),
            'optimizer_state_dict': trainer.optimizer.state_dict(),
            'metric': metric,
            'config': trainer.config
        }, path)
        return path

class MetricsTracker:
    """Tracks and manages training/validation metrics"""
    
    def __init__(self):
        self.metrics = {
            'train_loss': AverageMeter(),
            'val_loss': AverageMeter(),
            'val_acc': AverageMeter()
        }
        
    def update(self, metric_name: str, value: float, n: int = 1):
        self.metrics[metric_name].update(value, n)
        
    def reset(self):
        for meter in self.metrics.values():
            meter.reset()

class MultiOmicsTrainer:
    """Main trainer class for multi-omics models"""
    
    def __init__(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        criterion: torch.nn.Module,
        config: Dict,
        device: Optional[torch.device] = None,
        checkpoint_dir: str = "./checkpoints"
    ):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.core = TrainerCore(model, optimizer, criterion, self.device, config)
        self.checkpointer = Checkpointer(checkpoint_dir)
        self.metrics = MetricsTracker()
        self.data_loaders = {}

    def prepare_data(self, loaders: Dict[str, DataLoader]):
        """Register data loaders for different phases"""
        self.data_loaders.update(loaders)

    def train(self, num_epochs: int):
        """Main training loop"""
        for epoch in range(num_epochs):
            self.core.current_epoch = epoch
            self._train_epoch()
            
            if 'val' in self.data_loaders:
                self._validate_epoch()
                
            self._log_progress()
            self._save_checkpoint()
            
    def _train_epoch(self):
        """Single training epoch"""
        self.core.model.train()
        self.metrics.reset()
        
        for batch in self.data_loaders['train']:
            inputs = self._prepare_batch(batch)
            outputs = self.core.model(*inputs.values())
            loss = self.core.criterion(outputs, batch['label'].to(self.device))
            
            self.core.optimizer.zero_grad()
            loss.backward()
            self.core.optimizer.step()
            
            self.metrics.update('train_loss', loss.item(), batch['label'].size(0))

    def _validate_epoch(self):
        """Validation phase"""
        self.core.model.eval()
        self.metrics.reset()
        
        with torch.no_grad():
            for batch in self.data_loaders['val']:
                inputs = self._prepare_batch(batch)
                outputs = self.core.model(*inputs.values())
                loss = self.core.criterion(outputs, batch['label'].to(self.device))
                
                # Calculate accuracy
                _, preds = torch.max(outputs, 1)
                correct = (preds == batch['label'].to(self.device)).sum().item()
                
                self.metrics.update('val_loss', loss.item(), batch['label'].size(0))
                self.metrics.update('val_acc', correct / batch['label'].size(0), 1)

    def _prepare_batch(self, batch: Dict) -> Dict:
        """Move batch data to appropriate device"""
        return {k: v.to(self.device) for k, v in batch.items() if k != 'label'}

    def _log_progress(self):
        """Format and print training progress"""
        log_str = [
            f"Epoch {self.core.current_epoch + 1}/{self.core.config['n_epochs']}",
            f"Train Loss: {self.metrics.metrics['train_loss'].avg:.4f}"
        ]
        
        if 'val' in self.data_loaders:
            log_str.extend([
                f"Val Loss: {self.metrics.metrics['val_loss'].avg:.4f}",
                f"Val Acc: {self.metrics.metrics['val_acc'].avg:.2%}"
            ])
            
        print(" | ".join(log_str))

    def _save_checkpoint(self):
        """Save model checkpoint based on validation performance"""
        if self.metrics.metrics['val_acc'].avg > self.checkpointer.best_metric:
            self.checkpointer.best_metric = self.metrics.metrics['val_acc'].avg
            path = self.checkpointer.save_checkpoint(self.core, self.checkpointer.best_metric)
            print(f"Saved best checkpoint to {path}")

    def test(self) -> float:
        """Run model evaluation on test set"""
        if 'test' not in self.data_loaders:
            raise ValueError("No test loader registered")
            
        self.core.model.eval()
        test_acc = AverageMeter()
        
        with torch.no_grad():
            for batch in self.data_loaders['test']:
                inputs = self._prepare_batch(batch)
                outputs = self.core.model(*inputs.values())
                _, preds = torch.max(outputs, 1)
                correct = (preds == batch['label'].to(self.device)).sum().item()
                test_acc.update(correct / batch['label'].size(0), 1)
                
        print(f"\nFinal Test Accuracy: {test_acc.avg:.2%}")
        return test_acc.avg

class AverageMeter:
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val: float, n: int = 1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count