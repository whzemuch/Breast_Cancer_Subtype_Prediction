from .base_trainer  import BaseTrainer
import torch

class CallbackTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, device='cpu', callbacks=None, seed=None):
        super().__init__(model, optimizer, loss_fn, device, seed)
        self.callbacks = callbacks or []
        self.val_loader = None  # Initialize val_loader as None

    def fit(self, train_loader, val_loader=None, epochs=10):
        """Training loop with proper mode handling and deterministic behavior.

        Args:
            train_loader: DataLoader for training data
            val_loader: Optional DataLoader for validation data
            epochs: Number of training epochs
        """
        # Store the validation loader
        self.val_loader = val_loader

    
        # Set deterministic flags at start (critical for reproducibility)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Initialize callbacks
        for cb in self.callbacks:
            cb.on_train_begin(trainer=self)

        try:
            for epoch in range(1, epochs + 1):
                # Training phase
                self.model.train()
                train_loss = self.train_one_epoch(train_loader)
                print(f"Epoch {epoch}/{epochs} | Train Loss: {train_loss:.4f}")

                # Validation phase
                val_loss = None
                if val_loader:
                    self.model.eval()
                    with torch.no_grad():
                        val_loss = self.evaluate(val_loader)
                    print(f"Epoch {epoch}/{epochs} | Val Loss: {val_loss:.4f}")

                # Callbacks
                for cb in self.callbacks:
                    cb.on_epoch_end(
                        epoch=epoch,
                        train_loss=train_loss,
                        val_loss=val_loss,
                        model=self.model,
                        trainer=self
                    )

        finally:
            # Ensure callbacks complete even if training fails
            for cb in self.callbacks:
                cb.on_train_end(trainer=self)