from .base_trainer  import BaseTrainer


class CallbackTrainer(BaseTrainer):
    def __init__(self, model, optimizer, loss_fn, device='cpu', callbacks=None, seed=None):
        super().__init__(model, optimizer, loss_fn, device, seed)
        self.callbacks = callbacks or []

    def fit(self, train_loader, val_loader=None, epochs=10):
        self.val_loader = val_loader

        for cb in self.callbacks:
            cb.on_train_begin(trainer=self)

        for epoch in range(1, epochs + 1):
            print(f"\nEpoch {epoch}/{epochs}")
            train_loss = self.train_one_epoch(train_loader)
            print(f"Train Loss: {train_loss:.4f}")

            val_loss = None
            if val_loader:
                val_loss = self.evaluate(val_loader)
                print(f"Val Loss:   {val_loss:.4f}")

            for cb in self.callbacks:
                cb.on_epoch_end(
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    model=self.model,
                    trainer=self
                )

        for cb in self.callbacks:
            cb.on_train_end(trainer=self)
