class EarlyStoppingCallback:
    def __init__(self, patience=5):
        self.best = float("inf")
        self.counter = 0
        self.patience = patience

    def on_train_begin(self, trainer=None):
        self.best = float("inf")
        self.counter = 0

    def on_epoch_end(self, epoch, val_loss=None, trainer=None, **kwargs):
        if val_loss is None:
            return
        if val_loss < self.best:
            self.best = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("⏹️ Early stopping triggered.")
                raise StopIteration()

    def on_train_end(self, trainer=None):
        print(f"Training stopped early at best val loss: {self.best:.4f}")

