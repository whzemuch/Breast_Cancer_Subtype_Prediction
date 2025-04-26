from sklearn.metrics import accuracy_score
import torch
import json
from pathlib import Path
from collections import defaultdict

class AccuracyRecorderCallback:
    def __init__(self, save_path: Path = Path("logs/accuracy_history.json")):
        self.save_path = Path(save_path)
        self.history = defaultdict(list)

    def on_train_begin(self, trainer=None):
        pass

    def on_epoch_end(self, epoch, train_loss, val_loss=None, trainer=None, **kwargs):
        # Only compute accuracy if val_loader is available
        if hasattr(trainer, "val_loader") and trainer.val_loader is not None:
            trainer.model.eval()
            preds, targets = [], []

            with torch.no_grad():
                for batch_data, batch_labels in trainer.val_loader:
                    batch_data = {k: v.to(trainer.device) for k, v in batch_data.items()}
                    batch_labels = batch_labels.to(trainer.device)

                    outputs = trainer.model(**batch_data)
                    logits = outputs["logits"]
                    pred_classes = torch.argmax(logits, dim=1)

                    preds.append(pred_classes.cpu())
                    targets.append(batch_labels.cpu())

            preds = torch.cat(preds)
            targets = torch.cat(targets)
            acc = accuracy_score(targets.numpy(), preds.numpy())
            self.history["val_accuracy"].append(acc)

            print(f"✅ Val Accuracy: {acc:.4f}")

    def on_train_end(self, trainer=None):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open("w") as f:
            json.dump(self.history, f, indent=2)
