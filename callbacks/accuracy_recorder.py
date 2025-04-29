from sklearn.metrics import accuracy_score
import torch
import json
from pathlib import Path
from collections import defaultdict


class AccuracyRecorderCallback:
    def __init__(self, save_path: Path = Path("logs/accuracy_history.json"), debug=False):
        self.save_path = Path(save_path)
        self.history = defaultdict(list)
        self.debug = debug

    def on_train_begin(self, trainer=None):
        if self.debug:
            print("🔍 [Debug] AccuracyRecorderCallback initialized")

    def on_epoch_end(self, epoch, train_loss, val_loss=None, trainer=None, **kwargs):
        if not hasattr(trainer, "val_loader") or trainer.val_loader is None:
            if self.debug:
                print("⚠️ No validation loader found - skipping accuracy calculation")
            return

        trainer.model.eval()
        preds, targets = [], []

        with torch.no_grad():
            for i, batch in enumerate(trainer.val_loader):
                try:
                    # Debug: Print batch structure
                    if self.debug and i == 0:
                        print(f"\n🔍 [Debug] Batch {i} structure:")
                        print(f"Batch type: {type(batch)}")
                        if isinstance(batch, (tuple, list)):
                            print(f"Batch length: {len(batch)}")
                            print(f"First element type: {type(batch[0])}")
                            if isinstance(batch[0], dict):
                                print(f"Data keys: {batch[0].keys()}")
                                for k, v in batch[0].items():
                                    print(f"{k} shape: {v.shape if hasattr(v, 'shape') else 'N/A'}")
                            print(f"Labels shape: {batch[1].shape if hasattr(batch[1], 'shape') else 'N/A'}")

                    # Unpack batch
                    if isinstance(batch, (tuple, list)):
                        batch_data, batch_labels = batch
                    else:
                        raise ValueError("Batch must be a tuple/list of (data, labels)")

                    # Move data to device
                    if not isinstance(batch_data, dict):
                        raise ValueError("Batch data must be a dictionary of modalities")

                    # Debug: Print shapes before processing
                    if self.debug and i == 0:
                        print("\n🔍 [Debug] Input shapes:")
                        for k, v in batch_data.items():
                            print(f"{k}: {v.shape}")

                    # Process each modality
                    methyl = batch_data['methyl'].to(trainer.device)
                    mirna = batch_data['mirna'].to(trainer.device)
                    rna = batch_data['rna'].to(trainer.device)

                    # Get model outputs
                    outputs = trainer.model(methyl=methyl, mirna=mirna, rna=rna)

                    # Debug: Print model outputs
                    if self.debug and i == 0:
                        print("\n🔍 [Debug] Model outputs:")
                        print(f"Output keys: {outputs.keys()}")
                        print(f"Logits shape: {outputs['logits'].shape}")

                    # Get predictions
                    logits = outputs['logits']
                    pred_classes = torch.argmax(logits, dim=1)

                    preds.append(pred_classes.cpu())
                    targets.append(batch_labels.cpu())

                    # Debug: Print first prediction
                    if self.debug and i == 0:
                        print(
                            f"\n🔍 [Debug] First prediction: {pred_classes[0].item()} (True: {batch_labels[0].item()})")

                except Exception as e:
                    print(f"❌ Error processing batch {i}: {str(e)}")
                    if self.debug:
                        import traceback
                        traceback.print_exc()
                    raise

        if preds:
            preds = torch.cat(preds)
            targets = torch.cat(targets)

            # Debug: Print final shapes
            if self.debug:
                print(f"\n🔍 [Debug] Final shapes - Preds: {preds.shape}, Targets: {targets.shape}")
                print(f"Unique predictions: {torch.unique(preds)}")
                print(f"Unique targets: {torch.unique(targets)}")

            acc = accuracy_score(targets.numpy(), preds.numpy())
            self.history["val_accuracy"].append(acc)
            print(f"✅ Val Accuracy: {acc:.4f}")
        else:
            print("⚠️ No predictions were generated")

    def on_train_end(self, trainer=None):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        with self.save_path.open("w") as f:
            json.dump(self.history, f, indent=2)
        if self.debug:
            print(f"💾 Saved accuracy history to {self.save_path}")
