import pickle
from pathlib import Path

class AttentionLoggerCallback:
    def __init__(self, save_path="logs/attention_weights.pkl", modality_names=None, log_every=5):
        self.save_path = Path(save_path)
        self.log_every = log_every
        self.modality_names = modality_names or ["methy", "mirna", "rna"]
        self.results = []

    def on_train_begin(self, trainer):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def on_epoch_end(self, epoch, train_loss, val_loss, model, trainer=None, **kwargs):
        if epoch % self.log_every != 0:
            return
        print(f"🚩 End of epoch {epoch} — Logging attention")

        # Access the fusion submodule
        fusion_module = getattr(model, "fusion", None)
        if fusion_module is None:
            print("⚠️ fusion submodule not found")
            return

        # Get attention weights from fusion module
        attn_weights = getattr(fusion_module, "attn_weights", None)
        # print(f"[Epoch {epoch}] fusion.attn_weights: {attn_weights}"
        if attn_weights is None:
            return

        # Extract sample from first batch item (all heads)
        sample_attn = attn_weights[0].detach().cpu().numpy()  # [Heads=4, 3, 3]

        # Wrap in a list (1 entry per layer; you have 1 layer)
        extracted = [sample_attn]

        self.results.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "modality_names": self.modality_names,
            "attn": extracted  # Length 1 (1 layer)
        })

    def on_train_end(self, trainer=None):
        with self.save_path.open("wb") as f:
            pickle.dump(self.results, f)
        print(f"📦 Saved attention log to: {self.save_path}")




