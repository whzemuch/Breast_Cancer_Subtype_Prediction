import pickle
import torch
from pathlib import Path
from sklearn.manifold import TSNE


class TSNERecorderCallback:
    def __init__(self, val_loader, device='cpu', save_path="tsne_results.pkl", perplexity=30):
        self.val_loader = val_loader
        self.device = device
        self.save_path = Path(save_path)
        self.perplexity = perplexity
        self.results = []

    def on_train_begin(self, trainer=None):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

    def extract_fused_features(self, model, batch_data):
        """Extracts fused features from MultimodalFusion"""
        with torch.no_grad():
            # Get individual features
            z_methyl = model.methy_encoder(batch_data['methyl'])
            z_mirna = model.mirna_vae(batch_data['mirna'])[0]  # Get z, ignore mu/logvar
            z_rna = model.rna_vae(batch_data['rna'])[0]

            # Get fused representation
            return model.fusion([z_methyl, z_mirna, z_rna])  # [B, D]

    def on_epoch_end(self, epoch, model=None, trainer=None, **kwargs):
        features, labels = [], []
        model.eval()

        for batch_data, batch_labels in self.val_loader:
            batch_data = {k: v.to(self.device) for k, v in batch_data.items()}
            batch_labels = batch_labels.to(self.device)

            # Get fused features
            fused = self.extract_fused_features(model, batch_data)
            features.append(fused.cpu())
            labels.append(batch_labels.cpu())

        # Convert to numpy
        features = torch.cat(features).numpy()
        labels = torch.cat(labels).numpy()

        # Compute TSNE
        tsne = TSNE(n_components=2, perplexity=self.perplexity, random_state=42)
        embeddings = tsne.fit_transform(features)

        self.results.append({
            'epoch': epoch,
            'embeddings': embeddings,
            'labels': labels
        })

    def on_train_end(self, trainer=None):
        with self.save_path.open('wb') as f:
            pickle.dump(self.results, f)
