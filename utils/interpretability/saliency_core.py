

class SaliencyComputer:
    def __init__(self, model, modality_key, device='cuda'):
        """
        Args:
            model: the trained model
            modality_key: one of 'rna', 'mirna', 'methyl', etc.
        """
        self.model = model
        self.modality_key = modality_key
        self.device = device

    def compute_saliency(self, sample, class_idx=None):
        self.model.eval()
        self.model.zero_grad()

        # Prepare inputs with gradients enabled for target modality
        inputs = {k: v.to(self.device) for k, v in sample.items() if k in ['methyl', 'mirna', 'rna']}
        inputs[self.modality_key] = inputs[self.modality_key].clone().detach().requires_grad_(True)

        output = self.model(**inputs)
        logits = output['logits']

        if class_idx is None:
            class_idx = logits[0].argmax().item()

        assert logits.shape[1] > class_idx, f"Invalid class index {class_idx} for logits shape {logits.shape}"

        logits[0, class_idx].backward()  # ✅ safer than logits[:, class_idx].sum().backward()

        saliency = inputs[self.modality_key].grad.abs().squeeze().detach().cpu().numpy()
        return saliency

