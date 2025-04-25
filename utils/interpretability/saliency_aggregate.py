from collections import defaultdict
from .saliency_core import SaliencyComputer

def compute_average_saliency_by_class(model, dataloader, modality_key, max_per_class=50, num_classes=None, device='cuda'):
    model.eval()
    saliency_sums = defaultdict(lambda: None)
    counts = defaultdict(int)
    sal_computer = SaliencyComputer(model, modality_key, device)

    for batch in dataloader:
        # ✅ Support tuple-style batches: (inputs_dict, labels_tensor)
        if isinstance(batch, (list, tuple)) and isinstance(batch[0], dict):
            inputs, labels = batch
        else:
            raise ValueError("Batch must be (dict of tensors, labels tensor) format.")

        labels = labels.to(device)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        batch_size = labels.size(0)

        for i in range(batch_size):
            cls = labels[i].item()
            if counts[cls] >= max_per_class:
                continue

            sample = {k: v[i:i+1] for k, v in inputs.items()}
            sal = sal_computer.compute_saliency(sample)

            if saliency_sums[cls] is None:
                saliency_sums[cls] = sal
            else:
                saliency_sums[cls] += sal

            counts[cls] += 1

    return {cls: saliency_sums[cls] / counts[cls] for cls in saliency_sums}

