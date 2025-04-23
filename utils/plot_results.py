




def plot_loss(loss_path="loss_history.json", figsize=(12, 6)):
    import json
    import matplotlib.pyplot as plt

    with open(loss_path) as f:
        history = json.load(f)

    plt.figure(figsize=figsize)
    
    # Get number of epochs from the data
    num_epochs = len(history['train_loss'])
    
    plt.plot(history['train_loss'], label='Train Loss')
    if 'val_loss' in history:
        plt.plot(history['val_loss'], label='Val Loss')
    
    plt.title("Loss Curve")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    # Set correct x-axis limits and ticks
    plt.xlim(0, num_epochs-1)
    plt.xticks(range(0, num_epochs, 5))  # Integer epochs
    
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



def plot_tsne(tsne_path="tsne_results.pkl", epochs=None, cols=3, figsize=(10, 8)):
    import pickle
    import matplotlib.pyplot as plt
    import numpy as np
    import math

    with open(tsne_path, "rb") as f:
        results = pickle.load(f)

    if not results:
        print("No t-SNE results found.")
        return

    # Map class index to PAM50 subtype
    int_to_pam50 = {
        0: 'LumA',
        1: 'LumB',
        2: 'Normal',
        3: 'Basal',
        4: 'Her2'
    }

    # Build a dict for quick epoch lookup
    epoch_map = {r['epoch']: r for r in results}
    available_epochs = sorted(epoch_map.keys())

    # Default to first, middle, last
    if epochs is None:
        epochs = [available_epochs[0],
                  available_epochs[len(available_epochs) // 2],
                  available_epochs[-1]]

    selected = [epoch_map[e] for e in epochs if e in epoch_map]
    n_plots = len(selected)
    rows = math.ceil(n_plots / cols)

    fig, axes = plt.subplots(rows, cols, figsize=(figsize[0] * cols, figsize[1] * rows))
    axes = np.array(axes).reshape(-1)  # Flatten for easy iteration

    handles = []
    labels_seen = set()

    for ax, result in zip(axes, selected):
        emb = result['embeddings']
        labels = result['labels']
        epoch = result['epoch']

        for label in np.unique(labels):
            idx = labels == label
            label_name = int_to_pam50.get(label, f"Class {label}")
            sc = ax.scatter(emb[idx, 0], emb[idx, 1], label=label_name, alpha=0.7)
            if label_name not in labels_seen:
                handles.append(sc)
                labels_seen.add(label_name)

        ax.set_title(f"Epoch {epoch}")
        ax.set_xlabel("t-SNE 1")
        ax.set_ylabel("t-SNE 2")
        ax.grid(True)

    # Remove extra/empty axes
    for j in range(n_plots, len(axes)):
        fig.delaxes(axes[j])

    # Shared legend at the bottom
    fig.legend(handles=handles, loc='lower center', ncol=len(handles),
               bbox_to_anchor=(0.5, -0.05))

    plt.subplots_adjust(bottom=0.25)
    plt.tight_layout()
    plt.show()



