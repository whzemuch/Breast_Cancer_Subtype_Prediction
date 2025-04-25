import matplotlib.pyplot as plt
import numpy as np

def plot_saliency_radar(class_saliency, top_n=10, gene_names=None, id_to_type=None,
                        return_data=False, ncol=None, figsize=None):
    """
    Create radar plots of saliency values for each class's top-N features.
    Maintains consistent radar chart shape and handles subplot arrangement properly.

    Args:
        class_saliency: {class_id: saliency_scores}
        top_n: Number of top features to display
        gene_names: List of gene names for labeling
        id_to_type: Mapping from class IDs to subtype names
        return_data: If True, returns top genes dictionary
        ncol: Number of columns for subplot arrangement
        figsize: Overall figure size (width, height) in inches
    """
    num_classes = len(class_saliency)
    ncol = min(ncol or num_classes, num_classes)
    nrow = int(np.ceil(num_classes / ncol))

    # Set default figure size with good proportions
    if figsize is None:
        fig_width = 5 * ncol + 1  # Extra space for labels
        fig_height = 5 * nrow + 1
        figsize = (fig_width, fig_height)

    fig = plt.figure(figsize=figsize)
    axs = []

    colors = plt.cm.tab10(np.linspace(0, 1, num_classes))
    top_genes_by_class = {}

    for i, (cls, sal) in enumerate(sorted(class_saliency.items())):
        # Create subplot with proper position
        ax = fig.add_subplot(nrow, ncol, i + 1, polar=True)
        axs.append(ax)

        # Get top genes and values
        top_indices = sal.argsort()[-top_n:][::-1]
        top_genes = [gene_names[idx] if gene_names else f"Gene_{idx}" for idx in top_indices]
        class_name = id_to_type[cls] if id_to_type else f"Class {cls}"
        top_genes_by_class[class_name] = top_genes

        # Prepare radar plot data
        values = sal[top_indices]
        angles = np.linspace(0, 2 * np.pi, top_n, endpoint=False)
        values = np.concatenate((values, [values[0]]))
        angles = np.concatenate((angles, [angles[0]]))

        # Plot with consistent scaling
        ax.plot(angles, values, color=colors[i])
        ax.fill(angles, values, color=colors[i], alpha=0.25)
        ax.set_title(class_name, pad=20)
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(top_genes, fontsize=8)
        ax.set_yticklabels([])
        ax.set_rlabel_position(0)
        ax.grid(True)
        ax.set_aspect('equal')

    # Adjust layout without tight_layout
    plt.subplots_adjust(wspace=0.4, hspace=-0.4,
                        left=0.15, right=0.95,
                        bottom=0.15, top=0.95)

    plt.show()

    return top_genes_by_class if return_data else None