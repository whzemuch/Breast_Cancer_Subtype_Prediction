import json
import matplotlib.pyplot as plt
from pathlib import Path

def plot_accuracy(accuracy_path="logs/accuracy_history.json", title="Validation Accuracy", figsize=(6, 4)):
    accuracy_path = Path(accuracy_path)

    if not accuracy_path.exists():
        print(f"❌ File not found: {accuracy_path}")
        return

    with accuracy_path.open("r") as f:
        history = json.load(f)

    if "val_accuracy" not in history:
        print("⚠️ 'val_accuracy' not found in history.")
        return

    val_acc = history["val_accuracy"]
    epochs = list(range(1, len(val_acc) + 1))

    plt.figure(figsize=figsize)
    plt.plot(epochs, val_acc, marker="o", label="Val Accuracy")
    plt.title(title)
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.ylim(0, 1.05)
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
