from pathlib import Path
import json
from collections import defaultdict

class LossRecorderCallback:
    def __init__(self, save_path: Path):
        self.save_path = Path(save_path)  # ensure it's a Path object
        self.history = defaultdict(list)

    def on_train_begin(self, trainer=None):
        pass

    def on_epoch_end(self, epoch, train_loss, val_loss=None, trainer=None, **kwargs):
        self.history['train_loss'].append(train_loss)
        if val_loss is not None:
            self.history['val_loss'].append(val_loss)

    def on_train_end(self, trainer=None):
        self.save_path.parent.mkdir(parents=True, exist_ok=True)  # make sure directory exists
        with self.save_path.open("w") as f:
            json.dump(self.history, f, indent=2)

