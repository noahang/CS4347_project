import os
import numpy as np
import torch
from torch.utils.data import Dataset

class CQTModeDataset(Dataset):
    """Dataset for pairing precomputed CQT features with mode annotations."""

    MODES = [
        "ionian",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "aeolian",
        "locrian",
    ]

    def __init__(self, root_dir, sr=44100):
        """
        Args:
            root_dir (str): Path to 'train' directory containing 'cqts/' and 'labels/'.
        """
        self.cqt_dir = os.path.join(root_dir, "cqts")
        self.label_dir = os.path.join(root_dir, "labels")
        self.sr = sr
        self.mode_to_idx = {m: i for i, m in enumerate(self.MODES)}

        self.samples = []
        for file in os.listdir(self.cqt_dir):
            if file.endswith("_cqt.npy"):
                base = file.replace("_cqt.npy", "")
                label_path = os.path.join(self.label_dir, base + ".txt")
                if os.path.exists(label_path):
                    self.samples.append((os.path.join(self.cqt_dir, file), label_path))
                else:
                    print(f"⚠️ No label found for {base}, skipping.")

    def __len__(self):
        return len(self.samples)

    def one_hot_mode(self, mode):
        vec = torch.zeros(len(self.MODES), dtype=torch.float32)
        if mode in self.mode_to_idx:
            vec[self.mode_to_idx[mode]] = 1.0
        else:
            print(f"⚠️ Unknown mode '{mode}' — returning zero vector.")
        return vec

    def read_mode(self, label_path):
        with open(label_path, "r", encoding="utf-8") as f:
            text = f.readline().strip()
        # e.g. "A dorian" → "dorian"
        parts = text.split()
        return parts[-1].lower()

    def __getitem__(self, idx):
        cqt_path, label_path = self.samples[idx]
        cqts = np.load(cqt_path)
        cqts = torch.tensor(cqts, dtype=torch.float32)

        mode = self.read_mode(label_path)
        label_vec = self.one_hot_mode(mode)

        return cqts, label_vec