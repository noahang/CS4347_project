import os
import librosa
import torch
from torch.utils.data import Dataset

class MP3ModeDataset(Dataset):
    """Dataset for test data — loads audio + mode labels."""

    MODES = [
        "ionian",
        "dorian",
        "phrygian",
        "lydian",
        "mixolydian",
        "aeolian",
        "locrian",
    ]

    def __init__(self, root_dir, sr=22050, load_cqt=True, cqt_params=None):
        """
        Args:
            root_dir (str): Path to 'test' directory containing 'audio/' and 'labels/'.
            sr (int): Audio sample rate.
            load_cqt (bool): If True, compute CQT; otherwise return waveform.
            cqt_params (dict): Extra arguments for librosa.cqt().
        """
        self.audio_dir = os.path.join(root_dir, "audio")
        self.label_dir = os.path.join(root_dir, "labels")
        self.sr = sr
        self.load_cqt = load_cqt
        self.cqt_params = cqt_params or {}
        self.mode_to_idx = {m: i for i, m in enumerate(self.MODES)}

        self.samples = []
        for file in os.listdir(self.audio_dir):
            if file.lower().endswith(".mp3"):
                base = os.path.splitext(file)[0]
                label_path = os.path.join(self.label_dir, base + ".txt")
                if os.path.exists(label_path):
                    self.samples.append((os.path.join(self.audio_dir, file), label_path))
                else:
                    print(f"⚠️ No label for {base}, skipping.")

    def __len__(self):
        return len(self.samples)

    def one_hot_mode(self, mode):
        vec = torch.zeros(len(self.MODES), dtype=torch.float32)
        if mode in self.mode_to_idx:
            vec[self.mode_to_idx[mode]] = 1.0
        else:
            print(f"⚠️ Unknown mode '{mode}' — returning zero vector.")
        return vec

    def read_mode(self, path):
        with open(path, "r", encoding="utf-8") as f:
            line = f.readline().strip()
        parts = line.split()
        return parts[-1].lower()  # extract mode only

    def __getitem__(self, idx):
        audio_path, label_path = self.samples[idx]
        y, sr = librosa.load(audio_path, sr=self.sr, mono=True)

        if self.load_cqt:
            C = librosa.cqt(y=y, sr=sr, **self.cqt_params)
            features = torch.tensor(np.abs(C), dtype=torch.float32)
        else:
            features = torch.tensor(y, dtype=torch.float32)

        mode = self.read_mode(label_path)
        label_vec = self.one_hot_mode(mode)

        return features, label_vec