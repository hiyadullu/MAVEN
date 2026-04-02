import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class IEMOCAPVisualDataset(Dataset):
    def __init__(self, dataframe, transform=None, num_frames=16):
        self.df = dataframe
        self.transform = transform
        self.num_frames = num_frames

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        folder_path = row["path"]   # <-- must point to Ses01F_scriptXX folder
        label = row["label"]

        frame_files = sorted(os.listdir(folder_path))
        frame_paths = [os.path.join(folder_path, f) for f in frame_files]

        total_frames = len(frame_paths)

        # 🔥 Uniform sampling
        if total_frames >= self.num_frames:
            indices = torch.linspace(0, total_frames - 1, steps=self.num_frames).long()
        else:
            indices = torch.randint(0, total_frames, (self.num_frames,))

        selected_frames = [frame_paths[i] for i in indices]

        frames = []
        for frame_path in selected_frames:
            img = Image.open(frame_path).convert("RGB")

            if self.transform:
                img = self.transform(img)

            frames.append(img)

        frames = torch.stack(frames)  # (T, C, H, W)

        return frames, label