import os
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms

class VideoDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        for label in os.listdir(root_dir):
            label_path = os.path.join(root_dir, label)

            for video in os.listdir(label_path):
                video_path = os.path.join(label_path, video)
                self.samples.append((video_path, int(label)))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label = self.samples[idx]

        frames = sorted(os.listdir(video_path))[:8]

        images = []
        for frame in frames:
            img = Image.open(os.path.join(video_path, frame)).convert("RGB")
            img = self.transform(img)
            images.append(img)

        frames_tensor = torch.stack(images)  # (8, 3, 224, 224)

        return frames_tensor, label