import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def get_fer_dataloaders(data_dir, batch_size=32):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
    ])

    train_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "train"),
        transform=transform
    )

    test_dataset = datasets.ImageFolder(
        os.path.join(data_dir, "test"),
        transform=transform
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader