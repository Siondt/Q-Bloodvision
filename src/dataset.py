import torch
import numpy as np
from torchvision import transforms as T
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split
from src.config import cfg

def get_dataloaders():
    train_transform = T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.3),
        T.RandomRotation(degrees=15),
        T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.05),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_test_transform = T.Compose([
        T.Resize((cfg.IMAGE_SIZE, cfg.IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    full_dataset = ImageFolder(root=cfg.DATA_ROOT)
    class_names = full_dataset.classes
    num_classes = len(class_names)
    all_targets = np.array(full_dataset.targets)

    all_idx = np.arange(len(full_dataset))
    train_val_idx, test_idx = train_test_split(
        all_idx, test_size=cfg.TEST_RATIO, stratify=all_targets, random_state=cfg.SEED
    )
    val_size = cfg.VAL_RATIO / (cfg.TRAIN_RATIO + cfg.VAL_RATIO)
    train_idx, val_idx = train_test_split(
        train_val_idx, test_size=val_size, stratify=all_targets[train_val_idx], random_state=cfg.SEED
    )

    train_dataset = Subset(ImageFolder(cfg.DATA_ROOT, transform=train_transform), train_idx)
    val_dataset = Subset(ImageFolder(cfg.DATA_ROOT, transform=val_test_transform), val_idx)
    test_dataset = Subset(ImageFolder(cfg.DATA_ROOT, transform=val_test_transform), test_idx)

    train_labels = all_targets[train_idx]
    class_counts = np.bincount(train_labels, minlength=num_classes).astype(float)
    class_weights = torch.tensor(class_counts.sum() / (num_classes * class_counts), dtype=torch.float32).to(cfg.DEVICE)

    train_loader = DataLoader(train_dataset, batch_size=cfg.BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader, class_names, class_weights