import os
from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T
import random

class VOCDataset(Dataset):
    def __init__(self, root, split='train'):
        self.root = root
        self.image_dir = os.path.join(root, "JPEGImages")
        self.mask_dir = os.path.join(root, "SegmentationClass")

        split_file = os.path.join(root, "ImageSets/Segmentation", split + ".txt")

        with open(split_file) as f:
            self.files = f.read().splitlines()

        # -------- IMAGE TRANSFORM --------
        self.img_transform = T.Compose([
            T.Resize((300,300)),
            T.RandomHorizontalFlip(),
            T.RandomRotation(20),
            T.ColorJitter(brightness=0.3, contrast=0.3),
            T.ToTensor(),
        ])

        # -------- MASK TRANSFORM (IMPORTANT) --------
        self.mask_transform = T.Compose([
            T.Resize((300,300), interpolation=Image.NEAREST)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        name = self.files[idx]

        img_path = os.path.join(self.image_dir, name + ".jpg")
        mask_path = os.path.join(self.mask_dir, name + ".png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        # Apply transforms
        image = self.img_transform(image)
        mask = self.mask_transform(mask)

        mask = torch.from_numpy(
            torch.ByteTensor(torch.ByteStorage.from_buffer(mask.tobytes()))
            .view(300,300)
            .numpy()
        ).long()

        # FIX invalid label
        mask[mask == 255] = 0

        return image, mask