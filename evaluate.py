import os
import torch
import numpy as np
from PIL import Image
from model import LiteSeg
import torchvision.transforms as T

device = "cuda" if torch.cuda.is_available() else "cpu"

model = LiteSeg(num_classes=21).to(device)
model.load_state_dict(torch.load("best_model.pth"))
model.eval()

transform = T.Compose([
    T.Resize((300,300)),
    T.ToTensor()
])

def dice_score(pred, target):
    pred = pred.flatten()
    target = target.flatten()

    intersection = (pred * target).sum()
    return (2. * intersection) / (pred.sum() + target.sum() + 1e-6)

def evaluate(root):
    image_dir = os.path.join(root, "JPEGImages")
    mask_dir = os.path.join(root, "SegmentationClass")
    split_file = os.path.join(root, "ImageSets/Segmentation/val.txt")

    with open(split_file) as f:
        file_list = f.read().splitlines()

    scores = []

    for name in file_list:
        img_path = os.path.join(image_dir, name + ".jpg")
        mask_path = os.path.join(mask_dir, name + ".png")

        image = Image.open(img_path).convert("RGB")
        gt = Image.open(mask_path)

        image = transform(image).unsqueeze(0).to(device)
        gt = T.Resize((300,300))(gt)
        gt = np.array(gt)

        gt[gt == 255] = 0
        gt = (gt > 0).astype(np.float32)

        with torch.no_grad():
            output = model(image)

        pred = torch.argmax(output, dim=1).squeeze().cpu().numpy()
        pred = (pred > 0).astype(np.float32)

        score = dice_score(pred, gt)
        scores.append(score)

    print("Average Dice Score:", np.mean(scores))


if __name__ == "__main__":
    evaluate("data/VOCdevkit/VOC2012")