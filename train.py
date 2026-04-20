import torch
from torch.utils.data import DataLoader
from model import LiteSeg
from loss import CombinedLoss
from dataset import VOCDataset
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

# -------- DATA --------
train_dataset = VOCDataset("data/VOCdevkit/VOC2012", split="train")
val_dataset = VOCDataset("data/VOCdevkit/VOC2012", split="val")

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
val_loader = DataLoader(val_dataset, batch_size=16)

# -------- MODEL --------
model = LiteSeg(num_classes=21).to(device)

# -------- OPTIMIZER --------
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=60)

criterion = CombinedLoss()

# -------- TRAIN --------
epochs = 60
best_val_loss = float('inf')

train_losses = []
val_losses = []

for epoch in range(epochs):

    # -------- TRAIN --------
    model.train()
    total_train_loss = 0

    for images, masks in train_loader:
        images, masks = images.to(device), masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)
    train_losses.append(avg_train_loss)

    # -------- VALIDATION --------
    model.eval()
    total_val_loss = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images, masks = images.to(device), masks.to(device)

            outputs = model(images)
            loss = criterion(outputs, masks)

            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)
    val_losses.append(avg_val_loss)

    scheduler.step()

    print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # -------- SAVE BEST MODEL --------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), "best_model.pth")

# -------- SAVE LOGS --------
with open("train_log.txt", "w") as f:
    for t, v in zip(train_losses, val_losses):
        f.write(f"{t},{v}\n")

print("Training Completed!")