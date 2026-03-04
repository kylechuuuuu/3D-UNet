import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import MRADataset
from model import UNet3D
import os
from tqdm import tqdm

# Hyperparameters
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 2 # 3D data is heavy
NUM_EPOCHS = 50
NUM_WORKERS = 2 # Set to 0 to avoid multiprocessing issues in some envs
PIN_MEMORY = True
LOAD_MODEL = False
TRAIN_IMG_DIR = "/hy-tmp/TOF-MRA/train/"
VAL_IMG_DIR = "/hy-tmp/TOF-MRA/test/" # Use test as val for now

def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)

    for batch_idx, (data, targets) in enumerate(loop):
        data = data.to(DEVICE)
        targets = targets.to(DEVICE)

        # forward
        with torch.cuda.amp.autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets) # ComboLoss expects raw logits for BCE part

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # update tqdm loop
        loop.set_postfix(loss=loss.item())

class DiceLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceLoss, self).__init__()

    def forward(self, inputs, targets, smooth=1):
        # inputs are logits, so apply sigmoid
        inputs = torch.sigmoid(inputs)       
        
        # flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        intersection = (inputs * targets).sum()                            
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        
        return 1 - dice

class ComboLoss(nn.Module):
    def __init__(self):
        super(ComboLoss, self).__init__()
        self.dice = DiceLoss()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets):
        # DiceLoss takes logits (handles sigmoid internally)
        # BCEWithLogitsLoss takes logits
        return self.dice(inputs, targets) + self.bce(inputs, targets)

def main():
    model = UNet3D(in_channels=1, out_channels=1).to(DEVICE)
    loss_fn = ComboLoss() # Use Combo of BCE and Dice
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    scaler = torch.cuda.amp.GradScaler()
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=5) # Scheduler

    train_ds = MRADataset(
        root_dir=TRAIN_IMG_DIR,
        mode='train',
        patch_size=(128, 128, 64)
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=True,
    )

    # Simple validation loader (using train dataset in 'test' mode or actual test set)
    val_ds = MRADataset(
        root_dir=VAL_IMG_DIR,
        mode='test',
        patch_size=(128, 128, 64)
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        num_workers=NUM_WORKERS,
        pin_memory=PIN_MEMORY,
        shuffle=False,
    )

    best_dice = 0.0

    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch+1}/{NUM_EPOCHS}")
        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # Check accuracy/dice on validation
        current_dice = check_accuracy(val_loader, model, device=DEVICE)
        scheduler.step(current_dice)
        
        # Save best model
        if current_dice > best_dice:
            best_dice = current_dice
            print(f"New best Dice: {best_dice:.4f}. Saving model...")
            checkpoint = {
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            }
            torch.save(checkpoint, "my_checkpoint.pth.tar")

def check_accuracy(loader, model, device="cuda"):
    dice_score = 0
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            preds = torch.sigmoid(model(x))
            preds = (preds > 0.5).float()
            
            # simple dice
            dice_score += (2 * (preds * y).sum()) / (
                (preds + y).sum() + 1e-8
            )

    mean_dice = dice_score/len(loader)
    print(f"Dice score: {mean_dice}")
    model.train()
    return mean_dice

if __name__ == "__main__":
    main()
