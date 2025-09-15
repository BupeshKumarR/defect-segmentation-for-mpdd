#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from tqdm import tqdm
from seg_models import build_seg_model

class SegDataset(Dataset):
    def __init__(self, root, split, img_size=512, aug=False):
        self.img_dir = Path(root) / split / 'images'
        self.mask_dir = Path(root) / split / 'masks'
        self.paths = sorted(list(self.img_dir.glob('*.png')))
        self.img_size = img_size
        if aug == 'strong':
            self.tf = A.Compose([
                A.RandomResizedCrop(size=(img_size, img_size), scale=(0.7, 1.0), ratio=(0.9, 1.1), p=0.6),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.Affine(rotate=(-15, 15), translate_percent=(0.0, 0.05), scale=(0.9, 1.1), shear=(-5, 5), p=0.5),
                A.RandomBrightnessContrast(p=0.4),
                A.HueSaturationValue(p=0.2),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        elif aug:
            self.tf = A.Compose([
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.2),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=10, p=0.5, border_mode=cv2.BORDER_REFLECT),
                A.RandomBrightnessContrast(p=0.3),
                A.GaussNoise(p=0.2),
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])
        else:
            self.tf = A.Compose([
                A.Normalize(mean=(0.485,0.456,0.406), std=(0.229,0.224,0.225)),
                ToTensorV2()
            ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        ip = self.paths[idx]
        mp = self.mask_dir / ip.name
        img = cv2.cvtColor(cv2.imread(str(ip)), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
        mask = (mask > 0).astype(np.float32)  # [0,1]
        aug = self.tf(image=img, mask=mask)
        img_t = aug['image']
        mask_t = aug['mask'].unsqueeze(0)  # (1,H,W)
        return img_t, mask_t

def dice_loss(pred, target, eps=1e-6):
    pred = torch.sigmoid(pred)
    num = 2 * (pred * target).sum(dim=(2,3))
    den = (pred + target).sum(dim=(2,3)) + eps
    dice = 1 - (num + eps) / den
    return dice.mean()

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = probs*targets + (1-probs)*(1-targets)
        focal = self.alpha * (1-pt).pow(self.gamma) * bce
        if self.reduction == 'mean':
            return focal.mean()
        elif self.reduction == 'sum':
            return focal.sum()
        return focal

def iou_score(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    inter = (preds * targets).sum(dim=(2,3))
    union = (preds + targets - preds*targets).sum(dim=(2,3)) + eps
    return ((inter + eps)/union).mean().item()

def f1_score_bin(logits, targets, threshold=0.5, eps=1e-6):
    preds = (torch.sigmoid(logits) > threshold).float()
    tp = (preds * targets).sum(dim=(2,3))
    fp = (preds * (1-targets)).sum(dim=(2,3))
    fn = ((1-preds) * targets).sum(dim=(2,3))
    f1 = (2*tp + eps) / (2*tp + fp + fn + eps)
    return f1.mean().item()

def main():
    ap = argparse.ArgumentParser(description="Train segmentation (U-Net++ / DeepLabV3+)")
    ap.add_argument('--data_root', required=True, help='Prepared dataset root for a part')
    ap.add_argument('--arch', default='unet++', choices=['unet++','deeplabv3+'])
    ap.add_argument('--encoder', default='timm-efficientnet-b0')
    ap.add_argument('--epochs', type=int, default=40)
    ap.add_argument('--batch_size', type=int, default=4)
    ap.add_argument('--lr', type=float, default=3e-4)
    ap.add_argument('--out_dir', default='results_seg')
    ap.add_argument('--focal_alpha', type=float, default=0.25)
    ap.add_argument('--focal_gamma', type=float, default=2.0)
    ap.add_argument('--strong_aug', action='store_true')
    ap.add_argument('--plateau_patience', type=int, default=5)
    ap.add_argument('--plateau_factor', type=float, default=0.5)
    ap.add_argument('--min_lr', type=float, default=1e-6)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    train_ds = SegDataset(args.data_root, 'train', aug=('strong' if args.strong_aug else True))
    val_ds   = SegDataset(args.data_root, 'val',   aug=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = build_seg_model(args.arch, args.encoder, encoder_weights='imagenet', num_classes=1).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode='max', factor=args.plateau_factor, patience=args.plateau_patience, min_lr=args.min_lr
    )
    fl = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    best_iou = 0.0
    history = {'train_loss':[], 'val_iou':[], 'val_f1':[]}

    for epoch in range(1, args.epochs+1):
        model.train()
        running = 0.0
        for imgs, masks in tqdm(train_loader, desc=f'Epoch {epoch}/{args.epochs}'):
            imgs, masks = imgs.to(device), masks.to(device)
            logits = model(imgs)
            loss = fl(logits, masks) + dice_loss(logits, masks)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()
        train_loss = running / max(1, len(train_loader))

        model.eval()
        iou_vals, f1_vals = [], []
        with torch.no_grad():
            for imgs, masks in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                logits = model(imgs)
                iou_vals.append(iou_score(logits, masks))
                f1_vals.append(f1_score_bin(logits, masks))
        val_iou = float(np.mean(iou_vals)) if len(iou_vals) else 0.0
        val_f1  = float(np.mean(f1_vals)) if len(f1_vals) else 0.0

        history['train_loss'].append(train_loss)
        history['val_iou'].append(val_iou)
        history['val_f1'].append(val_f1)

        print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_iou={val_iou:.4f}  val_f1={val_f1:.4f}")

        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), os.path.join(args.out_dir, f'{Path(args.data_root).name}_{args.arch}_best.pth'))

        # step scheduler on validation IoU (maximize)
        scheduler.step(val_iou)

    with open(os.path.join(args.out_dir, f'{Path(args.data_root).name}_{args.arch}_history.json'),'w') as f:
        json.dump(history, f, indent=2)

    print(f"\nDone. Best val IoU: {best_iou:.4f}. Model saved in {args.out_dir}")

if __name__ == '__main__':
    main()


