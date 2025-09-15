#!/usr/bin/env python3
import argparse, os, json
from pathlib import Path
import numpy as np
import torch
import cv2
from tqdm import tqdm
from seg_models import build_seg_model
from train_segmentation import SegDataset

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def compute_iou_f1(pred_mask, gt_mask, eps=1e-6):
    # pred_mask, gt_mask are binary {0,1} np arrays
    inter = np.logical_and(pred_mask, gt_mask).sum()
    union = np.logical_or(pred_mask, gt_mask).sum() + eps
    iou = (inter + eps) / union
    tp = inter
    fp = np.logical_and(pred_mask, np.logical_not(gt_mask)).sum()
    fn = np.logical_and(np.logical_not(pred_mask), gt_mask).sum()
    f1 = (2*tp + eps) / (2*tp + fp + fn + eps)
    return float(iou), float(f1)

def apply_morphology(mask, open_ks=3, close_ks=3):
    mask_u8 = (mask.astype(np.uint8) * 255)
    if open_ks and open_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_ks, open_ks))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_OPEN, k)
    if close_ks and close_ks > 1:
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_ks, close_ks))
        mask_u8 = cv2.morphologyEx(mask_u8, cv2.MORPH_CLOSE, k)
    return (mask_u8 > 127).astype(np.uint8)

@torch.no_grad()
def predict_masks(model, loader, device):
    model.eval()
    preds = []
    gts = []
    for imgs, masks in tqdm(loader, desc='Predict'):
        imgs = imgs.to(device)
        logits = model(imgs)
        prob = torch.sigmoid(logits).cpu().numpy()  # (B,1,H,W)
        preds.append(prob)
        gts.append(masks.numpy())
    preds = np.concatenate(preds, axis=0)
    gts = np.concatenate(gts, axis=0)
    return preds, gts  # shapes: (N,1,H,W)

def find_best_threshold(val_preds, val_gts):
    # sweep thresholds for best F1 on defect class
    thrs = np.linspace(0.2, 0.8, 13)
    best_thr = 0.5
    best_f1 = -1
    for t in thrs:
        f1s = []
        for i in range(val_preds.shape[0]):
            pm = (val_preds[i,0] >= t).astype(np.uint8)
            gm = (val_gts[i,0] > 0.5).astype(np.uint8)
            _, f1 = compute_iou_f1(pm, gm)
            f1s.append(f1)
        mf1 = float(np.mean(f1s)) if f1s else 0.0
        if mf1 > best_f1:
            best_f1 = mf1
            best_thr = float(t)
    return best_thr, best_f1

def save_visuals(test_root, preds_bin, preds_prob, out_dir, count=8):
    img_dir = Path(test_root)/'images'
    mask_dir = Path(test_root)/'masks'
    img_paths = sorted(list(img_dir.glob('*.png')))
    out_vis = Path(out_dir)/'qualitative'
    out_vis.mkdir(parents=True, exist_ok=True)
    n = min(count, len(img_paths))
    for i in range(n):
        img = cv2.cvtColor(cv2.imread(str(img_paths[i])), cv2.COLOR_BGR2RGB)
        gt = cv2.imread(str(mask_dir/img_paths[i].name), cv2.IMREAD_GRAYSCALE)
        gt = (gt>127).astype(np.uint8)
        pred = preds_bin[i,0].astype(np.uint8)
        prob = (preds_prob[i,0]*255).astype(np.uint8)

        # overlay
        overlay = img.copy()
        r = overlay[:,:,0]
        r[pred==1] = np.clip(r[pred==1]*0.5 + 255*0.5, 0, 255)
        overlay = overlay.astype(np.uint8)

        canvas = np.concatenate([
            img,
            np.repeat(gt[...,None]*255, 3, axis=2),
            np.repeat(pred[...,None]*255, 3, axis=2),
            np.repeat(prob[...,None], 3, axis=2),
            overlay
        ], axis=1)
        cv2.imwrite(str(out_vis/f'vis_{i:02d}.png'), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))

def main():
    ap = argparse.ArgumentParser(description='Evaluate segmentation model with thresholding and morphology')
    ap.add_argument('--data_root', required=True, help='Prepared dataset root for part, e.g., data_supervised/metal_plate')
    ap.add_argument('--arch', required=True, choices=['unet++','deeplabv3+'])
    ap.add_argument('--encoder', default='timm-efficientnet-b0')
    ap.add_argument('--ckpt', required=True, help='Path to best model checkpoint (.pth)')
    ap.add_argument('--out_dir', default='results_seg')
    ap.add_argument('--open_ks', type=int, default=3)
    ap.add_argument('--close_ks', type=int, default=3)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.out_dir, exist_ok=True)

    # datasets
    val_ds = SegDataset(args.data_root, 'val', aug=False)
    test_ds = SegDataset(args.data_root, 'test', aug=False)
    from torch.utils.data import DataLoader
    val_loader = DataLoader(val_ds, batch_size=4, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

    # model
    model = build_seg_model(args.arch, args.encoder, encoder_weights=None, num_classes=1)
    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state)
    model = model.to(device)

    # predict on val to choose threshold
    val_pred_prob, val_gt = predict_masks(model, val_loader, device)
    best_thr, best_val_f1 = find_best_threshold(val_pred_prob, val_gt)

    # evaluate on test with morphology
    test_pred_prob, test_gt = predict_masks(model, test_loader, device)
    iou_list, f1_list = [], []
    preds_bin_all = []
    for i in range(test_pred_prob.shape[0]):
        pm = (test_pred_prob[i,0] >= best_thr).astype(np.uint8)
        pm = apply_morphology(pm, args.open_ks, args.close_ks)
        gm = (test_gt[i,0] > 0.5).astype(np.uint8)
        iou, f1 = compute_iou_f1(pm, gm)
        iou_list.append(iou)
        f1_list.append(f1)
        preds_bin_all.append(pm[None,...])
    preds_bin_all = np.stack(preds_bin_all, axis=0)

    mean_iou = float(np.mean(iou_list)) if iou_list else 0.0
    mean_f1 = float(np.mean(f1_list)) if f1_list else 0.0

    # save visuals and metrics
    save_visuals(Path(args.data_root)/'test', preds_bin_all, test_pred_prob, args.out_dir, count=8)
    metrics = {
        'arch': args.arch,
        'encoder': args.encoder,
        'checkpoint': args.ckpt,
        'best_val_threshold': best_thr,
        'best_val_f1_at_thr': best_val_f1,
        'test_mean_iou': mean_iou,
        'test_mean_f1': mean_f1,
        'open_ks': args.open_ks,
        'close_ks': args.close_ks,
        'num_test_samples': int(test_pred_prob.shape[0])
    }
    out_json = Path(args.out_dir)/f"{Path(args.data_root).name}_{args.arch}_test_metrics.json"
    with open(out_json, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()


