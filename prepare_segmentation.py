#!/usr/bin/env python3
import argparse, json, random
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

random.seed(42)
np.random.seed(42)

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def load_mask(mask_path: Path, target_size=(512, 512)):
    m = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if m is None:
        return np.zeros(target_size, np.uint8)
    m = cv2.resize(m, target_size, interpolation=cv2.INTER_NEAREST)
    m = (m > 0).astype(np.uint8) * 255
    return m

def load_image(img_path: Path, target_size=(512, 512)):
    im = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    if im is None:
        raise RuntimeError(f"Failed to read image: {img_path}")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    im = cv2.resize(im, target_size, interpolation=cv2.INTER_AREA)
    return im

def find_mask_for_image(gt_root: Path, defect_dir_name: str, img_name: str):
    stem = Path(img_name).stem
    candidate = gt_root / defect_dir_name / f"{stem}_mask.png"
    return candidate if candidate.exists() else None

def split_bad_images(bad_imgs, train_ratio=0.7):
    n = len(bad_imgs)
    k = int(round(train_ratio * n))
    random.shuffle(bad_imgs)
    return bad_imgs[:k], bad_imgs[k:]

def visualize_grid(images, masks, out_path: Path, title: str):
    n = min(8, len(images))
    cols = 4
    rows = int(np.ceil(n / cols))
    plt.figure(figsize=(4*cols, 3*rows))
    for i in range(n):
        plt.subplot(rows, cols, i+1)
        if masks is not None:
            overlay = images[i].copy()
            mask = masks[i]
            if mask.ndim == 2:
                mask3 = np.stack([mask, np.zeros_like(mask), np.zeros_like(mask)], axis=-1)
            else:
                mask3 = mask
            alpha = 0.35
            overlay = (overlay * (1-alpha) + (mask3>0)*255*alpha).astype(np.uint8)
            plt.imshow(overlay)
        else:
            plt.imshow(images[i])
        plt.axis('off')
    plt.suptitle(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

def save_pair(img, mask, out_img_path: Path, out_mask_path: Path):
    ensure_dir(out_img_path.parent)
    ensure_dir(out_mask_path.parent)
    cv2.imwrite(str(out_img_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(str(out_mask_path), mask)

def main():
    ap = argparse.ArgumentParser(description="Prepare supervised segmentation dataset (re-split + EDA)")
    ap.add_argument('--data_dir', default='anomaly_dataset', help='Root MPDD dataset in this repo')
    ap.add_argument('--part', required=True, choices=['metal_plate','bracket_black','bracket_brown','bracket_white'], help='Part to prepare')
    ap.add_argument('--out_dir', default='data_supervised', help='Output root for prepared dataset')
    ap.add_argument('--img_size', type=int, default=512, help='Resize square size')
    ap.add_argument('--train_ratio', type=float, default=0.7, help='Portion of bad test imgs moved to train')
    ap.add_argument('--val_ratio', type=float, default=0.15, help='Portion of train routed to validation')
    args = ap.parse_args()

    src_root = Path(args.data_dir) / args.part
    assert src_root.exists(), f"Dataset not found at {src_root}"

    out_root = Path(args.out_dir) / args.part
    img_size = (args.img_size, args.img_size)

    for split in ['train','val','test']:
        for sub in ['images','masks']:
            ensure_dir(out_root / split / sub)

    train_good = sorted((src_root / 'train' / 'good').glob('*.png'))
    test_good = sorted((src_root / 'test' / 'good').glob('*.png'))
    gt_root = src_root / 'ground_truth'
    bad_dirs = [d for d in (src_root / 'test').glob('*') if d.is_dir() and d.name != 'good']

    bad_all = []
    for d in bad_dirs:
        for p in sorted(d.glob('*.png')):
            bad_all.append((d.name, p))

    bad_train, bad_test = split_bad_images(bad_all, args.train_ratio)

    images_train, masks_train = [], []
    images_val, masks_val = [], []
    images_test, masks_test = [], []
    defect_pixel_ratios = []

    def append_item(img_p: Path, mask_p: Path, target='train'):
        img = load_image(img_p, img_size)
        mask = load_mask(mask_p, img_size) if mask_p else np.zeros(img_size, np.uint8)
        ratio = (mask > 0).mean()
        defect_pixel_ratios.append(float(ratio))
        if target == 'train':
            images_train.append(img); masks_train.append(mask)
        elif target == 'val':
            images_val.append(img); masks_val.append(mask)
        else:
            images_test.append(img); masks_test.append(mask)

    for p in tqdm(train_good, desc='Train good'):
        append_item(p, None, 'train')

    for defect_name, p in tqdm(bad_train, desc='Train bad'):
        m = find_mask_for_image(gt_root, defect_name, p.name)
        append_item(p, m, 'train')

    for p in tqdm(test_good, desc='Test good'):
        append_item(p, None, 'test')

    for defect_name, p in tqdm(bad_test, desc='Test bad'):
        m = find_mask_for_image(gt_root, defect_name, p.name)
        append_item(p, m, 'test')

    # create val split (configurable)
    n_train = len(images_train)
    idx = list(range(n_train))
    random.shuffle(idx)
    n_val = max(1, int(args.val_ratio * n_train))
    val_idx = set(idx[:n_val])

    _images_train, _masks_train = [], []
    for i in range(n_train):
        if i in val_idx:
            images_val.append(images_train[i]); masks_val.append(masks_train[i])
        else:
            _images_train.append(images_train[i]); _masks_train.append(masks_train[i])
    images_train, masks_train = _images_train, _masks_train

    def dump_split(images, masks, split):
        for i, (im, m) in enumerate(zip(images, masks)):
            out_img = out_root / split / 'images' / f'{split}_{i:05d}.png'
            out_msk = out_root / split / 'masks' / f'{split}_{i:05d}.png'
            ensure_dir(out_img.parent); ensure_dir(out_msk.parent)
            cv2.imwrite(str(out_img), cv2.cvtColor(im, cv2.COLOR_RGB2BGR))
            cv2.imwrite(str(out_msk), m)

    dump_split(images_train, masks_train, 'train')
    dump_split(images_val, masks_val, 'val')
    dump_split(images_test, masks_test, 'test')

    eda_dir = out_root / 'eda'
    ensure_dir(eda_dir)
    visualize_grid(images_train, masks_train, eda_dir / 'train_samples.png', 'Train Samples (overlay)')
    visualize_grid(images_val, masks_val, eda_dir / 'val_samples.png', 'Val Samples (overlay)')
    visualize_grid(images_test, masks_test, eda_dir / 'test_samples.png', 'Test Samples (overlay)')

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    sns.histplot(defect_pixel_ratios, bins=30, kde=True, color='tomato')
    plt.title('Defect Pixel Ratio Distribution')
    plt.xlabel('Ratio'); plt.ylabel('Count'); plt.tight_layout()
    plt.savefig(eda_dir / 'defect_ratio_hist.png', dpi=150)
    plt.close()

    summary = {
        'part': args.part,
        'img_size': args.img_size,
        'train': len(images_train),
        'val': len(images_val),
        'test': len(images_test),
        'train_ratio_bad': args.train_ratio
    }
    with open(eda_dir / 'summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    print(f"\nPrepared dataset at: {out_root}")
    print(f"Train: {len(images_train)}, Val: {len(images_val)}, Test: {len(images_test)}")
    print(f"EDA saved in: {eda_dir}")

if __name__ == '__main__':
    main()


