# Industrial Defect Detection: MPDD Dataset Analysis

## 1. Objective and Approach

**Objective**: Build an automated system to detect and localize defects in industrial metal parts using the MPDD (Metal Parts Defect Detection) dataset.

**Chosen Approach**: A supervised semantic segmentation model. This approach was selected because the dataset provided pixel-perfect defect annotations, which allows for a high-precision solution that can not only classify but also precisely localize defects. This is a superior method to unsupervised anomaly detection for a problem with known defect types.

## 2. Methodology and Implementation

### Preprocessing

The preprocessing pipeline transforms the MPDD dataset into a format suitable for supervised segmentation training:

- **Data Re-splitting**: The original test set was re-split to create a balanced dataset containing both "good" and "bad" parts for supervised training (70% of bad test images moved to training)
- **Mask Generation**: Ground truth annotations were converted into binary segmentation masks (white for defects, black for background)
- **EDA**: Exploratory Data Analysis was performed to understand the distribution of defect pixel ratios and dataset characteristics
- **Data Augmentation**: Applied robust augmentations including rotations, flips, brightness/contrast adjustments, and morphological transformations

**Script**: `prepare_segmentation.py`

### Model Creation

Two state-of-the-art segmentation architectures were compared:

- **U-Net++**: Nested and dense skip connections for multi-scale feature capture
- **DeepLabV3+**: Atrous Spatial Pyramid Pooling (ASPP) for multi-scale contextual information
- **Backbone**: Pre-trained EfficientNet-B0 encoder for both architectures
- **Loss Functions**: Combination of Focal Loss and Dice Loss to handle severe class imbalance
- **Training Strategy**: ReduceLROnPlateau scheduler with model checkpointing for optimal performance

**Scripts**: `seg_models.py` and `train_segmentation.py`

### Post-Processing

Several techniques were applied to improve final prediction quality:

- **Thresholding**: Optimal threshold selection (0.75 for U-Net++, 0.45 for DeepLabV3+) based on validation F1-score
- **Morphological Operations**: Open/close operations (3x3 kernel) to remove noise and fill gaps
- **Visualization**: Generated qualitative results showing predictions, ground truth, and overlays

**Script**: `evaluate_segmentation.py`

## 3. Results and Discussion

### Performance Summary

| Model | Best Validation IoU | Best Validation F1-Score | Test IoU | Test F1-Score |
|-------|-------------------|-------------------------|----------|---------------|
| U-Net++ | 0.9027 | 0.9222 | 0.9551 | 0.9739 |
| DeepLabV3+ | 0.9001 | 0.9209 | 0.8675 | 0.8866 |

**Conclusion**: U-Net++ was selected for final evaluation due to its superior performance on both validation and test sets, demonstrating better capability to localize and outline defects with higher precision and recall.

### Visual Results

Qualitative results are available in `results_seg/qualitative/` showing:
- Original images with ground truth masks
- Model predictions with probability maps
- Overlay visualizations highlighting detected defects

## 4. How to Run the Code

### Prerequisites

```bash
# Clone the repository
git clone https://github.com/BupeshKumarR/defect-segmentation-for-mpdd

# Create and activate virtual environment
bash activate_env.sh

# Install dependencies
pip install -r requirements.txt
```

### Dataset Setup

1. Download the MPDD dataset from the provided link
2. Place the unzipped folder in the project root as `anomaly_dataset/`
3. Ensure the structure follows: `anomaly_dataset/metal_plate/train/`, `anomaly_dataset/metal_plate/test/`, `anomaly_dataset/metal_plate/ground_truth/`

### Step-by-Step Execution

```bash
# 1. Prepare the dataset
python prepare_segmentation.py --part metal_plate --data_dir anomaly_dataset --out_dir data_supervised

# 2. Train U-Net++ model
python train_segmentation.py --data_root data_supervised/metal_plate --arch unet++ --epochs 40

# 3. Train DeepLabV3+ model (for comparison)
python train_segmentation.py --data_root data_supervised/metal_plate --arch deeplabv3+ --epochs 40

# 4. Evaluate best model
python evaluate_segmentation.py --data_root data_supervised/metal_plate --arch unet++ --ckpt results_seg/metal_plate_unet++_best.pth
```

### Output Structure

```
results_seg/
├── metal_plate_unet++_best.pth          # Best U-Net++ model
├── metal_plate_deeplabv3+_best.pth      # Best DeepLabV3+ model
├── metal_plate_unet++_test_metrics.json # Test evaluation metrics
├── metal_plate_deeplabv3+_test_metrics.json
├── qualitative/                          # Visual results
│   ├── vis_00.png
│   ├── vis_01.png
│   └── ...
└── training_*.log                        # Training logs
```

## 5. Potential Improvements

- **Multi-class Segmentation**: Extend the model to identify and segment multiple defect types if more detailed labels were available
- **Transfer Learning**: Experiment with other pre-trained backbones (ResNeXt, ViT) for further performance gains
- **Active Learning**: Implement an active learning loop to continuously improve the model by focusing on hard-to-classify examples
- **Ensemble Methods**: Combine multiple models for improved robustness and performance

## 6. Technical Details

- **Framework**: PyTorch with segmentation-models-pytorch
- **Augmentation**: Albumentations library for robust data augmentation
- **Optimization**: AdamW optimizer with ReduceLROnPlateau scheduler
- **Loss Functions**: Focal Loss (α=0.25, γ=2.0) + Dice Loss
- **Image Size**: 512×512 pixels
- **Batch Size**: 4 (optimized for available hardware)

## 7. Project Structure

```
assessment/
├── prepare_segmentation.py      # Data preprocessing and EDA
├── train_segmentation.py        # Model training pipeline
├── evaluate_segmentation.py     # Model evaluation and post-processing
├── seg_models.py               # Model architecture definitions
├── requirements.txt            # Python dependencies
├── activate_env.sh            # Environment setup script
├── README.md                  # This file
├── data_supervised/           # Processed dataset
│   └── metal_plate/
│       ├── train/
│       ├── val/
│       ├── test/
│       └── eda/
└── results_seg/               # Training results and outputs
    ├── *.pth                  # Model checkpoints
    ├── *.json                 # Metrics and history
    ├── *.log                  # Training logs
    └── qualitative/           # Visual results
```

## 8. Contact

For questions or issues, please refer to the project repository or contact the development team.
