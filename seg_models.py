#!/usr/bin/env python3
import segmentation_models_pytorch as smp

def build_seg_model(arch='unet++', encoder='timm-efficientnet-b0', encoder_weights='imagenet', num_classes=1):
    arch_low = arch.lower()
    if arch_low in ['unet++','unetpp','unet_plus_plus','unetpp+']:
        model = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    elif arch_low in ['deeplabv3+','deeplabv3plus']:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=num_classes,
            activation=None
        )
    else:
        raise ValueError(f"Unknown architecture: {arch}")
    return model


