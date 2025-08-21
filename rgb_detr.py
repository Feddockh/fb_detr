#!/usr/bin/env python3
# train_rgb_detr.py

import os
import time
import math
from typing import List, Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms.functional as TF

# ---- DETR repo utilities
from detr.util.misc import nested_tensor_from_tensor_list

# ---- Helpers / model
import detr_r50

# ---- Loss pieces from official DETR
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

from dataset import YoloV5MultiCamDataset, SetType, BoxType, multicam_collate_fn
from vis import CSVLogger, extract_components, plot_from_csv


# ======================
# ====== CONFIG ========
# ======================
OUTPUT_DIR       = "runs/rgb_detr_r50_rivendale_v5"
EPOCHS           = 200
FROZEN_EPOCHS    = 20
BATCH_SIZE       = 4 # Larger batch size possible with single model
NUM_WORKERS      = 4
BASE_LR          = 1e-4
BACKBONE_LR      = 1e-5
WEIGHT_DECAY     = 1e-4
LR_DROP_EPOCH    = 40
MAX_NORM         = 0.1      # gradient clipping

DATA_ROOT        = "datasets/rivendale_v5"
CAM_NAME         = "firefly_left" # This is the RGB camera


# ============================
# ====== TRANSFORMS ==========
# ============================
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def to_tensor_and_norm(img):
    if not torch.is_tensor(img):
        img = TF.to_tensor(img)
    return TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

def prepare_tensors(imgs: List, device):
    out = []
    for im in imgs:
        if not torch.is_tensor(im):
            im = to_tensor_and_norm(im)
        else:
            # assume already 0-1; normalize if roughly in [0,1] range
            im = TF.normalize(im, IMAGENET_MEAN, IMAGENET_STD)
        out.append(im.to(device))
    return out


# ==================================
# ====== OPTIMIZER GROUPS ==========
# ==================================
def param_groups(model: nn.Module, lr=BASE_LR, lr_backbone=BACKBONE_LR):
    """
    Separate model parameters into two groups: backbone and non-backbone,
    to allow for different learning rates.
    """
    param_dicts = [
        {"params": [p for n, p in model.named_parameters() if "backbone" not in n and p.requires_grad]},
        {
            "params": [p for n, p in model.named_parameters() if "backbone" in n and p.requires_grad],
            "lr": lr_backbone,
        },
    ]
    return param_dicts


# ==================================
# ====== DATA LOADERS ==============
# ==================================
def make_loaders():
    """
    Initializes the training and validation datasets and dataloaders.
    Uses the YoloV5MultiCamDataset but only the 'cam0' image will be used for training.
    """
    train_ds = YoloV5MultiCamDataset(
        base_dir=DATA_ROOT,
        cam0=CAM_NAME,
        cam1=CAM_NAME, # Pass same name, this image will be loaded but ignored
        set_type=SetType.TRAIN,
        transforms=None,                     # we normalize in the loop
        return_type=BoxType.CXCYWH           # DETR expects cx,cy,w,h normalized
    )
    val_ds = YoloV5MultiCamDataset(
        base_dir=DATA_ROOT,
        cam0=CAM_NAME,
        cam1=CAM_NAME, # Pass same name, this image will be loaded but ignored
        set_type=SetType.VAL,
        transforms=None,
        return_type=BoxType.CXCYWH
    )

    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          collate_fn=multicam_collate_fn)
    val_dl   = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False,
                          num_workers=NUM_WORKERS, pin_memory=True,
                          collate_fn=multicam_collate_fn)
    return train_ds, val_ds, train_dl, val_dl


# ==================================
# ====== LOSSES / MATCHER ==========
# ==================================
def build_criterion(num_classes: int, dec_layers: int):
    matcher = HungarianMatcher(cost_class=1, cost_bbox=5, cost_giou=2)
    weight_dict = {"loss_ce": 1, "loss_bbox": 5, "loss_giou": 2}

    # aux losses for intermediate decoder layers
    aux_weight = {k + f"_{i}": v for i in range(dec_layers - 1) for k, v in weight_dict.items()}
    weight_dict.update(aux_weight)

    losses = ["labels", "boxes", "cardinality"]
    criterion = SetCriterion(
        num_classes=num_classes,
        matcher=matcher,
        weight_dict=weight_dict,
        eos_coef=0.1,
        losses=losses,
    )
    return criterion, weight_dict


# ==================================
# ====== TRAIN / EVAL LOOPS ========
# ==================================
def loss_from_dict(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> torch.Tensor:
    return sum(loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict)

@torch.no_grad()
def evaluate_epoch(model, criterion, val_loader, device):
    model.eval()
    criterion.eval()

    totals = {"loss": 0.0, "ce": 0.0, "bbox": 0.0, "giou": 0.0, "card": 0.0}
    n_batches = 0

    for batch in val_loader:
        imgs = prepare_tensors(batch["img0"], device)
        samples = nested_tensor_from_tensor_list(imgs)

        outputs = model(samples)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = criterion(outputs, targets)
        comps = extract_components(loss_dict, criterion.weight_dict)

        for k in totals:
            totals[k] += comps[k]
        n_batches += 1

    if n_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / n_batches for k, v in totals.items()}

def train_epoch(model, criterion, optimizer, train_loader, device, epoch, max_norm=0.0):
    model.train()
    criterion.train()

    totals = {"loss": 0.0, "ce": 0.0, "bbox": 0.0, "giou": 0.0, "card": 0.0}
    n_batches = 0
    t0 = time.time()

    for i, batch in enumerate(train_loader):
        imgs = prepare_tensors(batch["img0"], device)
        samples = nested_tensor_from_tensor_list(imgs)

        outputs = model(samples)
        targets = [{k: v.to(device) for k, v in t.items()} for t in batch["target"]]

        loss_dict = criterion(outputs, targets)
        loss = loss_from_dict(loss_dict, criterion.weight_dict)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if max_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        optimizer.step()

        comps = extract_components(loss_dict, criterion.weight_dict)
        for k in totals:
            totals[k] += comps[k]
        n_batches += 1

        if (i + 1) % 50 == 0:
            it_s = time.time() - t0
            print(f"Epoch {epoch:03d} | Iter {i+1:05d}/{len(train_loader):05d} "
                  f"| loss {totals['loss']/n_batches:.4f} "
                  f"| ce {totals['ce']/n_batches:.3f} "
                  f"| bbox {totals['bbox']/n_batches:.3f} "
                  f"| giou {totals['giou']/n_batches:.3f} "
                  f"| card {totals['card']/n_batches:.2f} "
                  f"| {it_s/(i+1):.3f}s/it")

    if n_batches == 0:
        return {k: 0.0 for k in totals}
    return {k: v / n_batches for k, v in totals.items()}


# ==================================
# ============ MAIN ================
# ==================================
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # CSV logger
    log_csv = os.path.join(OUTPUT_DIR, "training_log.csv")
    logger = CSVLogger(log_csv)

    # Load the datasets
    train_ds, val_ds, train_loader, val_loader = make_loaders()
    NUM_CLASSES = len(train_ds.classes)
    print(f"Training for {NUM_CLASSES} classes: {train_ds.classes}")

    # Build from pretrained DETR-R50 and adapt to NUM_CLASSES
    model, _ = detr_r50.pretrained_detr_r50()
    model = detr_r50.adapt_num_classes(model, num_classes=NUM_CLASSES)
    model.to(device)

    # Freeze backbone layers for initial epochs
    for n, p in model.named_parameters():
        if "backbone" in n:
            p.requires_grad = False

    # Loss / Optim / Sched
    dec_layers = model.transformer.decoder.num_layers
    criterion, weight_dict = build_criterion(NUM_CLASSES, dec_layers)
    criterion.to(device)

    param_groups_for_opt = param_groups(model, lr=BASE_LR, lr_backbone=BACKBONE_LR)
    optimizer = torch.optim.AdamW(param_groups_for_opt, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    lr_sched  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DROP_EPOCH)

    print("Start training.")
    best_val = math.inf

    for epoch in range(EPOCHS):
        # Unfreeze backbone layers after FROZEN_EPOCHS
        if epoch >= FROZEN_EPOCHS:
            for n, p in model.named_parameters():
                if "backbone" in n:
                    p.requires_grad = True

        tr = train_epoch(model, criterion, optimizer, train_loader, device, epoch, max_norm=MAX_NORM)
        lr_sched.step()
        va = evaluate_epoch(model, criterion, val_loader, device)

        # Current LR (first param group is fine to report)
        cur_lr = optimizer.param_groups[0]["lr"]

        # Console summary
        print(f"Epoch {epoch:03d} done | "
            f"train loss: {tr['loss']:.4f} (ce {tr['ce']:.3f} bbox {tr['bbox']:.3f} giou {tr['giou']:.3f} card {tr['card']:.2f}) | "
            f"val loss: {va['loss']:.4f} (ce {va['ce']:.3f} bbox {va['bbox']:.3f} giou {va['giou']:.3f} card {va['card']:.2f})")

        # CSV log
        logger.log_epoch(epoch, "train", cur_lr, tr)
        logger.log_epoch(epoch, "val",   cur_lr, va)

        # Save last + best checkpoints
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "num_classes": NUM_CLASSES,
                "base_lr": BASE_LR, "backbone_lr": BACKBONE_LR,
                "weight_decay": WEIGHT_DECAY, "lr_drop": LR_DROP_EPOCH
            }
        }
        torch.save(ckpt, os.path.join(OUTPUT_DIR, "checkpoint_last.pth"))
        if va["loss"] < best_val:
            best_val = va["loss"]
            torch.save(ckpt, os.path.join(OUTPUT_DIR, "checkpoint_best.pth"))
            print(f"  ↳ new best (val loss {best_val:.4f}) saved.")

    # Close CSV and plot
    logger.close()
    plot_from_csv(log_csv, os.path.join(OUTPUT_DIR, "plots"))

    print("Training complete.")

if __name__ == "__main__":
    main()
