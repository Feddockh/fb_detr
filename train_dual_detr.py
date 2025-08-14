#!/usr/bin/env python3
# train_dual_detr.py

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
from dual_detr_r50 import DualDetrCrossEnc

# ---- Loss pieces from official DETR
from detr.models.matcher import HungarianMatcher
from detr.models.detr import SetCriterion

from dataset import YoloV5MultiCamDataset, SetType, BoxType, multicam_collate_fn
from vis import CSVLogger, show_pair, extract_components, plot_from_csv


# ======================
# ====== CONFIG ========
# ======================
OUTPUT_DIR       = "runs/dual_detr_r50_rivendale_v4"
EPOCHS           = 100
FROZEN_EPOCHS    = 10
BATCH_SIZE       = 2
NUM_WORKERS      = 4
BASE_LR          = 1e-4
BACKBONE_LR      = 1e-5
NEW_LR           = 1e-3
WEIGHT_DECAY     = 1e-4
LR_DROP_EPOCH    = 40
MAX_NORM         = 0.1      # gradient clipping; 0 disables

DATA_ROOT        = "datasets/rivendale_v4"
CAM0_NAME        = "firefly_left"
CAM1_NAME        = "ximea_demosaic"


# ============================
# ====== TRANSFORMS ==========
# ============================
# Simple no-aug baseline: convert PIL->Tensor + Imagenet norm for BOTH cameras
# TODO: Remove and implement in dataset object class
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def to_tensor_and_norm(img):
    if not torch.is_tensor(img):
        img = TF.to_tensor(img)                  # [0..1], CxHxW
    return TF.normalize(img, IMAGENET_MEAN, IMAGENET_STD)

def prepare_tensors(imgs: List, device):
    """
    imgs: list of PIL or tensors (B), returns list of normalized tensors on device.
    """
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
def param_groups_dual(model: nn.Module, lr=BASE_LR, lr_backbone=BACKBONE_LR, lr_new=NEW_LR):
    backbone_params, old_params, new_params = [], [], []
    for n, p in model.named_parameters():
        if not p.requires_grad:
            continue
        if "backbone_" in n:                           # both RGB & NIR backbones
            backbone_params.append(p)
        elif n.startswith("fuse.") or n.startswith("class_embed."):
            new_params.append(p)                       # the new cross-fuse + reinit class head
        else:
            old_params.append(p)                       # encoder/decoder/input_proj/bbox head/query_embed
    return [
        {"params": old_params,      "lr": lr},
        {"params": backbone_params, "lr": lr_backbone},
        {"params": new_params,      "lr": lr_new},
    ]


# ==================================
# ====== DATA LOADERS ==============
# ==================================
def make_loaders():
    train_ds = YoloV5MultiCamDataset(
        base_dir=DATA_ROOT,
        cam0=CAM0_NAME,
        cam1=CAM1_NAME,
        set_type=SetType.TRAIN,
        transforms=None,                     # we normalize in the loop below
        return_type=BoxType.CXCYWH           # DETR expects cx,cy,w,h normalized
    )
    val_ds = YoloV5MultiCamDataset(
        base_dir=DATA_ROOT,
        cam0=CAM0_NAME,
        cam1=CAM1_NAME,
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
        imgs0 = prepare_tensors(batch["img0"], device)
        imgs1 = prepare_tensors(batch["img1"], device)
        samples_rgb = nested_tensor_from_tensor_list(imgs0)
        samples_nir = nested_tensor_from_tensor_list(imgs1)

        outputs = model(samples_rgb, samples_nir)
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
        imgs0 = prepare_tensors(batch["img0"], device)
        imgs1 = prepare_tensors(batch["img1"], device)
        samples_rgb = nested_tensor_from_tensor_list(imgs0)
        samples_nir = nested_tensor_from_tensor_list(imgs1)

        outputs = model(samples_rgb, samples_nir)
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
    # device = "cpu"

    # CSV logger
    log_csv = os.path.join(OUTPUT_DIR, "training_log.csv")
    logger = CSVLogger(log_csv)

    # Load the datasets
    train_ds, val_ds, train_loader, val_loader = make_loaders()
    NUM_CLASSES = len(train_ds.classes)
    print(f"Training for {NUM_CLASSES} classes: {train_ds.classes}")

    # Build dual model from pretrained DETR-R50 and adapt to NUM_CLASSES
    base_detr, _ = detr_r50.pretrained_detr_r50()
    base_detr = detr_r50.adapt_num_classes(base_detr, num_classes=NUM_CLASSES)
    print("heads: ", base_detr.transformer.nhead)
    model = DualDetrCrossEnc(base_detr, num_classes=NUM_CLASSES, nhead=base_detr.transformer.nhead)
    model.to(device)

    # Freeze backbone layers
    for n, p in model.named_parameters():
        if n.startswith("backbone_rgb") or n.startswith("backbone_nir"):
            p.requires_grad = False

    # Loss / Optim / Sched
    dec_layers = model.decoder.num_layers
    criterion, weight_dict = build_criterion(NUM_CLASSES, dec_layers)
    criterion.to(device)

    param_groups = param_groups_dual(model)
    optimizer = torch.optim.AdamW(param_groups, lr=BASE_LR, weight_decay=WEIGHT_DECAY)
    lr_sched  = torch.optim.lr_scheduler.StepLR(optimizer, step_size=LR_DROP_EPOCH)

    print("Start training.")
    best_val = math.inf

    for epoch in range(EPOCHS):
        # Unfreeze backbone layers
        if epoch >= FROZEN_EPOCHS:
            for n, p in model.named_parameters():
                if n.startswith("backbone_rgb") or n.startswith("backbone_nir"):
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

        # CSV log (two rows per epoch)
        logger.log_epoch(epoch, "train", cur_lr, tr)
        logger.log_epoch(epoch, "val",   cur_lr, va)

        # Save last + best (unchanged)
        ckpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "config": {
                "num_classes": NUM_CLASSES,
                "base_lr": BASE_LR, "backbone_lr": BACKBONE_LR, "new_lr": NEW_LR,
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
