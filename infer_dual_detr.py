#!/usr/bin/env python3
import os
import argparse
from typing import List, Optional, Tuple

import torch
from PIL import Image
import torchvision.transforms.functional as TF

# Your stuff
import detr_r50
from dual_detr_r50 import DualDetrCrossEnc
from vis import show_pair
from utils import BoxType

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def load_class_names(path: Optional[str]) -> Optional[List[str]]:
    if path is None or not os.path.exists(path):
        return None
    if path.endswith(".txt"):
        with open(path) as f:
            return [ln.strip() for ln in f if ln.strip()]
    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path) as f:
            for ln in f:
                if ln.strip().startswith("names:"):
                    raw = ln.split("names:", 1)[1].strip()
                    raw = raw.strip("[]").replace("'", "").replace('"', "")
                    return [s.strip() for s in raw.split(",") if s.strip()]
    return None

def build_model_from_ckpt(ckpt_path: str, device: torch.device) -> Tuple[DualDetrCrossEnc, int]:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = ckpt.get("config", {})
    num_classes = int(cfg.get("num_classes", 3))

    base_detr, _ = detr_r50.pretrained_detr_r50()
    base_detr = detr_r50.adapt_num_classes(base_detr, num_classes=num_classes)

    model = DualDetrCrossEnc(base_detr, num_classes=num_classes, nhead=base_detr.transformer.nhead)
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    if missing:   print("[load] missing keys (truncated):", list(missing)[:5], "...")
    if unexpected:print("[load] unexpected keys (truncated):", list(unexpected)[:5], "...")
    model.to(device).eval()
    return model, num_classes

def prep_inputs_for_model(img0: Image.Image, img1: Image.Image, device: torch.device,
                          nir_imagenet_like: bool):
    """Return two versions per image:
       - *_vis : un-normalized tensors (0..1) for plotting
       - *_in  : normalized tensors for the model
    """
    img0_vis = TF.to_tensor(img0)  # (3,H,W), 0..1
    img1_vis = TF.to_tensor(img1)

    img0_in = TF.normalize(img0_vis.clone(), IMAGENET_MEAN, IMAGENET_STD)
    if nir_imagenet_like:
        img1_in = TF.normalize(img1_vis.clone(), IMAGENET_MEAN, IMAGENET_STD)
    else:
        img1_in = TF.normalize(img1_vis.clone(), [0.5, 0.5, 0.5], [0.5, 0.5, 0.5])

    return img0_vis.to(device), img1_vis.to(device), img0_in.to(device), img1_in.to(device)

def derive_cam1_path(cam0_path: str, cam0_name: str, cam1_name: str) -> Optional[str]:
    if cam0_name in cam0_path:
        return cam0_path.replace(cam0_name, cam1_name)
    return None

def select_predictions(outputs: dict, score_thresh: float, topk_fallback: int = 5):
    """
    Use raw DETR outputs:
      pred_logits: (1,Q,C+1) — softmax over classes incl. no-object
      pred_boxes : (1,Q,4)   — cxcywh in [0,1]
    Returns boxes (N,4) in cxcywh (normalized), labels (N,), scores (N,)
    """
    logits = outputs["pred_logits"].softmax(-1)[0]   # (Q, C+1) -> probs
    scores_fg, labels = logits[:, :-1].max(-1)       # drop the last 'no-object' class
    boxes_cxcywh = outputs["pred_boxes"][0]          # (Q,4), normalized

    keep = scores_fg >= score_thresh
    if keep.sum().item() == 0:
        # fallback: pick top-K highest-confidence foreground predictions
        topk = min(topk_fallback, scores_fg.numel())
        scores_vals, idxs = torch.topk(scores_fg, k=topk)
        print(f"[warn] No predictions >= {score_thresh:.2f}. Showing top-{topk} with scores:", scores_vals.tolist())
        return boxes_cxcywh[idxs].cpu(), labels[idxs].cpu(), scores_vals.cpu()

    return boxes_cxcywh[keep].cpu(), labels[keep].cpu(), scores_fg[keep].cpu()

def main():
    ap = argparse.ArgumentParser("Dual DETR inference (show_pair)")
    ap.add_argument("--ckpt", required=True, help="Path to checkpoint_best.pth")
    ap.add_argument("--img0", required=True, help="Path to cam0 (RGB) image")
    ap.add_argument("--img1", default=None, help="Path to cam1 (NIR) image")
    ap.add_argument("--cam0_name", default="firefly_left")
    ap.add_argument("--cam1_name", default="ximea")
    ap.add_argument("--classes", default=None, help="Optional classes.txt or data.yaml")
    ap.add_argument("--score_thresh", type=float, default=0.5)
    ap.add_argument("--nir_imagenet_like", action="store_true",
                    help="Normalize NIR with ImageNet stats (else use 0.5/0.5/0.5).")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Build model + load weights
    model, num_classes = build_model_from_ckpt(args.ckpt, device)
    class_names = load_class_names(args.classes)

    # Load images
    img0 = Image.open(args.img0).convert("RGB")
    if args.img1 is None:
        guess = derive_cam1_path(args.img0, args.cam0_name, args.cam1_name)
        if guess and os.path.exists(guess):
            args.img1 = guess
        else:
            raise FileNotFoundError(
                "cam1 image not provided and could not be derived.\n"
                f"Tried: {guess}\nProvide --img1 explicitly."
            )
    img1 = Image.open(args.img1)

    # Prepare tensors (both vis & model inputs)
    img0_vis, img1_vis, img0_in, img1_in = prep_inputs_for_model(img0, img1, device, args.nir_imagenet_like)

    # Wrap as NestedTensors
    from detr.util.misc import nested_tensor_from_tensor_list
    samples_rgb = nested_tensor_from_tensor_list([img0_in])
    samples_nir = nested_tensor_from_tensor_list([img1_in])

    with torch.no_grad():
        outputs = model(samples_rgb, samples_nir)  # {"pred_logits": (1,Q,C+1), "pred_boxes": (1,Q,4) normalized}

    boxes_cxcywh, labels, scores = select_predictions(outputs, args.score_thresh)

    print(f"[info] showing {len(boxes_cxcywh)} predictions "
          f"(score >= {args.score_thresh:.2f})")

    # Use your visual function: expects normalized boxes and a BoxType to convert
    # We draw only on cam0 by design inside show_pair.
    from matplotlib import pyplot as plt
    show_pair(
        img0=img0_vis.cpu(),
        img1=img1_vis.cpu(),
        boxes=boxes_cxcywh,                 # still NORMALIZED
        classes=labels,
        class_names=class_names,
        box_type=BoxType.CXCYWH             # tell it our boxes are cxcywh in [0,1]
    )
    # If you want to save instead of (or in addition to) showing:
    # plt.savefig("pred_vis.jpg", bbox_inches="tight", dpi=150)
    # print("[ok] saved -> pred_vis.jpg")

    # Also print to console
    for i in range(len(boxes_cxcywh)):
        cx, cy, w, h = boxes_cxcywh[i].tolist()
        print(f"[det] cls={int(labels[i])} score={float(scores[i]):.3f} "
              f"cxcywh=({cx:.3f},{cy:.3f},{w:.3f},{h:.3f})")

if __name__ == "__main__":
    main()
