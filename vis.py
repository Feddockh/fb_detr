import torch
from typing import List, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt
from utils import BoxType, to_xyxy_norm
import csv
from pathlib import Path
import matplotlib.pyplot as plt

def show_pair(img0: torch.Tensor, img1: torch.Tensor, boxes: torch.Tensor, classes: torch.Tensor, 
              class_names: List = None, box_type: BoxType = BoxType.XYXY,
              colors: List = ['green', 'blue', 'yellow', 'red', 'orange', 'cyan', 'purple']):
    """
    Show a pair of images with their corresponding bounding boxes.
    """

    # Display the images from the two cameras
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(img0.permute(1, 2, 0).cpu())
    ax[0].set_title("Camera 0")
    ax[1].imshow(img1.permute(1, 2, 0).cpu())
    ax[1].set_title("Camera 1")

    # Format the boxes as xyxy
    boxes_xyxy = to_xyxy_norm(boxes, box_type)

    h0, w0 = img0.shape[1], img0.shape[2]
    h1, w1 = img1.shape[1], img1.shape[2]

    # Add a key for the colors and the class names
    if class_names is not None:
        unique_classes = torch.unique(classes)
        handles = [plt.Line2D([0], [0], color=colors[i], lw=4) for i in unique_classes]
        labels = [class_names[i] for i in unique_classes]
        ax[0].legend(handles=handles, labels=labels, title="Classes")

    for box, cls in zip(boxes_xyxy, classes):
        x1, y1, x2, y2 = box
        
        # Scale box for image 0 (we only want boxes on this image)
        x1_0, y1_0, x2_0, y2_0 = x1 * w0, y1 * h0, x2 * w0, y2 * h0
        ax[0].add_patch(plt.Rectangle((x1_0, y1_0), x2_0 - x1_0, y2_0 - y1_0, fill=False, color=colors[cls], linewidth=2))

    plt.show()

# ======================
# ===== CSV LOGGER =====
# ======================
class CSVLogger:
    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        write_header = not self.csv_path.exists()
        self.fh = open(self.csv_path, "a", newline="")
        self.wr = csv.writer(self.fh)
        if write_header:
            self.wr.writerow([
                "epoch", "split", "lr",
                "loss", "loss_ce", "loss_bbox", "loss_giou", "cardinality_error"
            ])
            self.fh.flush()

    def log_epoch(self, epoch: int, split: str, lr: float, metrics: dict):
        self.wr.writerow([
            epoch, split, lr,
            metrics.get("loss", 0.0),
            metrics.get("ce", 0.0),
            metrics.get("bbox", 0.0),
            metrics.get("giou", 0.0),
            metrics.get("card", 0.0),
        ])
        self.fh.flush()

    def close(self):
        try:
            self.fh.close()
        except Exception:
            pass


# ==========================
# === METRIC AGGREGATION ===
# ==========================
def _sum_prefixed(d: Dict[str, torch.Tensor], prefix: str) -> float:
    """Sum main + aux keys, e.g. loss_ce + loss_ce_0 + ..."""
    out = 0.0
    for k, v in d.items():
        if k.startswith(prefix):
            out += float(v.detach().item())
    return out

def extract_components(loss_dict: Dict[str, torch.Tensor], weight_dict: Dict[str, float]) -> Dict[str, float]:
    """
    Returns scalar metrics for logging:
      - total 'loss' (weighted sum using weight_dict)
      - per-component sums including aux: ce, bbox, giou
      - cardinality error (if present)
    """
    total = 0.0
    for k, v in loss_dict.items():
        if k in weight_dict:
            total += float(v.detach().item()) * float(weight_dict[k])

    ce   = _sum_prefixed(loss_dict, "loss_ce")
    bbox = _sum_prefixed(loss_dict, "loss_bbox")
    giou = _sum_prefixed(loss_dict, "loss_giou")

    card = float(loss_dict.get("cardinality_error", torch.tensor(0.0)).detach().item())
    return {"loss": total, "ce": ce, "bbox": bbox, "giou": giou, "card": card}


# ==========================
# ====== PLOTTING ==========
# ==========================
def plot_from_csv(csv_path: str, out_dir: str):
    """
    Reads the epoch-level CSV and writes:
      - loss_total.png
      - loss_ce.png
      - loss_bbox.png
      - loss_giou.png
      - cardinality_error.png
    """
    outp = Path(out_dir)
    outp.mkdir(parents=True, exist_ok=True)

    # Read CSV (no pandas dependency)
    rows = []
    with open(csv_path, "r") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            rows.append({
                "epoch": int(r["epoch"]),
                "split": r["split"],
                "lr": float(r["lr"]),
                "loss": float(r["loss"]),
                "loss_ce": float(r["loss_ce"]),
                "loss_bbox": float(r["loss_bbox"]),
                "loss_giou": float(r["loss_giou"]),
                "cardinality_error": float(r["cardinality_error"]),
            })

    # Split
    tr = [r for r in rows if r["split"] == "train"]
    va = [r for r in rows if r["split"] == "val"]

    def _plot(metric_key: str, title: str, fname: str):
        plt.figure()
        if tr:
            xs = [r["epoch"] for r in tr]
            ys = [r[metric_key] for r in tr]
            plt.plot(xs, ys, label="train")
        if va:
            xs = [r["epoch"] for r in va]
            ys = [r[metric_key] for r in va]
            plt.plot(xs, ys, label="val")
        plt.xlabel("epoch")
        plt.ylabel(metric_key)
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(outp / fname, dpi=150)
        plt.close()

    _plot("loss", "Total Loss", "loss_total.png")
    _plot("loss_ce", "Classification Loss (sum incl. aux)", "loss_ce.png")
    _plot("loss_bbox", "BBox L1 Loss (sum incl. aux)", "loss_bbox.png")
    _plot("loss_giou", "GIoU Loss (sum incl. aux)", "loss_giou.png")
    _plot("cardinality_error", "Cardinality Error", "cardinality_error.png")

    print(f"[plots] saved PNGs to: {outp.resolve()}")