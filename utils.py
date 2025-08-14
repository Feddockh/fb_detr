import os
import torch
from typing import List, Tuple
from PIL import Image

def read_yolo_labels(label_path: str) -> Tuple[List[List[float]], List[int]]:
    """
    Reads a YOLO .txt label file.
    Returns:
        boxes_xyxy_norm: list of [x0,y0,x1,y1] in NORMALIZED [0,1] coords
        class_ids:       list of int
    """
    boxes, class_ids = [], []
    if not os.path.exists(label_path):
        # no labels -> empty target
        return boxes, class_ids

    with open(label_path, "r") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5 and len(parts) != 6:
                # YOLOv5 sometimes stores cls xc yc w h [optional_conf]
                if len(parts) < 5:
                    continue
                parts = parts[:5]
            cls, xc, yc, w, h = map(float, parts[:5])
            x0, y0 = (xc - w / 2), (yc - h / 2)
            x1, y1 = (xc + w / 2), (yc + h / 2)
            # clamp to [0,1] just in case
            x0, y0 = max(0.0, x0), max(0.0, y0)
            x1, y1 = min(1.0, x1), min(1.0, y1)
            boxes.append([x0, y0, x1, y1])
            class_ids.append(int(cls))
    return boxes, class_ids

class BoxType:
    XYXY = "xyxy"; XYWH = "xywh"; CXCYWH = "cxcywh"

def to_xywh_norm(boxes: torch.Tensor, input_type: BoxType = BoxType.XYXY) -> torch.Tensor:
    """
    boxes: (N,4) normalized [0,1]
    returns xywh normalized [0,1]
    """
    if input_type == BoxType.XYXY:
        x, y, x1, y1 = boxes.unbind(-1)
        w = (x1 - x).clamp(min=0.0)
        h = (y1 - y).clamp(min=0.0)
    elif input_type == BoxType.XYWH:
        x, y, w, h = boxes.unbind(-1)
    elif input_type == BoxType.CXCYWH:
        cx, cy, w, h = boxes.unbind(-1)
        x = (cx - w / 2).clamp(min=0.0)
        y = (cy - h / 2).clamp(min=0.0)
    else:
        raise ValueError(f"Unknown box type: {input_type}")

    return torch.stack([x, y, w, h], dim=-1)

def to_cxcywh_norm(boxes: torch.Tensor, input_type: BoxType = BoxType.XYXY) -> torch.Tensor:
    """
    boxes: (N,4) normalized [0,1]
    returns cxcywh normalized [0,1]
    """
    if input_type == BoxType.XYXY:
        x0, y0, x1, y1 = boxes.unbind(-1)
        w = (x1 - x0).clamp(min=0.0)
        h = (y1 - y0).clamp(min=0.0)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
    elif input_type == BoxType.XYWH:
        x, y, w, h = boxes.unbind(-1)
        cx = x + w / 2
        cy = y + h / 2
    elif input_type == BoxType.CXCYWH:
        cx, cy, w, h = boxes.unbind(-1)
    else:
        raise ValueError(f"Unknown box type: {input_type}")

    return torch.stack([cx, cy, w, h], dim=-1)

def to_xyxy_norm(boxes: torch.Tensor, input_type: BoxType = BoxType.XYXY) -> torch.Tensor:
    """
    boxes: (N,4) normalized [0,1]
    returns xyxy normalized [0,1]
    """
    if input_type == BoxType.XYXY:
        x0, y0, x1, y1 = boxes.unbind(-1)
    elif input_type == BoxType.XYWH:
        x0, y0, w, h = boxes.unbind(-1)
        x1 = x0 + w
        y1 = y0 + h
    elif input_type == BoxType.CXCYWH:
        cx, cy, w, h = boxes.unbind(-1)
        x0 = (cx - w / 2).clamp(min=0.0)
        y0 = (cy - h / 2).clamp(min=0.0)
        x1 = (cx + w / 2).clamp(min=0.0)
        y1 = (cy + h / 2).clamp(min=0.0)
    else:
        raise ValueError(f"Unknown box type: {input_type}")

    return torch.stack([x0, y0, x1, y1], dim=-1)

def pil_to_tensor(img: Image.Image) -> torch.Tensor:
    """
    Converts a PIL image to a PyTorch tensor.
    """
    if img.mode == "RGB":
        img = img.convert("RGB")
    else:
        if img.mode != "L":
            img = img.convert("L")
    arr = torch.from_numpy(
        (torch.ByteTensor(bytearray(img.tobytes())).numpy()
         if False else
         # Safer path using numpy from PIL (no copy if possible)
         __import__("numpy").array(img))  # HxW[xC]
    )
    # convert to CHW float
    if img.mode == "RGB":
        t = torch.as_tensor(arr).permute(2, 0, 1).float().div_(255.0)
    elif img.mode == "L":
        t = torch.as_tensor(arr).unsqueeze(0).float().div_(255.0)
    else:
        # fallback
        t = torch.as_tensor(arr).permute(2, 0, 1).float().div_(255.0)
    return t

