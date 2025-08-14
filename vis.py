import torch
from typing import List, Dict, Any
from PIL import Image
import matplotlib.pyplot as plt
from utils import BoxType, to_xyxy_norm

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
