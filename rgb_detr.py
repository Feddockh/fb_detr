import os
import torch
import torch.nn as nn

# Your helper that builds + loads the COCO-pretrained R50 DETR
import detr_r50

NUM_CLASSES = 3  # your foreground classes


def main():
    # 1) Build + load COCO-pretrained DETR-R50
    model, postprocessors = detr_r50.pretrained_detr_r50()
    print("Model built & pretrained weights loaded.")

    # 2) Swap the classifier to 3 classes (+1 background)
    model = detr_r50.adapt_num_classes(model, NUM_CLASSES)


    # 5) (Optional) save this adapted init to resume training from it
    # torch.save({"model": model.state_dict()}, "detr_r50_3class_init.pth")
    # print("\nSaved adapted weights -> detr_r50_3class_init.pth")

if __name__ == "__main__":
    main()
