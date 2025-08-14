import os
import yaml
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms.functional import pil_to_tensor
from typing import List, Tuple
from PIL import Image
from utils import BoxType, read_yolo_labels, to_xywh_norm, to_cxcywh_norm, to_xyxy_norm
from vis import show_pair


class SetType:
    TRAIN = "train"; VAL = "val"; TEST = "test"

class YoloV5MultiCamDataset(Dataset):
    def __init__(self, base_dir, cam0, cam1, set_type, transforms=None, return_type=BoxType.XYXY):
        """
        This is the initialization method for the YoloV5MultiCamDataset class.
        The expected input is a base directory containing datasets for both 
        cam0 and cam1. The labels from the cam0 dataset will be used to find the
        images, labels, and classes for both cameras. The image from both cameras
        will be retrieved along with the label from camera 0.

        dataset
        - firefly_left (cam0)
            - images
                - train
                - val
            - labels
                - train
                - val
            train.txt
            val.txt
            data.yaml
        - ximea (cam1)
            - images
                - train
                - val
        """

        self.base_dir    = base_dir
        self.cam0        = cam0
        self.cam1        = cam1
        self.set_type    = set_type
        self.transforms  = transforms
        self.return_type = return_type
        self.classes     = []

        set_type_txt = os.path.join(base_dir, cam0, f"{set_type}.txt")
        if not os.path.exists(set_type_txt):
            raise FileNotFoundError(set_type_txt)
        with open(set_type_txt) as f:
            self.image_paths = [ln.strip() for ln in f]

        classes_txt = os.path.join(base_dir, cam0, "classes.txt")
        data_yaml = os.path.join(base_dir, cam0, "data.yaml")
        if os.path.exists(classes_txt):
            with open(classes_txt) as f:
                self.classes = [ln.strip() for ln in f]
        elif os.path.exists(data_yaml):
            with open(data_yaml) as f:
                data = yaml.safe_load(f)
                self.classes = list(data['names'].values())
                print(f"Loaded classes from {data_yaml}: {self.classes}")
        else:
            raise FileNotFoundError("No class labels found.")

    def __len__(self):  
        return len(self.image_paths)
    
    def __load_pair(self, img_path: str):
        # Get just the image filename
        img_filename = os.path.basename(img_path)

        # Find the corresponding image paths and label path
        img0_path = os.path.join(self.base_dir, self.cam0, "images", self.set_type, img_filename)
        img1_path = os.path.join(self.base_dir, self.cam1, "images", self.set_type, img_filename)
        lbl_path =  img0_path.replace("images", "labels").rsplit(".", 1)[0] + ".txt"

        # Load images as PIL and labels as List
        img0 = Image.open(img0_path).convert("RGB")
        img1 = Image.open(img1_path).convert("RGB")
        lbls = read_yolo_labels(lbl_path)

        return img0, img1, lbls, img0_path, img1_path, lbl_path

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img0, img1, lbls, img0_path, img1_path, lbl_path = self.__load_pair(img_path)

        
        # Convert the boxes to the desired format
        boxes_t = torch.as_tensor(lbls[0], dtype=torch.float32)
        if self.return_type == BoxType.XYXY:
            boxes_t = to_xyxy_norm(boxes_t)
        elif self.return_type == BoxType.XYWH:
            boxes_t = to_xywh_norm(boxes_t)
        elif self.return_type == BoxType.CXCYWH:
            boxes_t = to_cxcywh_norm(boxes_t)
        else:
            raise ValueError(f"Unknown return type: {self.return_type}")
        
        # Convert all to tensors
        img0 = pil_to_tensor(img0).float() / 255.0
        img1 = pil_to_tensor(img1).float() / 255.0
        boxes_t = boxes_t if len(lbls[0]) > 0 else torch.zeros((0,4), dtype=torch.float32)
        classes_t = torch.as_tensor(lbls[1], dtype=torch.int64) if len(lbls[1]) > 0 else torch.zeros((0,), dtype=torch.int64)


        # Create the target dictionary
        target = {
            "boxes": boxes_t,
            "labels": classes_t
        }

        if self.transforms:
            img0, img1, target = self.transforms(img0, img1, target)

        return {
            "img0": img0,
            "img1": img1,
            "target":  target
        }

def multicam_collate_fn(batch):
    imgs0 = [b["img0"] for b in batch]
    imgs1 = [b["img1"] for b in batch]
    targets = [b["target"] for b in batch]
    return {
        "img0": imgs0,
        "img1": imgs1,
        "target": targets
    }

def main():
    dataset = YoloV5MultiCamDataset(
        base_dir="rivendale_v4",
        cam0="firefly_left",
        cam1="ximea_demosaic",
        set_type=SetType.TRAIN,
        return_type=BoxType.CXCYWH
    )
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True, collate_fn=multicam_collate_fn)
    print("Number of batches:", len(dataloader))
    for batch in dataloader:
        print(batch)
        break
    
    for i in range(len(batch["img0"])):
        show_pair(batch["img0"][i], batch["img1"][i], batch["target"][i]["boxes"], batch["target"][i]["labels"], 
                  class_names=dataset.classes, box_type=BoxType.CXCYWH)
        break

if __name__ == "__main__":
    main()
