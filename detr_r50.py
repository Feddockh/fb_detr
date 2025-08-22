import os
import torch
import torch.nn as nn

# Imports from the official DETR repo
from detr.models.backbone import Backbone, Joiner
from detr.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from detr.models.transformer import Transformer
from detr.models.detr import DETR, PostProcess, DETRsegm  # (segm unused here)

# ------------------------
# Simple config (no argparse)
# ------------------------
class Cfg:
    # dataset / classes
    dataset_file       = "coco"   # keeps num_classes=91 to match COCO pretrained
    num_classes        = 91       # COCO uses max class id + 1 = 91
    # model
    backbone           = "resnet50"
    hidden_dim         = 256
    position_embedding = "sine"   # 'sine' or 'learned'
    nheads             = 8
    dim_feedforward    = 2048
    enc_layers         = 6
    dec_layers         = 6
    pre_norm           = False
    dropout            = 0.1
    dilation           = False
    masks              = False
    num_queries        = 100
    aux_loss           = True
    # loss (not strictly needed just to print model)
    bbox_loss_coef     = 5.0
    giou_loss_coef     = 2.0
    eos_coef           = 0.1
    # lr backbone flag only affects trainability, not structure
    lr_backbone        = 1e-5
    device             = "cuda" if torch.cuda.is_available() else "cpu"

CFG = Cfg()

# ------------------------
# Builders (no argparse)
# ------------------------
def build_position_encoding(cfg: Cfg):
    d_half = cfg.hidden_dim // 2
    if cfg.position_embedding in ("v2", "sine"):
        return PositionEmbeddingSine(d_half, normalize=True)
    elif cfg.position_embedding in ("v3", "learned"):
        return PositionEmbeddingLearned(d_half)
    else:
        raise ValueError(f"Unsupported position embedding: {cfg.position_embedding}")

def build_backbone(cfg: Cfg):
    position_embedding = build_position_encoding(cfg)
    train_backbone = cfg.lr_backbone > 0
    return_interm_layers = cfg.masks  # only used for segmentation variants
    backbone = Backbone(cfg.backbone, train_backbone, return_interm_layers, cfg.dilation)
    model = Joiner(backbone, position_embedding)
    model.num_channels = backbone.num_channels
    return model

def build_transformer(cfg: Cfg):
    return Transformer(
        d_model=cfg.hidden_dim,
        dropout=cfg.dropout,
        nhead=cfg.nheads,
        dim_feedforward=cfg.dim_feedforward,
        num_encoder_layers=cfg.enc_layers,
        num_decoder_layers=cfg.dec_layers,
        normalize_before=cfg.pre_norm,
        return_intermediate_dec=True,
    )

def build_detr(cfg: Cfg):
    device = torch.device(cfg.device)

    backbone     = build_backbone(cfg)
    transformer  = build_transformer(cfg)

    model = DETR(
        backbone=backbone,
        transformer=transformer,
        num_classes=cfg.num_classes,
        num_queries=cfg.num_queries,
        aux_loss=cfg.aux_loss,
    )
    model.to(device)

    postprocessors = {'bbox': PostProcess()}
    return model, postprocessors

def build_detr_for_finetune(num_classes, cfg: Cfg = CFG):
    """
    Builds a DETR model with a specified number of classes, intended for
    fine-tuning where the checkpoint may have a different number of classes.
    """
    # We can override the default config's num_classes here
    cfg.num_classes = num_classes
    return build_detr(cfg)

# ------------------------
# Weights loader (URL or local path)
# ------------------------
def load_pretrained(model: nn.Module, weights: str = "detr-r50-e632da11.pth"):
    """
    Loads COCO-pretrained DETR weights. Accepts URL or local .pth path.
    Uses strict=False so mismatched heads are ignored if num_classes differs.
    """
    if not os.path.isfile(weights):
        raise FileNotFoundError(f"Checkpoint not found: {weights}")
    print(f"Loading checkpoint from {weights} ...")
    ckpt = torch.load(weights, map_location="cpu")

    # official checkpoints store the model state under 'model'
    state_dict = ckpt.get("model", ckpt)

    # If the number of classes in the checkpoint does not match the model,
    # we pop the class_embed weights from the state_dict to avoid size mismatch errors.
    model_num_classes = model.class_embed.weight.shape[0]
    ckpt_num_classes = state_dict['class_embed.weight'].shape[0]
    if model_num_classes != ckpt_num_classes:
        print(f"  ↳ Deleting class_embed from checkpoint (model: {model_num_classes}, ckpt: {ckpt_num_classes})")
        del state_dict['class_embed.weight']
        del state_dict['class_embed.bias']

    incompat = model.load_state_dict(state_dict, strict=False)
    print("Loaded pretrained weights with strict=False")
    if hasattr(incompat, "missing_keys"):
        print("  Missing keys:", incompat.missing_keys)
    if hasattr(incompat, "unexpected_keys"):
        print("  Unexpected keys:", incompat.unexpected_keys)

def pretrained_detr_r50(weights="detr-r50-e632da11.pth", num_classes=None):
    if num_classes is not None:
        model, postprocessors = build_detr_for_finetune(num_classes)
    else:
        model, postprocessors = build_detr(CFG)
    load_pretrained(model, weights=weights)
    return model, postprocessors

def adapt_num_classes(model: nn.Module, num_classes: int):
    """Replace the DETR classification head to predict num_classes + 1 (background)."""
    in_dim = model.class_embed.in_features
    new_head = nn.Linear(in_dim, num_classes + 1)

    # fresh init for the new head (common TL practice)
    nn.init.xavier_uniform_(new_head.weight)
    nn.init.constant_(new_head.bias, 0.0)

    model.class_embed = new_head
    return model

# ------------------------
# Main
# ------------------------
def main():
    print("Building DETR ...")
    model, post = pretrained_detr_r50()
    print("Model built.")

    # Print the full model
    print("\n=== DETR Model ===")
    print(model)

if __name__ == "__main__":
    main()
