# visualize_dual_detr.py
import torch
from torchviz import make_dot

# NestedTensor helper from DETR
from detr.util.misc import nested_tensor_from_tensor_list

# Your helpers
import detr_r50
from dual_detr_r50 import DualDetrCrossEnc

# --------- config ----------
NUM_CLASSES = 7          # set to the # of foreground classes you trained with
H, W        = 224, 224   # keep small for visualization
DEVICE      = "cpu"      # keep on CPU for TorchViz graph

def build_model():
    # Build COCO-pretrained DETR-R50 and adapt head
    base_detr, _ = detr_r50.pretrained_detr_r50()
    base_detr = detr_r50.adapt_num_classes(base_detr, num_classes=NUM_CLASSES)

    # Wrap into your dual-stream model (reuses pretrained parts)
    model = DualDetrCrossEnc(base_detr, num_classes=NUM_CLASSES, nhead=base_detr.transformer.nhead)
    model.to(DEVICE).eval()
    return model

def dummy_inputs():
    """
    Create dummy RGB + NIR images.
    Your model expects lists of (C,H,W) tensors wrapped via nested_tensor_from_tensor_list.
    """
    img_rgb = torch.randn(3, H, W, device=DEVICE)
    img_nir = torch.randn(3, H, W, device=DEVICE)  # NIR duplicated to 3ch (as your backbone expects 3 channels)
    samples_rgb = nested_tensor_from_tensor_list([img_rgb])  # batch of 1
    samples_nir = nested_tensor_from_tensor_list([img_nir])  # batch of 1
    return samples_rgb, samples_nir

def main():
    model = build_model()
    samples_rgb, samples_nir = dummy_inputs()

    # Forward pass — your model returns a dict with 'pred_logits' and 'pred_boxes'
    outputs = model(samples_rgb, samples_nir)

    # TorchViz needs a single tensor. Tie the graph to both heads by summing them.
    y = outputs["pred_logits"].sum() + outputs["pred_boxes"].sum()

    # Make and save the graph
    dot = make_dot(y, params=dict(model.named_parameters()))
    dot.format = "png"
    out_path = dot.render("dual_detr_graph")  # writes dual_detr_graph.png
    print(f"Graph saved to: {out_path}")

if __name__ == "__main__":
    main()
