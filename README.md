Install the pretrained detr r50 weights from https://dl.fbaipublicfiles.com/detr/detr-r50-e632da11.pth

Setup the conda env by creating an env, activating it, and then pip install -r requirements.txt


                 ┌──────────────────────────────────────────────────────────────┐
                 │                          INPUT                              │
                 │  cam0: RGB image (B,3,H,W)          cam1: NIR→3ch (B,3,H,W) │
                 └──────────────┬───────────────────────┬───────────────────────┘
                                │                       │
                                ▼                       ▼
     ┌─────────────────────────────────┐  ┌─────────────────────────────────┐
     │     Backbone_RGB (ResNet50)     │  │     Backbone_NIR (ResNet50)     │
     │  + PositionEmbeddingSine (pe)   │  │  + PositionEmbeddingSine (pe)   │
     │  outputs last level only:       │  │  outputs last level only:       │
     │  src_r: (B,2048,H',W')          │  │  src_n: (B,2048,H',W')          │
     │  mask_r: (B,H',W')              │  │  mask_n: (B,H',W')              │
     │  pos_r: (B,256,H',W')           │  │  pos_n: (B,256,H',W')           │
     └───────────┬─────────────────────┘  └───────────┬─────────────────────┘
                 │ shared 1×1 conv (copied from DETR) │
                 ▼                                     ▼
       input_proj(src_r)                      input_proj(src_n)
          (B,256,H',W')                          (B,256,H',W')
                 │                                     │
                 │   flatten HW → S=H'·W'              │
                 ▼                                     ▼
  src_r_seq: (S,B,256)     mask_r_seq: (B,S)  src_n_seq: (S,B,256)   mask_n_seq: (B,S)
  pos_r_seq: (S,B,256)                           pos_n_seq: (S,B,256)
                 │                                     │
                 └──────────────┬──────────────────────┘
                                ▼
               ┌───────────────────────────────────────────┐
               │       Shared Transformer ENCODER          │
               │  (6 layers, d=256, nhead=8, pre-norm)     │
               │  memory_r = Encoder(src_r_seq, ...)       │
               │             → (S,B,256)                   │
               │  memory_n = Encoder(src_n_seq, ...)       │
               │             → (S,B,256)                   │
               └──────────────┬───────────────┬────────────┘
                              │               │
                       to (B,S,256)     to (B,S,256)
                              │               │
                              ▼               ▼
               ┌───────────────────────────────────────────┐
               │         CrossEncFuse (new block)          │
               │  MHA (Q = mem_r + pos_r,                  │
               │       K,V = mem_n + pos_n, mask = mask_n) │
               │  + residual + FFN + LayerNorm             │
               │  out: fused_r (B,S,256)                   │
               └──────────────┬────────────────────────────┘
                              │  to (S,B,256)
                              ▼
                   fused_r_seq (S,B,256), mask_r_seq (B,S), pos_r_seq (S,B,256)

                      ┌────────────────────────────────────────────┐
                      │     Shared Transformer DECODER (6 layers)  │
                      │  inputs:                                    │
                      │    - tgt = zeros (Q,B,256)                  │
                      │    - query_pos = learned queries (Q,B,256)  │
                      │    - memory = fused_r_seq (S,B,256)         │
                      │    - key padding mask = mask_r_seq (B,S)    │
                      │    - pos = pos_r_seq (S,B,256)              │
                      │  output: hs (L,B,Q,256)                     │
                      └──────────────┬──────────────────────────────┘
                                     │
                          heads per layer (final used + aux)
                                     │
         ┌───────────────────────────┴─────────────────────────────┐
         │                    Prediction Heads                      │
         │ class_embed: Linear(256 → K+1) -> (L,B,Q,K+1)           │
         │ bbox_embed : MLP(256 → 4) + sigmoid -> (L,B,Q,4)        │
         └───────────────────┬─────────────────────────────────────┘
                             │ take last layer (and aux if enabled)
                             ▼
             outputs["pred_logits"]: (B,Q,K+1)  softmax over classes
             outputs["pred_boxes"] : (B,Q,4)    (cx,cy,w,h) ∈ [0,1]

