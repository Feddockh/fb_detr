# dual_detr_crossenc.py
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from detr.util.misc import NestedTensor, nested_tensor_from_tensor_list
import detr_r50


class CrossEncFuse(nn.Module):
    """
    One residual cross-attention + FFN block:
    RGB memory attends to NIR memory. Shapes are (B, S, D).
    """
    def __init__(self, d_model=256, nhead=8, dropout=0.1, ffn_dim=2048):
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.ln1  = nn.LayerNorm(d_model)
        self.ffn  = nn.Sequential(
            nn.Linear(d_model, ffn_dim), nn.ReLU(inplace=True),
            nn.Linear(ffn_dim, d_model), nn.Dropout(dropout),
        )
        self.ln2  = nn.LayerNorm(d_model)

    @staticmethod
    def _with_pos(x, pos):
        return x if pos is None else x + pos

    def forward(self, mem_rgb, mem_nir, pos_rgb=None, pos_nir=None, nir_key_padding_mask=None):
        """
        mem_rgb: (B,Sr,D)  — queries
        mem_nir: (B,Sn,D)  — keys/values
        pos_*:   (B,S*,D)  — positional encodings for each stream (optional but recommended)
        nir_key_padding_mask: (B,Sn) True=pad
        """
        q = self._with_pos(mem_rgb, pos_rgb)
        k = self._with_pos(mem_nir, pos_nir)
        v = mem_nir
        attn_out, _ = self.attn(q, k, v, key_padding_mask=nir_key_padding_mask)
        x = self.ln1(mem_rgb + attn_out)
        x = self.ln2(x + self.ffn(x))
        return x  # (B,Sr,D)


class DualDetrCrossEnc(nn.Module):
    """
    Build a 2-stream DETR that:
      - runs two backbones (RGB & NIR) -> input_proj -> encoder (shared weights)
      - cross-attends RGB memory to NIR memory (1 block)
      - decodes on the fused memory with the stock DETR decoder
    Everything except the fuse block is initialized from a pretrained DETR.
    """

    def __init__(self, base_detr: nn.Module, num_classes: int = 3, nhead: int = 8):
        super().__init__()

        # Clone / reuse pretrained parts
        # Backbones (two copies so each modality can specialize)
        self.backbone_rgb = copy.deepcopy(base_detr.backbone)
        self.backbone_nir = copy.deepcopy(base_detr.backbone)

        # Position enc is inside the Joiner(backbone, posenc) already
        # Input projection 1x1 (we can share one for both streams)
        self.input_proj   = copy.deepcopy(base_detr.input_proj)

        # Transformer encoder/decoder (shared)
        self.encoder = copy.deepcopy(base_detr.transformer.encoder)
        self.decoder = copy.deepcopy(base_detr.transformer.decoder)
        self.d_model = base_detr.transformer.d_model

        # Query embeddings
        self.query_embed = copy.deepcopy(base_detr.query_embed)

        # Heads (re-init class head to (num_classes+1); keep bbox head)
        in_dim = base_detr.class_embed.in_features
        self.class_embed = nn.Linear(in_dim, num_classes + 1)
        nn.init.xavier_uniform_(self.class_embed.weight)
        nn.init.constant_(self.class_embed.bias, 0.0)

        self.bbox_embed  = copy.deepcopy(base_detr.bbox_embed)

        # Aux loss the same as base
        self.aux_loss = getattr(base_detr, "aux_loss", True)

        # New: cross-encoder fusion block (train from scratch)
        self.fuse = CrossEncFuse(d_model=self.d_model, nhead=nhead, dropout=0.1, ffn_dim=2048)

    # -----------------------------
    # helpers to flatten / unflatten
    # -----------------------------
    @staticmethod
    def _decompose(nested: NestedTensor):
        # Take last level feature map, as in stock DETR
        feats, pos_list = nested
        src, mask = feats[-1].decompose()     # src: (B,C,H,W), mask: (B,H,W)
        pos = pos_list[-1]                    # (B,C,H,W)
        return src, mask, pos

    @staticmethod
    def _flatten(src, mask, pos):
        """
        Convert (B,C,H,W) to sequences for transformer:
        returns src_seq, mask_seq, pos_seq with shapes:
            (S,B,D), (B,S), (S,B,D)   — following DETR internals
        """
        B, C, H, W = src.shape
        src_seq = src.flatten(2).permute(2, 0, 1)          # (S,B,C)
        pos_seq = pos.flatten(2).permute(2, 0, 1)          # (S,B,C)
        mask_seq = mask.flatten(1) if mask is not None else torch.zeros((B, H*W), dtype=torch.bool, device=src.device)
        return src_seq, mask_seq, pos_seq

    @staticmethod
    def _to_batch_first(x):
        # (S,B,D) -> (B,S,D)
        return x.transpose(0, 1)

    @staticmethod
    def _to_seq_first(x):
        # (B,S,D) -> (S,B,D)
        return x.transpose(0, 1)

    def encode_backbone(self, samples: NestedTensor):
        """
        samples: NestedTensor of images; run backbone -> input_proj -> encoder
        returns: memory(S,B,D), mask(B,S), pos(S,B,D), plus shapes (B,C,H,W) for decoder reshaping if needed
        """
        # backbone Joiner returns (features, pos) lists
        features, pos = samples
        src, mask, pos_ = self._decompose((features, pos))                 # last level
        src_proj = self.input_proj(src)                                    # (B,D,H,W)
        src_seq, mask_seq, pos_seq = self._flatten(src_proj, mask, pos_)   # (S,B,D), (B,S), (S,B,D)
        memory = self.encoder(src_seq, src_key_padding_mask=mask_seq, pos=pos_seq)  # (S,B,D)
        return memory, mask_seq, pos_seq, src_proj.shape  # src_proj shape = (B,D,H,W)

    def forward(self, samples_rgb: NestedTensor, samples_nir: NestedTensor):
        """
        Each samples_* should be the output of backbone Joiner input type; the **easiest**
        is to pass NestedTensors like stock DETR does: nested_tensor_from_tensor_list([tensor,...]).
        If you have raw tensors (B,3,H,W), call nested_tensor_from_tensor_list first.
        """
        device = self.query_embed.weight.device

        # --- Backbone + Encoder (two streams) ---
        feats_rgb, poses_rgb = self.backbone_rgb(samples_rgb)  # lists
        feats_nir, poses_nir = self.backbone_nir(samples_nir)

        # Wrap to match what _decompose expects
        mem_r, mask_r, pos_r, _ = self.encode_backbone((feats_rgb, poses_rgb))
        mem_n, mask_n, pos_n, _ = self.encode_backbone((feats_nir, poses_nir))

        # --- Cross-encoder fusion (RGB attends to NIR) ---
        mem_r_b = self._to_batch_first(mem_r)  # (B,Sr,D)
        mem_n_b = self._to_batch_first(mem_n)  # (B,Sn,D)
        pos_r_b = self._to_batch_first(pos_r)
        pos_n_b = self._to_batch_first(pos_n)

        fused_b = self.fuse(mem_r_b, mem_n_b, pos_rgb=pos_r_b, pos_nir=pos_n_b,
                            nir_key_padding_mask=mask_n)      # (B,Sr,D)
        fused   = self._to_seq_first(fused_b)                 # (Sr,B,D)

        # --- Decode on fused memory (stock DETR decoder) ---
        bs = fused.shape[1]
        num_queries = self.query_embed.weight.shape[0]
        tgt = torch.zeros(num_queries, bs, self.d_model, device=device)    # (Q,B,D)
        query_pos = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)  # (Q,B,D)

        hs = self.decoder(tgt, fused,
                          memory_key_padding_mask=mask_r,  # use RGB mask (Sr)
                          pos=pos_r, query_pos=query_pos)  # (L,Q,B,D)
        
        # Normalize to (L, B, Q, D) so everything downstream matches the matcher
        hs = hs.permute(0,2,1,3).contiguous() # (L,B,Q,D)

        # heads
        outputs_class = self.class_embed(hs)               # (L,B,Q,C+1)
        outputs_coord = self.bbox_embed(hs).sigmoid()      # (L,B,Q,4)

        out = {
            "pred_logits": outputs_class[-1],  # (B,Q,C+1)
            "pred_boxes":  outputs_coord[-1]   # (B,Q,4)
        }
        if self.aux_loss:
            out["aux_outputs"] = [
                {
                    "pred_logits": outputs_class[l],
                    "pred_boxes":  outputs_coord[l],
                } for l in range(outputs_class.shape[0] - 1)
            ]
        return out

    # -------------------------
    # weight loading convenience
    # -------------------------
    @classmethod
    def from_pretrained(cls, num_classes=3):
        """
        Build from your COCO-pretrained DETR-R50 helper and adapt classifier to K=3.
        Only the fuse block is newly initialized; everything else inherits pretrained weights.
        """
        base, _ = detr_r50.pretrained_detr_r50()  # loads COCO R50 weights
        model = cls(base, num_classes=num_classes, nhead=base.transformer.nhead)
        return model


# -----------------------------
# simple usage with your dataset
# -----------------------------
if __name__ == "__main__":
    # Build the dual model (K=3)
    dual = DualDetrCrossEnc.from_pretrained(num_classes=3)
    dual = dual.cuda() if torch.cuda.is_available() else dual
    print("DualDetrCrossEnc ready.")

    # Decoder parameters
    for n, p in dual.named_parameters():
        if "decoder" in n:
            print(n, p.shape, "trainable=", p.requires_grad)
    

