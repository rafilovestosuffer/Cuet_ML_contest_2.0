import torch
import torch.nn as nn

try:
    import timm
    from transformers import AutoModel
except ImportError:
    timm = None
    AutoModel = None


class JointMM(nn.Module):

    def __init__(self, cfg: dict, n_cls: int = 8):
        super().__init__()
        assert timm is not None and AutoModel is not None, \
            "pip install timm transformers"

        self.img_enc = timm.create_model(
            cfg["img_model"], pretrained=True, num_classes=0,
            drop_path_rate=cfg.get("drop_path", 0.10),
        )
        if hasattr(self.img_enc, "set_grad_checkpointing"):
            try:
                self.img_enc.set_grad_checkpointing(enable=True)
            except Exception:
                pass

        self.text_enc = AutoModel.from_pretrained(cfg["text_model"])
        if hasattr(self.text_enc, "gradient_checkpointing_enable"):
            self.text_enc.gradient_checkpointing_enable()

        fd = cfg["fusion_dim"]
        self.img_proj = nn.Sequential(
            nn.LayerNorm(cfg["img_dim"]), nn.Linear(cfg["img_dim"], fd), nn.GELU())
        self.text_proj = nn.Sequential(
            nn.LayerNorm(cfg["text_dim"]), nn.Linear(cfg["text_dim"], fd), nn.GELU())
        self.cross_attn = nn.MultiheadAttention(
            fd, cfg["n_heads"], dropout=0.1, batch_first=True)
        self.classifier = nn.Sequential(
            nn.LayerNorm(fd * 2), nn.Dropout(0.2), nn.Linear(fd * 2, n_cls))

    def forward(self, image, input_ids, attention_mask, token_type_ids=None):
        img_f = self.img_enc(image)

        kw = dict(input_ids=input_ids, attention_mask=attention_mask)
        if token_type_ids is not None:
            kw["token_type_ids"] = token_type_ids
        t_out = self.text_enc(**kw).last_hidden_state
        m = attention_mask.unsqueeze(-1).float()
        txt_f = (t_out * m).sum(1) / m.sum(1).clamp(min=1e-9)

        img_q = self.img_proj(img_f).unsqueeze(1)
        txt_q = self.text_proj(txt_f).unsqueeze(1)
        attn_out, _ = self.cross_attn(query=txt_q, key=img_q, value=img_q)
        fused = torch.cat([attn_out.squeeze(1), self.text_proj(txt_f)], dim=-1)
        return self.classifier(fused)
