#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .losses import MarginClassificationHead


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: float, dropout: float) -> None:
        super().__init__()
        self.base = base
        for param in self.base.parameters():
            param.requires_grad = False
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.scale = alpha / max(rank, 1)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=5**0.5)
        nn.init.zeros_(self.lora_up.weight)

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.base, name)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + self.lora_up(self.lora_down(self.dropout(x))) * self.scale


def _iter_named_linears(module: nn.Module) -> Iterable[tuple[str, nn.Linear]]:
    for name, child in module.named_modules():
        if isinstance(child, nn.Linear):
            yield name, child


def _replace_module(root: nn.Module, dotted_name: str, new_module: nn.Module) -> None:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = getattr(parent, part)
    setattr(parent, parts[-1], new_module)


def apply_lora_to_linears(
    root: nn.Module,
    rank: int,
    alpha: float,
    dropout: float,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
) -> int:
    include_patterns = include_patterns or []
    exclude_patterns = exclude_patterns or []
    targets: list[tuple[str, nn.Linear]] = []
    for name, linear in _iter_named_linears(root):
        if include_patterns and not any(pattern in name for pattern in include_patterns):
            continue
        if exclude_patterns and any(pattern in name for pattern in exclude_patterns):
            continue
        targets.append((name, linear))
    for name, linear in targets:
        _replace_module(root, name, LoRALinear(linear, rank=rank, alpha=alpha, dropout=dropout))
    return len(targets)


def set_last_ratio_trainable(module: nn.Module, ratio: float) -> None:
    params = list(module.parameters())
    for param in params:
        param.requires_grad = False
    if ratio <= 0 or not params:
        return
    train_count = max(1, int(round(len(params) * ratio)))
    for param in params[-train_count:]:
        param.requires_grad = True


@dataclass
class MultitaskOutput:
    image_embedding: torch.Tensor
    text_embedding: torch.Tensor | None
    logits: torch.Tensor


class MobileClipS4MultitaskModel(nn.Module):
    def __init__(self, cfg: dict, num_classes: int) -> None:
        super().__init__()
        import open_clip

        self.cfg = cfg
        model_cfg = cfg["model"]
        loss_cfg = cfg.get("loss", {}).get("classification", {})
        training_cfg = cfg.get("training", {})
        pretrained = model_cfg.get("pretrained", "dfndr2b")
        self.clip, _, _ = open_clip.create_model_and_transforms(model_cfg["model_name"], pretrained=pretrained)
        self.tokenizer = open_clip.get_tokenizer(model_cfg["model_name"])
        for param in self.clip.parameters():
            param.requires_grad = False

        method = str(model_cfg.get("train_method", "partial_unfreeze"))
        if method == "partial_unfreeze":
            self._configure_partial_unfreeze(model_cfg)
        elif method == "lora":
            self._configure_lora(model_cfg)
        else:
            raise ValueError(f"Unsupported train_method: {method}")

        embedding_dim = int(training_cfg.get("embedding_dim", 512))
        self.image_projection = nn.Sequential(nn.LazyLinear(embedding_dim), nn.LayerNorm(embedding_dim))
        self.text_projection = nn.Sequential(nn.LazyLinear(embedding_dim), nn.LayerNorm(embedding_dim))
        cls_type = str(loss_cfg.get("type", "cross_entropy"))
        self.classifier = MarginClassificationHead(
            embedding_dim,
            num_classes,
            loss_type=cls_type,
            margin=float(loss_cfg.get("margin", 0.2)),
            scale=float(loss_cfg.get("scale", 30.0)),
        )

    def _text_tower(self) -> nn.Module | None:
        for name in ("transformer", "text", "text_model"):
            module = getattr(self.clip, name, None)
            if isinstance(module, nn.Module):
                return module
        return None

    def _configure_partial_unfreeze(self, model_cfg: dict) -> None:
        visual = getattr(self.clip, "visual", None)
        if isinstance(visual, nn.Module):
            set_last_ratio_trainable(visual, float(model_cfg.get("image_unfreeze_ratio", 0.25)))
        text_tower = self._text_tower()
        if text_tower is not None:
            set_last_ratio_trainable(text_tower, float(model_cfg.get("text_unfreeze_ratio", 0.15)))

    def _configure_lora(self, model_cfg: dict) -> None:
        lora_cfg = model_cfg.get("lora", {})
        rank = int(lora_cfg.get("rank", 8))
        alpha = float(lora_cfg.get("alpha", 16))
        dropout = float(lora_cfg.get("dropout", 0.05))
        include_patterns = list(lora_cfg.get("include_patterns", []))
        exclude_patterns = list(lora_cfg.get("exclude_patterns", []))
        visual = getattr(self.clip, "visual", None)
        image_count = apply_lora_to_linears(visual, rank, alpha, dropout, include_patterns, exclude_patterns) if isinstance(visual, nn.Module) else 0
        text_tower = self._text_tower()
        text_count = apply_lora_to_linears(text_tower, rank, alpha, dropout, include_patterns, exclude_patterns) if text_tower is not None else 0
        if image_count + text_count == 0:
            raise RuntimeError("LoRA enabled but no Linear modules were patched")

    def tokenize(self, texts: list[str], device: torch.device) -> torch.Tensor:
        return self.tokenizer(texts).to(device)

    def encode_image_embedding(self, images: torch.Tensor) -> torch.Tensor:
        features = self.clip.encode_image(images)
        return F.normalize(self.image_projection(features.float()), dim=-1)

    def encode_text_embedding(self, tokenized_text: torch.Tensor) -> torch.Tensor:
        features = self.clip.encode_text(tokenized_text)
        return F.normalize(self.text_projection(features.float()), dim=-1)

    def forward(
        self,
        images: torch.Tensor,
        labels: torch.Tensor | None = None,
        tokenized_text: torch.Tensor | None = None,
    ) -> MultitaskOutput:
        image_embedding = self.encode_image_embedding(images)
        text_embedding = self.encode_text_embedding(tokenized_text) if tokenized_text is not None and tokenized_text.numel() else None
        logits = self.classifier(image_embedding, labels)
        return MultitaskOutput(image_embedding=image_embedding, text_embedding=text_embedding, logits=logits)

    def trainable_parameter_groups(self, cfg: dict) -> list[dict]:
        lr_cfg = cfg.get("optimizer", {})
        head_lr = float(lr_cfg.get("head_lr", 1e-3))
        projection_lr = float(lr_cfg.get("projection_lr", 5e-4))
        encoder_lr = float(lr_cfg.get("encoder_lr", lr_cfg.get("image_encoder_lr", 2e-5)))

        head_params = list(self.classifier.parameters())
        projection_params = list(self.image_projection.parameters()) + list(self.text_projection.parameters())
        known = {id(param) for param in head_params + projection_params}
        encoder_params = [param for param in self.clip.parameters() if param.requires_grad and id(param) not in known]
        groups = [
            {"params": [p for p in encoder_params if p.requires_grad], "lr": encoder_lr, "name": "encoder"},
            {"params": [p for p in projection_params if p.requires_grad], "lr": projection_lr, "name": "projection"},
            {"params": [p for p in head_params if p.requires_grad], "lr": head_lr, "name": "head"},
        ]
        return [group for group in groups if group["params"]]
