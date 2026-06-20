#!/usr/bin/env python3
from __future__ import annotations

import math
from typing import Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


class MarginClassificationHead(nn.Module):
    def __init__(
        self,
        embedding_dim: int,
        num_classes: int,
        loss_type: Literal["cross_entropy", "cosface", "arcface"] = "cross_entropy",
        margin: float = 0.2,
        scale: float = 30.0,
    ) -> None:
        super().__init__()
        self.loss_type = loss_type
        self.margin = margin
        self.scale = scale
        self.weight = nn.Parameter(torch.empty(num_classes, embedding_dim))
        self.bias = nn.Parameter(torch.zeros(num_classes)) if loss_type == "cross_entropy" else None
        nn.init.xavier_uniform_(self.weight)

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor | None = None) -> torch.Tensor:
        if self.loss_type == "cross_entropy":
            return F.linear(embeddings, self.weight, self.bias)

        cosine = F.linear(F.normalize(embeddings), F.normalize(self.weight))
        if labels is None:
            return cosine * self.scale

        one_hot = torch.zeros_like(cosine)
        one_hot.scatter_(1, labels.view(-1, 1), 1.0)
        if self.loss_type == "cosface":
            target = cosine - self.margin
        elif self.loss_type == "arcface":
            sine = torch.sqrt(torch.clamp(1.0 - cosine.pow(2), min=1e-7))
            target = cosine * math.cos(self.margin) - sine * math.sin(self.margin)
        else:
            raise ValueError(f"Unsupported classification loss type: {self.loss_type}")
        logits = one_hot * target + (1.0 - one_hot) * cosine
        return logits * self.scale


class MultiPositiveImageTextContrastiveLoss(nn.Module):
    def __init__(self, temperature: float = 0.07) -> None:
        super().__init__()
        self.temperature = temperature

    def forward(
        self,
        image_embeddings: torch.Tensor,
        text_embeddings: torch.Tensor,
        image_labels: torch.Tensor,
        text_labels: torch.Tensor,
    ) -> torch.Tensor:
        if text_embeddings.numel() == 0 or text_labels.numel() == 0:
            return image_embeddings.sum() * 0.0

        image_embeddings = F.normalize(image_embeddings, dim=-1)
        text_embeddings = F.normalize(text_embeddings, dim=-1)
        logits = image_embeddings @ text_embeddings.t()
        logits = logits / self.temperature
        positive_mask = image_labels.view(-1, 1).eq(text_labels.view(1, -1))
        if not positive_mask.any():
            return logits.sum() * 0.0

        image_loss = self._masked_row_loss(logits, positive_mask)
        text_loss = self._masked_row_loss(logits.t(), positive_mask.t())
        return 0.5 * (image_loss + text_loss)

    @staticmethod
    def _masked_row_loss(logits: torch.Tensor, positive_mask: torch.Tensor) -> torch.Tensor:
        row_has_positive = positive_mask.any(dim=1)
        if not row_has_positive.any():
            return logits.sum() * 0.0
        selected_logits = logits[row_has_positive]
        selected_mask = positive_mask[row_has_positive]
        log_denominator = torch.logsumexp(selected_logits, dim=1)
        positive_logits = selected_logits.masked_fill(~selected_mask, float("-inf"))
        log_numerator = torch.logsumexp(positive_logits, dim=1)
        return -(log_numerator - log_denominator).mean()


class HardNegativeLogitMarginLoss(nn.Module):
    def __init__(self, margin: float = 0.1) -> None:
        super().__init__()
        self.margin = margin

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        hard_negative_indices: list[list[int]],
    ) -> torch.Tensor:
        losses: list[torch.Tensor] = []
        for row_index, negatives in enumerate(hard_negative_indices):
            if not negatives:
                continue
            true_logit = logits[row_index, labels[row_index]]
            negative_tensor = torch.tensor(negatives, dtype=torch.long, device=logits.device)
            negative_logits = logits[row_index].index_select(0, negative_tensor)
            losses.append(F.relu(self.margin - true_logit + negative_logits).mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()
