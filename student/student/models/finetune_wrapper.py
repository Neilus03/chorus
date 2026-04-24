"""Fine-tuning wrapper with learnable granularity prompt.

Wraps a pretrained :class:`StudentInstanceSegModel` (with continuous decoder)
and introduces a learnable scalar ``g_ft_logit`` that controls the granularity
condition during fine-tuning on downstream GT (e.g. ScanNet).

Uses sigmoid reparameterization instead of ``torch.clamp`` to ensure smooth
gradients everywhere in the scale space.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class FineTuningWrapper(nn.Module):
    """Wraps a pretrained model and adds a learnable granularity prompt.

    Parameters
    ----------
    pretrained_model:
        A :class:`StudentInstanceSegModel` with a
        :class:`ContinuousQueryInstanceDecoder`.
    init_g:
        Initial granularity value in (0, 1). Converted to logit space
        via ``logit(init_g) = log(init_g / (1 - init_g))``.
        Default 0.5 → logit = 0.0.
    backbone_lr_scale:
        LR multiplier for backbone parameters (default 0.01 = 100× reduction).
    """

    def __init__(
        self,
        pretrained_model: nn.Module,
        init_g: float = 0.5,
        backbone_lr_scale: float = 0.01,
    ) -> None:
        super().__init__()
        self.model = pretrained_model
        self._backbone_lr_scale = backbone_lr_scale

        # Sigmoid reparameterization: sigmoid(logit) = init_g
        # For init_g=0.5 → logit=0.0; for init_g=0.3 → logit≈-0.847
        init_logit = torch.log(torch.tensor(init_g / (1.0 - init_g)))
        self.g_ft_logit = nn.Parameter(init_logit.clone())

    @property
    def learned_granularity(self) -> float:
        """Current granularity value (for logging/inspection)."""
        return torch.sigmoid(self.g_ft_logit).item()

    def parameter_groups(self, backbone_lr_scale: float | None = None) -> list[dict]:
        """Return param groups with separate LR scaling.

        Three groups:
          1. Backbone parameters (very low LR)
          2. Decoder parameters (normal LR)
          3. g_ft_logit (normal LR)
        """
        if backbone_lr_scale is None:
            backbone_lr_scale = self._backbone_lr_scale

        backbone_params = list(self.model.backbone.parameters())
        decoder_params = list(self.model.decoder.parameters())

        return [
            {"params": backbone_params, "lr_scale": backbone_lr_scale},
            {"params": decoder_params, "lr_scale": 1.0},
            {"params": [self.g_ft_logit], "lr_scale": 1.0},
        ]

    def forward(
        self,
        points: torch.Tensor,
        features: torch.Tensor,
        *,
        point_offsets: torch.Tensor | None = None,
    ) -> dict | list[dict]:
        """Forward pass using the learned granularity prompt.

        The granularity is passed as a tensor (not float) to preserve
        the gradient path through ``g_ft_logit``.
        """
        g = torch.sigmoid(self.g_ft_logit)  # always in (0, 1), smooth gradients
        return self.model(points, features, target_g=g, point_offsets=point_offsets)
