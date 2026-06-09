from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from src.core.logging import get_logger

logger = get_logger(__name__)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    _TORCH_AVAILABLE = True
except ImportError:
    torch = None  # type: ignore[assignment]
    nn = None  # type: ignore[assignment]
    F = None  # type: ignore[assignment]
    _TORCH_AVAILABLE = False

try:
    import cv2 as _cv2
    _CV2_AVAILABLE = True
except ImportError:
    _cv2 = None  # type: ignore[assignment]
    _CV2_AVAILABLE = False

INPUT_SIZE: int = 256  # face crop resize target


if _TORCH_AVAILABLE:

    class CentralDifferenceConv2d(nn.Module):
        """
        Central Difference Convolution (Yu et al., CVPR 2020).

        Combines gradient and intensity information to capture fine-grained texture
        differences between live faces and print/screen/GAN attacks.

        y = θ · conv(x) + (1−θ) · CDC(x)
          = conv(x) − (1−θ) · (Σ_k w_k) · x_center
        """

        def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int = 3,
            stride: int = 1,
            padding: int = 1,
            bias: bool = False,
            theta: float = 0.7,
        ) -> None:
            super().__init__()
            self.conv = nn.Conv2d(
                in_channels, out_channels, kernel_size,
                stride=stride, padding=padding, bias=bias,
            )
            self.theta = theta

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            out = self.conv(x)
            if abs(self.theta - 1.0) < 1e-6:
                return out
            # [out_ch, in_ch, 1, 1] — kernel summed over spatial dims
            kernel_diff = self.conv.weight.sum(dim=[2, 3], keepdim=True)
            out_diff = F.conv2d(
                x, kernel_diff,
                bias=None,
                stride=self.conv.stride,
                padding=0,
                groups=self.conv.groups,
            )
            return out - (1.0 - self.theta) * out_diff

    class _CDCBlock(nn.Sequential):
        def __init__(self, in_ch: int, out_ch: int, stride: int = 1, theta: float = 0.7) -> None:
            super().__init__(
                CentralDifferenceConv2d(in_ch, out_ch, 3, stride=stride, padding=1, theta=theta),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            )

    class CDCNPlusPlus(nn.Module):
        """
        CDCN++ binary liveness classifier.

        Input:  (B, 3, INPUT_SIZE, INPUT_SIZE) float32 in [0, 1], RGB channel order.
        Output: (B,) live probabilities in [0, 1] via sigmoid.

        Backbone channel progression: 3→32→64→128→256→512→GAP→128→1.
        """

        def __init__(self, theta: float = 0.7) -> None:
            super().__init__()
            self.backbone = nn.Sequential(
                _CDCBlock(3,   32,  stride=1, theta=theta),
                _CDCBlock(32,  64,  stride=2, theta=theta),
                _CDCBlock(64,  128, stride=2, theta=theta),
                _CDCBlock(128, 256, stride=2, theta=theta),
                _CDCBlock(256, 512, stride=2, theta=theta),
            )
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.classifier = nn.Sequential(
                nn.Flatten(),
                nn.Linear(512, 128),
                nn.ReLU(inplace=True),
                nn.Dropout(0.3),
                nn.Linear(128, 1),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return torch.sigmoid(self.classifier(self.pool(self.backbone(x)))).squeeze(1)

        def predict(self, x: torch.Tensor) -> torch.Tensor:
            """Alias for forward — used during evaluation."""
            return self(x)

else:  # pragma: no cover — torch not installed

    class CentralDifferenceConv2d:  # type: ignore[no-redef]
        pass

    class CDCNPlusPlus:  # type: ignore[no-redef]
        pass


def preprocess_face(face_crop: np.ndarray, input_size: int = INPUT_SIZE) -> Any:
    """
    Convert a BGR uint8 face crop to a normalised (1, 3, H, W) float32 tensor.

    Returns None if torch or cv2 is unavailable.
    """
    if not _TORCH_AVAILABLE or not _CV2_AVAILABLE:
        return None
    resized = _cv2.resize(face_crop, (input_size, input_size))
    rgb = _cv2.cvtColor(resized, _cv2.COLOR_BGR2RGB)
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    return tensor.unsqueeze(0)


def load_cdcn_model(
    weights_path: Path,
    device: str = "cpu",
    theta: float = 0.7,
) -> CDCNPlusPlus:
    """
    Load a CDCNPlusPlus model from a .pt checkpoint.

    Checkpoint must be a dict with key "model_state_dict", or a bare state_dict.
    Raises ModelLoadError on missing file or incompatible architecture.
    """
    from src.core.exceptions import ModelLoadError

    if not _TORCH_AVAILABLE:
        raise ModelLoadError("torch is not installed; cannot load CDCN++ model")

    if not weights_path.exists():
        raise ModelLoadError(f"CDCN++ weights not found: {weights_path}")

    model = CDCNPlusPlus(theta=theta)
    checkpoint = torch.load(weights_path, map_location=device, weights_only=True)
    state = checkpoint.get("model_state_dict", checkpoint)
    try:
        model.load_state_dict(state)
    except RuntimeError as exc:
        raise ModelLoadError(f"CDCN++ checkpoint incompatible with current architecture: {exc}") from exc

    model.to(device)
    model.eval()
    logger.info("cdcn_loaded", path=str(weights_path), device=device)
    return model
