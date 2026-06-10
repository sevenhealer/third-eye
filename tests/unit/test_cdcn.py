from __future__ import annotations

import numpy as np
import pytest

try:
    import torch
    import torch.nn as nn
    _TORCH = True
except ImportError:
    _TORCH = False

pytestmark = pytest.mark.skipif(not _TORCH, reason="torch not installed")


from src.antispoofing.cdcn import (
    CDCNPlusPlus,
    CentralDifferenceConv2d,
    INPUT_SIZE,
    preprocess_face,
)


# ── CentralDifferenceConv2d ───────────────────────────────────────────────────

class TestCentralDifferenceConv2d:
    def test_output_shape_matches_standard_conv(self):
        layer = CentralDifferenceConv2d(3, 32, kernel_size=3, stride=1, padding=1)
        x = torch.randn(2, 3, 64, 64)
        out = layer(x)
        assert out.shape == (2, 32, 64, 64)

    def test_stride_2_halves_spatial_dims(self):
        layer = CentralDifferenceConv2d(32, 64, kernel_size=3, stride=2, padding=1)
        x = torch.randn(1, 32, 64, 64)
        out = layer(x)
        assert out.shape == (1, 64, 32, 32)

    def test_theta_1_equals_standard_conv(self):
        """θ=1.0 should reduce to a plain convolution."""
        layer = CentralDifferenceConv2d(3, 8, kernel_size=3, padding=1, theta=1.0)
        ref = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        ref.weight = layer.conv.weight  # share weights

        x = torch.randn(1, 3, 32, 32)
        assert torch.allclose(layer(x), ref(x), atol=1e-5)

    def test_theta_0_produces_different_output_from_standard_conv(self):
        """θ=0 (pure CDC) should differ from standard conv."""
        layer = CentralDifferenceConv2d(3, 8, kernel_size=3, padding=1, theta=0.0)
        ref = nn.Conv2d(3, 8, 3, padding=1, bias=False)
        ref.weight = layer.conv.weight

        x = torch.randn(1, 3, 32, 32)
        # Pure CDC output is standard_conv - 1.0 * kernel_sum * x_center
        assert not torch.allclose(layer(x), ref(x), atol=1e-4)

    def test_output_is_differentiable(self):
        """Gradients must flow back through the CDC layer."""
        layer = CentralDifferenceConv2d(3, 8, kernel_size=3, padding=1, theta=0.7)
        x = torch.randn(1, 3, 32, 32, requires_grad=True)
        out = layer(x).sum()
        out.backward()
        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


# ── CDCNPlusPlus model ────────────────────────────────────────────────────────

class TestCDCNPlusPlus:
    def _model(self) -> CDCNPlusPlus:
        return CDCNPlusPlus(theta=0.7)

    def test_forward_shape(self):
        model = self._model()
        x = torch.randn(2, 3, INPUT_SIZE, INPUT_SIZE)
        out = model(x)
        assert out.shape == (2,)

    def test_output_in_unit_interval(self):
        """Sigmoid output must be in [0, 1]."""
        model = self._model()
        x = torch.randn(4, 3, INPUT_SIZE, INPUT_SIZE)
        out = model(x)
        assert out.min().item() >= 0.0
        assert out.max().item() <= 1.0

    def test_single_sample_inference(self):
        model = self._model()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        out = model(x)
        assert out.shape == (1,)
        assert 0.0 <= out.item() <= 1.0

    def test_predict_alias(self):
        # eval mode: dropout makes train-mode forward passes stochastic
        model = self._model().eval()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        assert torch.equal(model(x), model.predict(x))

    def test_eval_mode_no_grad(self):
        model = self._model().eval()
        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (1,)

    def test_state_dict_roundtrip(self, tmp_path):
        """Save and reload state_dict; model output must be identical."""
        model = CDCNPlusPlus(theta=0.7)
        ckpt = tmp_path / "cdcn.pt"
        torch.save({"model_state_dict": model.state_dict(), "epoch": 1}, ckpt)

        loaded = CDCNPlusPlus(theta=0.7)
        checkpoint = torch.load(ckpt, map_location="cpu", weights_only=True)
        loaded.load_state_dict(checkpoint["model_state_dict"])
        loaded.eval()
        model.eval()

        x = torch.randn(1, 3, INPUT_SIZE, INPUT_SIZE)
        assert torch.allclose(model(x), loaded(x), atol=1e-6)


# ── preprocess_face ───────────────────────────────────────────────────────────

class TestPreprocessFace:
    def test_output_shape(self):
        crop = np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8)
        tensor = preprocess_face(crop)
        if tensor is None:
            pytest.skip("cv2 not available")
        assert tensor.shape == (1, 3, INPUT_SIZE, INPUT_SIZE)

    def test_values_in_unit_interval(self):
        crop = np.random.randint(0, 255, (64, 96, 3), dtype=np.uint8)
        tensor = preprocess_face(crop)
        if tensor is None:
            pytest.skip("cv2 not available")
        assert tensor.min().item() >= 0.0
        assert tensor.max().item() <= 1.0

    def test_float32_dtype(self):
        crop = np.zeros((100, 100, 3), dtype=np.uint8)
        tensor = preprocess_face(crop)
        if tensor is None:
            pytest.skip("cv2 not available")
        assert tensor.dtype == torch.float32


# ── load_cdcn_model ───────────────────────────────────────────────────────────

class TestLoadCDCNModel:
    def test_raises_on_missing_file(self, tmp_path):
        from src.antispoofing.cdcn import load_cdcn_model
        from src.core.exceptions import ModelLoadError
        with pytest.raises(ModelLoadError, match="not found"):
            load_cdcn_model(tmp_path / "nonexistent.pt")

    def test_loads_valid_checkpoint(self, tmp_path):
        from src.antispoofing.cdcn import load_cdcn_model
        model = CDCNPlusPlus(theta=0.7)
        ckpt = tmp_path / "cdcn.pt"
        torch.save({"model_state_dict": model.state_dict()}, ckpt)
        loaded = load_cdcn_model(ckpt, device="cpu")
        assert isinstance(loaded, CDCNPlusPlus)

    def test_model_in_eval_mode_after_load(self, tmp_path):
        from src.antispoofing.cdcn import load_cdcn_model
        model = CDCNPlusPlus()
        ckpt = tmp_path / "cdcn.pt"
        torch.save(model.state_dict(), ckpt)
        loaded = load_cdcn_model(ckpt)
        assert not loaded.training

    def test_raises_on_incompatible_checkpoint(self, tmp_path):
        from src.antispoofing.cdcn import load_cdcn_model
        from src.core.exceptions import ModelLoadError
        # Save a state dict with wrong key shapes
        bad_state = {"backbone.0.0.conv.weight": torch.randn(999, 3, 3, 3)}
        ckpt = tmp_path / "bad.pt"
        torch.save(bad_state, ckpt)
        with pytest.raises(ModelLoadError, match="incompatible"):
            load_cdcn_model(ckpt)
