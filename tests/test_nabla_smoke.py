"""Nabla API smoke tests for U-Net decoder (tasks 7.10-7.17)."""

import numpy as np

try:
    import nabla as nb
    import nabla.nn.functional as F
    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

import pytest

pytestmark = pytest.mark.skipif(not NABLA_AVAILABLE, reason="Nabla not available")


def _tensor(arr):
    """Create Nabla tensor from numpy array."""
    return nb.Tensor.from_dlpack(np.asarray(arr, dtype=np.float32))


def _shape(t):
    """Get shape as tuple of ints (Nabla returns Dim objects)."""
    return tuple(int(d) for d in t.shape)


def test_7_10_pixel_shuffle_autograd():
    """7.10: Pixel Shuffle via reshape+transpose with autograd."""
    x = _tensor(np.random.randn(1, 4, 4, 12))
    B, H, W, C = _shape(x)
    r = 2

    def pixel_shuffle(x):
        y = nb.reshape(x, (B, H, r, W, r, C // (r * r)))
        y = nb.permute(y, (0, 2, 1, 4, 3, 5))
        return nb.reshape(y, (B, H * r, W * r, C // (r * r)))

    out = pixel_shuffle(x)
    assert _shape(out) == (1, 8, 8, 3), f"Expected (1,8,8,3), got {_shape(out)}"

    def loss_fn(x):
        return nb.mean(pixel_shuffle(x) ** 2)

    val, grad = nb.value_and_grad(loss_fn)(x)
    assert _shape(grad) == _shape(x)
    assert val.to_numpy().item() > 0


def test_7_11_skip_concat():
    """7.11: Skip connection concatenation with autograd."""
    a = _tensor(np.random.randn(1, 8, 8, 64))
    b = _tensor(np.random.randn(1, 8, 8, 64))
    c = nb.concatenate([a, b], axis=-1)
    assert _shape(c) == (1, 8, 8, 128)

    def loss_fn(a, b):
        return nb.mean(nb.concatenate([a, b], axis=-1) ** 2)

    val, grad = nb.value_and_grad(loss_fn, argnums=0)(a, b)
    assert _shape(grad) == _shape(a)


def test_7_12_conv2d_stride2():
    """7.12: Conv2d stride=2 downsampling with autograd."""
    x = _tensor(np.random.randn(1, 16, 16, 32))
    w = _tensor(np.random.randn(3, 3, 32, 64) * 0.01)
    y = nb.conv2d(x, w, stride=(2, 2), padding=(1, 1))
    assert _shape(y) == (1, 8, 8, 64), f"Expected (1,8,8,64), got {_shape(y)}"

    def loss_fn(x):
        return nb.mean(nb.conv2d(x, w, stride=(2, 2), padding=(1, 1)) ** 2)

    val, grad = nb.value_and_grad(loss_fn)(x)
    assert _shape(grad) == _shape(x)


def test_7_14_value_and_grad_multiarg():
    """7.14: value_and_grad with multi-argument function."""
    a = _tensor(np.random.randn(1, 8, 8, 32))
    b = _tensor(np.random.randn(1, 8, 8, 32))

    def loss_fn(a, b):
        return nb.mean(a ** 2 + b ** 2)

    val, ga = nb.value_and_grad(loss_fn, argnums=0)(a, b)
    assert val.to_numpy().item() > 0
    assert _shape(ga) == _shape(a)


def test_7_16_optimizer_init():
    """7.16: Optimizer creation with nn.Module."""
    from nabla import nn

    class TinyModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.linear = nn.Linear(32, 64)

        def forward(self, x):
            return self.linear(x)

    model = TinyModel()
    model.train()
    opt = nb.nn.optim.AdamW(model, lr=1e-3)
    assert opt is not None


def test_7_17_dlpack_from_numpy():
    """7.17: nb.Tensor.from_dlpack zero-copy from numpy."""
    arr = np.random.randn(480, 640, 3).astype(np.float32)
    t = nb.Tensor.from_dlpack(arr)
    assert _shape(t) == (480, 640, 3)
    assert nb.mean(t).to_numpy().item() != 0.0


def test_conv2d_pixel_shuffle_decoder_path():
    """Integration: conv -> pixel shuffle upsample (decoder path)."""
    x = _tensor(np.random.randn(1, 4, 4, 64))
    w = _tensor(np.random.randn(3, 3, 64, 48) * 0.01)
    y = nb.conv2d(x, w, padding=(1, 1))
    B, H, W, C = _shape(y)
    r = 2
    out = nb.reshape(y, (B, H, r, W, r, C // (r * r)))
    out = nb.permute(out, (0, 2, 1, 4, 3, 5))
    out = nb.reshape(out, (B, H * r, W * r, C // (r * r)))
    assert _shape(out) == (1, 8, 8, 12), f"Expected (1,8,8,12), got {_shape(out)}"
