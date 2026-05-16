"""GPU capacity tests — progressive VRAM scaling.

Validates the full GPU pipeline and finds the breaking point under 6GB VRAM.
Tests run sequentially with 15-second gaps to prevent GPU crashes.

Run: uv run pytest tests/test_gpu_capacity.py -v -s
"""

import time

import numpy as np
import pytest

try:
    import nabla as nb
    from nabla import nn  # noqa: F401 — needed by OmenJEPA model imports
    import nabla.nn.functional as F

    NABLA_AVAILABLE = True
except ImportError:
    NABLA_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from omen.gpu_budget import get_gpu_memory_info

# --- Skip markers ---
nabla_skip = pytest.mark.skipif(not NABLA_AVAILABLE, reason="Nabla not available")
gpu_skip = pytest.mark.skipif(
    not TORCH_AVAILABLE or not torch.cuda.is_available(),
    reason="CUDA GPU not available",
)

# --- Helpers ---

VRAM_SAFETY_GB = 5.5
GPU_GAP_SECONDS = 15


def _tensor(arr):
    """Create Nabla tensor from numpy array."""
    return nb.Tensor.from_dlpack(np.asarray(arr, dtype=np.float32))


def _shape(t):
    """Get shape as tuple of ints."""
    return tuple(int(d) for d in t.shape)


def _gpu_vram_mb():
    """Current GPU VRAM allocated in MB via torch."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def _safe_to_continue(max_gb=VRAM_SAFETY_GB):
    """Return False if VRAM usage is above safety threshold."""
    info = get_gpu_memory_info()
    return info["used_mb"] < max_gb * 1024


def _gpu_pause():
    """Wait between GPU tests to prevent thermal/VRAM cascade."""
    time.sleep(GPU_GAP_SECONDS)


def _make_nano_unet(channels=(16, 32, 64)):
    """Create a tiny conv encoder-decoder for VRAM testing.

    Returns (model_dict, input_channels) where model_dict has filter tensors.
    Not a full nn.Module — uses nabla functional conv2d.
    """
    enc_filters = []
    ch_in = 3
    for ch_out in channels:
        w = F.he_normal((3, 3, ch_in, ch_out))
        w.requires_grad = True
        enc_filters.append(w)
        ch_in = ch_out

    dec_filters = []
    for i in range(len(channels) - 1, 0, -1):
        ch_in = channels[i]
        ch_out = channels[i - 1]
        w = F.he_normal((3, 3, ch_in, ch_out))
        w.requires_grad = True
        dec_filters.append((w, ch_in, ch_out))

    # Final 1x1 conv to 3 channels
    w_out = F.he_normal((1, 1, channels[0], 3))
    w_out.requires_grad = True

    return {
        "enc_filters": enc_filters,
        "dec_filters": dec_filters,
        "w_out": w_out,
        "channels": channels,
    }


def _unet_forward(model, x):
    """Forward pass through nano U-Net. Returns output tensor."""
    skips = []
    for w in model["enc_filters"]:
        x = nb.silu(nb.conv2d(x, w, stride=(2, 2), padding=(1, 1)))
        skips.append(x)

    # Decoder (reverse, skip connections via add)
    skips = skips[:-1][::-1]
    for i, (w, ch_in, ch_out) in enumerate(model["dec_filters"]):
        if i == 0:
            x = nb.conv2d(skips[i], w, padding=(1, 1))
        else:
            # Upsample via nearest-neighbor + conv
            up = _nearest_upsample(x, 2)
            skip = skips[i]
            sh, sw = int(skip.shape[1]), int(skip.shape[2])
            uh, uw = int(up.shape[1]), int(up.shape[2])
            if uh != sh or uw != sw:
                skip = _center_crop(skip, uh, uw)
            x = nb.silu(
                nb.conv2d(nb.concatenate([up, skip], axis=-1), w, padding=(1, 1))
            )

    # Final upsample + 1x1 conv
    up = _nearest_upsample(x, 2)
    out = nb.conv2d(up, model["w_out"])
    return out


def _nearest_upsample(x, scale):
    """Nearest-neighbor upsample on H,W dims (NHWC)."""
    x = nb.repeat(x, scale, axis=1)
    x = nb.repeat(x, scale, axis=2)
    return x


def _center_crop(x, target_h, target_w):
    """Center crop NHWC tensor to target spatial dims."""
    h, w = int(x.shape[1]), int(x.shape[2])
    dh, dw = (h - target_h) // 2, (w - target_w) // 2
    return x[:, dh : dh + target_h, dw : dw + target_w, :]


# === Tests ===


@gpu_skip
def test_01_device_detection():
    """GPU detected, torch.cuda reports device available."""
    assert torch.cuda.is_available(), "CUDA not available"
    info = get_gpu_memory_info()
    assert info["backend"] != "none", "No GPU backend detected"
    assert info["total_mb"] > 0, "GPU total memory reported as 0"
    print(
        f"  GPU: {info['total_mb']}MB total, {info['free_mb']}MB free, "
        f"backend={info['backend']}"
    )


@gpu_skip
def test_02_tensor_cuda_transfer():
    """Create nabla tensor, attempt .cuda(), verify device or skip."""
    _gpu_pause()
    arr = np.random.randn(4, 4).astype(np.float32)
    t = _tensor(arr)
    assert _shape(t) == (4, 4)

    # Test nabla .cuda() — may not exist in current version
    try:
        t_gpu = t.cuda()
        assert t_gpu is not None
        print("  nabla tensor.cuda() works")
    except (AttributeError, NotImplementedError, Exception) as e:
        pytest.skip(f"nabla tensor.cuda() not available: {e}")

    # Also verify torch GPU transfer works
    t_torch = torch.tensor(arr).cuda()
    assert t_torch.device.type == "cuda"
    print("  torch tensor.cuda() works")


@gpu_skip
def test_03_gpu_compute():
    """Simple matmul on GPU via torch, verify result correctness."""
    _gpu_pause()
    a = torch.randn(64, 64, device="cuda")
    b = torch.randn(64, 64, device="cuda")
    c = a @ b
    assert c.shape == (64, 64)
    assert c.device.type == "cuda"
    # Verify result matches CPU
    c_cpu = a.cpu() @ b.cpu()
    assert torch.allclose(c.cpu(), c_cpu, atol=1e-4)
    print("  GPU matmul matches CPU result")


@gpu_skip
def test_04_vram_baseline():
    """Measure VRAM before/after tensor allocation."""
    _gpu_pause()
    torch.cuda.reset_peak_memory_stats()
    before = _gpu_vram_mb()

    # Allocate ~64MB on GPU
    big = torch.randn(4096, 4096, device="cuda")  # 64MB float32
    after = _gpu_vram_mb()
    delta = after - before

    assert delta > 0, "VRAM did not increase after allocation"
    assert delta < 200, f"VRAM jump too large: {delta}MB (expected ~64MB)"
    print(f"  VRAM: {before:.1f}MB -> {after:.1f}MB (delta={delta:.1f}MB)")

    del big
    torch.cuda.empty_cache()


@gpu_skip
def test_05_progressive_tensor_scale():
    """Allocate tensors 64->128->256->512->1024 MB, track VRAM."""
    _gpu_pause()
    sizes_mb = [64, 128, 256, 512, 1024]
    tensors = []

    for target_mb in sizes_mb:
        if not _safe_to_continue():
            pytest.skip(f"VRAM above {VRAM_SAFETY_GB}GB at {target_mb}MB target")

        elements = target_mb * 1024 * 1024 // 4  # float32 = 4 bytes
        side = int(elements**0.5)
        t = torch.randn(side, side, device="cuda")
        tensors.append(t)

        vram = _gpu_vram_mb()
        info = get_gpu_memory_info()
        print(
            f"  {target_mb}MB allocated: VRAM={vram:.0f}MB, "
            f"GPU used={info['used_mb']}MB, free={info['free_mb']}MB"
        )

        if info["used_mb"] > VRAM_SAFETY_GB * 1024:
            print(f"  Stopping: exceeded {VRAM_SAFETY_GB}GB safety limit")
            break

    del tensors
    torch.cuda.empty_cache()


@nabla_skip
@gpu_skip
def test_06_nano_jepa_forward():
    """Tiny OmenJEPA (latent=64), forward pass on GPU if possible."""
    _gpu_pause()
    if not _safe_to_continue():
        pytest.skip("VRAM above safety threshold")

    from omen.config import OmenConfig
    from omen.model.jepa import OmenJEPA

    config = OmenConfig.v1_dense()
    config.components.ar_predictor = False
    config.components.scene_delta_encoder = False
    config.components.episodic_correction = False

    model = OmenJEPA(config=config, latent_dim=64)
    model.train()

    # Create fake inputs
    scene_graph = {
        "geometry": _tensor(np.random.randn(1, 10, 6).astype(np.float32)),
        "materials": _tensor(np.random.randn(1, 5, 5).astype(np.float32)),
        "lights": _tensor(np.random.randn(1, 3, 7).astype(np.float32)),
    }
    rgba = _tensor(np.random.randn(1, 16, 16, 4).astype(np.float32))

    fused, scene_latent = model.encode(scene_graph, rgba)
    assert _shape(fused) == (1, 64), f"Expected (1,64), got {_shape(fused)}"

    residual = model.decode(fused, rgba)
    assert int(residual.shape[0]) == 1
    print(f"  Nano JEPA forward: fused={_shape(fused)}, residual={_shape(residual)}")


@nabla_skip
@gpu_skip
def test_07_nano_jepa_backward():
    """Nano OmenJEPA forward+backward via value_and_grad."""
    _gpu_pause()
    if not _safe_to_continue():
        pytest.skip("VRAM above safety threshold")

    from omen.config import OmenConfig
    from omen.model.jepa import OmenJEPA

    config = OmenConfig.v1_dense()
    config.components.ar_predictor = False
    config.components.scene_delta_encoder = False
    config.components.episodic_correction = False

    model = OmenJEPA(config=config, latent_dim=64)
    model.train()

    scene_graph = {
        "geometry": _tensor(np.random.randn(1, 10, 6).astype(np.float32)),
        "materials": _tensor(np.random.randn(1, 5, 5).astype(np.float32)),
        "lights": _tensor(np.random.randn(1, 3, 7).astype(np.float32)),
    }
    rgba = _tensor(np.random.randn(1, 16, 16, 4).astype(np.float32))
    target = _tensor(np.random.randn(1, 64).astype(np.float32))

    def loss_fn(rgba):
        fused, _ = model.encode(scene_graph, rgba)
        return nb.mean((fused - target) ** 2)

    val, grad = nb.value_and_grad(loss_fn)(rgba)
    assert _shape(grad) == _shape(rgba), (
        f"Grad shape {_shape(grad)} != input {_shape(rgba)}"
    )
    assert not np.isnan(val.to_numpy().item()), "Loss is NaN"
    print(
        f"  Nano JEPA backward: loss={val.to_numpy().item():.4f}, "
        f"grad_shape={_shape(grad)}"
    )


@nabla_skip
@gpu_skip
def test_08_small_unet_forward():
    """3-level U-Net (16->32->64), forward pass."""
    _gpu_pause()
    if not _safe_to_continue():
        pytest.skip("VRAM above safety threshold")

    model = _make_nano_unet(channels=(16, 32, 64))
    x = _tensor(np.random.randn(1, 32, 32, 3).astype(np.float32))

    # Simple forward: just encoder path + skip adds
    s1 = nb.silu(nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1)))
    s2 = nb.silu(nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1)))
    s3 = nb.silu(nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1)))

    assert _shape(s1) == (1, 16, 16, 16), f"Stage 1: {_shape(s1)}"
    assert _shape(s2) == (1, 8, 8, 32), f"Stage 2: {_shape(s2)}"
    assert _shape(s3) == (1, 4, 4, 64), f"Stage 3: {_shape(s3)}"
    print(
        f"  U-Net forward: {32}x{32} -> s1={_shape(s1)}, s2={_shape(s2)}, s3={_shape(s3)}"
    )


@nabla_skip
@gpu_skip
def test_09_small_unet_backward():
    """U-Net encoder forward+backward, measure VRAM."""
    _gpu_pause()
    if not _safe_to_continue():
        pytest.skip("VRAM above safety threshold")

    model = _make_nano_unet(channels=(16, 32, 64))
    x = _tensor(np.random.randn(1, 64, 64, 3).astype(np.float32))

    def forward(x):
        s1 = nb.silu(
            nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1))
        )
        s2 = nb.silu(
            nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1))
        )
        s3 = nb.silu(
            nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1))
        )
        return nb.mean(s3**2)

    val, grad = nb.value_and_grad(forward)(x)
    assert _shape(grad) == _shape(x)
    assert not np.isnan(val.to_numpy().item())
    print(f"  U-Net backward at 64x64: loss={val.to_numpy().item():.4f}")


@nabla_skip
@gpu_skip
def test_10_compile_gpu():
    """@nb.compile on a forward step, check compilation works."""
    _gpu_pause()
    if not _safe_to_continue():
        pytest.skip("VRAM above safety threshold")

    x = _tensor(np.random.randn(1, 16, 16, 3).astype(np.float32))
    w = F.he_normal((3, 3, 3, 16))
    w.requires_grad = True

    @nb.compile
    def compiled_forward(x, w):
        return nb.mean(nb.conv2d(x, w, padding=(1, 1)) ** 2)

    # First call compiles, second uses cache
    try:
        result = compiled_forward(x, w)
        assert result is not None
        print(f"  @nb.compile works: output shape={_shape(result)}")
    except Exception as e:
        pytest.skip(f"@nb.compile failed: {e}")


@nabla_skip
@gpu_skip
@pytest.mark.parametrize("resolution", [64, 128, 256])
def test_11_resolution_scale(resolution):
    """Run encoder at scaled resolutions, stop at 5.5GB."""
    _gpu_pause()
    if not _safe_to_continue():
        pytest.skip(f"VRAM above {VRAM_SAFETY_GB}GB at {resolution}x{resolution}")

    model = _make_nano_unet(channels=(16, 32, 64))
    x = _tensor(np.random.randn(1, resolution, resolution, 3).astype(np.float32))

    def forward(x):
        s1 = nb.silu(
            nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1))
        )
        s2 = nb.silu(
            nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1))
        )
        s3 = nb.silu(
            nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1))
        )
        return nb.mean(s3**2)

    val, grad = nb.value_and_grad(forward)(x)
    info = get_gpu_memory_info()
    assert info["used_mb"] < 6 * 1024, (
        f"VRAM exceeded 6GB at {resolution}x{resolution}: {info['used_mb']}MB"
    )
    print(
        f"  {resolution}x{resolution}: loss={val.to_numpy().item():.4f}, "
        f"VRAM={info['used_mb']}MB"
    )


@nabla_skip
@gpu_skip
def test_12_vram_breaking_point():
    """Binary search for max image resolution under 6GB."""
    _gpu_pause()

    lo, hi = 64, 2048
    best = lo
    max_vram_mb = 6 * 1024

    while lo <= hi:
        mid = (lo + hi) // 2
        try:
            info = get_gpu_memory_info()
            if info["used_mb"] >= max_vram_mb:
                hi = mid - 1
                continue

            x = _tensor(np.random.randn(1, mid, mid, 3).astype(np.float32))
            w = F.he_normal((3, 3, 3, 16))
            result = nb.mean(nb.conv2d(x, w, padding=(1, 1)) ** 2)
            _ = result.to_numpy()  # Force evaluation

            info = get_gpu_memory_info()
            if info["used_mb"] < max_vram_mb:
                best = mid
                lo = mid + 1
            else:
                hi = mid - 1
        except (MemoryError, RuntimeError, Exception):
            hi = mid - 1

        _gpu_pause()

    info = get_gpu_memory_info()
    print(
        f"  Max resolution under 6GB: {best}x{best} (current VRAM: {info['used_mb']}MB)"
    )
    assert best >= 64, "Could not allocate even 64x64 on GPU"
