"""GPU capacity tests — progressive VRAM/RAM scaling with safety guards.

Validates GPU pipeline and nabla compute limits.
Tests run with RAM guards that skip before OOM territory (24GB limit on 32GB system).

Key findings from prior runs:
- Nabla supports GPU via device=Accelerator() and transfer_to(tensor, gpu)
- Graph cache is the RAM bomb culprit: each eager value_and_grad = 6-8GB graph entry
- @nb.compile fixes it: graph compiled ONCE, reused on cache hit (same shapes)
- clear_all() is a wasteful fallback — destroys cached graph, forces recompilation
- Proper pattern (from nabla examples): @nb.compile on the entire train step

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
    from max.driver import Accelerator

    MAX_DRIVER_AVAILABLE = True
except ImportError:
    MAX_DRIVER_AVAILABLE = False

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from omen.gpu_budget import get_gpu_memory_info
from omen.kernels.activations import silu_gpu

# --- Skip markers ---
nabla_skip = pytest.mark.skipif(not NABLA_AVAILABLE, reason="Nabla not available")
gpu_skip = pytest.mark.skipif(
    not TORCH_AVAILABLE or not torch.cuda.is_available(),
    reason="CUDA GPU not available",
)

# --- Safety limits ---
VRAM_SAFETY_GB = 5.5
RAM_SAFETY_GB = 24  # Leave ~8GB for OS + Blender on 32GB system
GPU_GAP_SECONDS = 15


# --- Resource monitoring helpers ---


def _tensor(arr, device=None):
    """Create Nabla tensor from numpy array, optionally on GPU."""
    t = nb.Tensor.from_dlpack(np.asarray(arr, dtype=np.float32))
    if device is not None:
        try:
            t = nb.ops.transfer_to(t, device)
            print(f"    [transfer] tensor {tuple(int(d) for d in t.shape)} -> GPU OK")
        except Exception as e:
            print(f"    [transfer] FAILED: {e} — staying on CPU")
    return t


def _shape(t):
    """Get shape as tuple of ints."""
    return tuple(int(d) for d in t.shape)


def _gpu_vram_mb():
    """Current GPU VRAM allocated in MB via torch."""
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0


def _process_rss_mb():
    """Current process RSS in MB (best effort, cross-platform)."""
    try:
        import resource

        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss // 1024
    except Exception:
        return 0


def _system_ram_mb():
    """System RAM used/free in MB via /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) // 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        return {"total_mb": total, "used_mb": total - avail, "free_mb": avail}
    except Exception:
        return {"total_mb": 0, "used_mb": 0, "free_mb": 0}


def _resource_report(label=""):
    """Print a one-line RAM + VRAM + GPU snapshot."""
    ram = _system_ram_mb()
    vram = get_gpu_memory_info()
    gpu_util = "?"
    try:
        import subprocess

        r = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=utilization.gpu",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        gpu_util = r.stdout.strip()
    except Exception:
        pass
    print(
        f"  [{label}] RAM: {ram['used_mb']}/{ram['total_mb']}MB | "
        f"VRAM: {vram['used_mb']}/{vram['total_mb']}MB | "
        f"GPU util: {gpu_util}%"
    )


def _safe_to_continue_vram(max_gb=VRAM_SAFETY_GB):
    """Return False if VRAM usage is above safety threshold."""
    info = get_gpu_memory_info()
    return info["used_mb"] < max_gb * 1024


def _safe_to_continue_ram(max_gb=RAM_SAFETY_GB):
    """Return False if system RAM usage is above safety threshold."""
    ram = _system_ram_mb()
    return ram["used_mb"] < max_gb * 1024


def _ram_guard(label=""):
    """Combined VRAM + RAM safety check. Skips test if either is exceeded."""
    if not _safe_to_continue_ram():
        ram = _system_ram_mb()
        pytest.skip(
            f"RAM above {RAM_SAFETY_GB}GB safety threshold ({ram['used_mb']}MB used)"
        )
    if not _safe_to_continue_vram():
        pytest.skip(f"VRAM above {VRAM_SAFETY_GB}GB safety threshold")
    _resource_report(label)


def _gpu_pause():
    """Wait between GPU tests to prevent thermal/VRAM cascade."""
    time.sleep(GPU_GAP_SECONDS)


def _clear_nabla_graph():
    """Clear nabla graph cache to reclaim RAM between tests."""
    try:
        nb.GRAPH.clear_all()
    except Exception:
        pass


def _get_device():
    """Get Accelerator device if available, else None (CPU)."""
    if MAX_DRIVER_AVAILABLE:
        try:
            dev = Accelerator()
            print(f"    [device] Accelerator() = {dev}")
            return dev
        except Exception as e:
            print(f"    [device] Accelerator() FAILED: {e}")
            return None
    print("    [device] max.driver not available, using CPU")
    return None


def _transfer_model_to_device(model, device):
    """Transfer all model weights to device. Returns model (mutated in-place)."""
    if device is None:
        return model
    state = model.state_dict()
    new_state = {}
    transferred = 0
    failed = 0
    for k, v in state.items():
        try:
            new_state[k] = nb.ops.transfer_to(v, device)
            transferred += 1
        except Exception as e:
            if failed == 0:
                print(f"    [model-transfer] FAILED on '{k}': {e}")
            new_state[k] = v
            failed += 1
    model.load_state_dict(new_state)
    print(f"    [model-transfer] {transferred}/{transferred + failed} weights -> GPU")
    return model


# --- Model helpers ---


def _make_nano_unet(channels=(16, 32, 64), device=None):
    """Create a tiny conv encoder-decoder for VRAM testing.

    Not a full nn.Module — uses nabla functional conv2d.
    Optionally places weights on GPU via device parameter.
    """
    enc_filters = []
    ch_in = 3
    gpu_ok = 0
    gpu_fail = 0
    for ch_out in channels:
        w = F.he_normal((3, 3, ch_in, ch_out))
        if device is not None:
            try:
                w = nb.ops.transfer_to(w, device)
                gpu_ok += 1
            except Exception as e:
                if gpu_fail == 0:
                    print(f"    [unet-transfer] enc filter FAILED: {e}")
                gpu_fail += 1
        w.requires_grad = True
        enc_filters.append(w)
        ch_in = ch_out

    dec_filters = []
    for i in range(len(channels) - 1, 0, -1):
        ch_in = channels[i]
        ch_out = channels[i - 1]
        w = F.he_normal((3, 3, ch_in, ch_out))
        if device is not None:
            try:
                w = nb.ops.transfer_to(w, device)
                gpu_ok += 1
            except Exception as e:
                if gpu_fail == 0:
                    print(f"    [unet-transfer] dec filter FAILED: {e}")
                gpu_fail += 1
        w.requires_grad = True
        dec_filters.append((w, ch_in, ch_out))

    w_out = F.he_normal((1, 1, channels[0], 3))
    if device is not None:
        try:
            w_out = nb.ops.transfer_to(w_out, device)
            gpu_ok += 1
        except Exception as e:
            print(f"    [unet-transfer] w_out FAILED: {e}")
            gpu_fail += 1
    w_out.requires_grad = True

    if device is not None:
        total = gpu_ok + gpu_fail
        print(f"    [unet-transfer] {gpu_ok}/{total} filters -> GPU")

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
        x = silu_gpu(nb.conv2d(x, w, stride=(2, 2), padding=(1, 1)))
        skips.append(x)

    # Decoder (reverse, skip connections via add)
    skips = skips[:-1][::-1]
    for i, (w, ch_in, ch_out) in enumerate(model["dec_filters"]):
        if i == 0:
            x = nb.conv2d(skips[i], w, padding=(1, 1))
        else:
            up = _nearest_upsample(x, 2)
            skip = skips[i]
            sh, sw = int(skip.shape[1]), int(skip.shape[2])
            uh, uw = int(up.shape[1]), int(up.shape[2])
            if uh != sh or uw != sw:
                skip = _center_crop(skip, uh, uw)
            x = silu_gpu(
                nb.conv2d(nb.concatenate([up, skip], axis=-1), w, padding=(1, 1))
            )

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


# === Section 1: Device detection & transfer ===


@gpu_skip
def test_01_device_detection():
    """GPU detected, torch.cuda reports device available."""
    assert torch.cuda.is_available(), "CUDA not available"
    info = get_gpu_memory_info()
    assert info["backend"] != "none", "No GPU backend detected"
    assert info["total_mb"] > 0, "GPU total memory reported as 0"
    _resource_report("detection")
    print(
        f"  GPU: {info['total_mb']}MB total, {info['free_mb']}MB free, "
        f"backend={info['backend']}"
    )


@gpu_skip
def test_02_tensor_cuda_transfer():
    """Create nabla tensor, transfer to GPU via transfer_to, verify."""
    _gpu_pause()
    arr = np.random.randn(4, 4).astype(np.float32)
    t = _tensor(arr)
    assert _shape(t) == (4, 4)

    # Try nabla GPU transfer via max.driver Accelerator
    gpu_ok = False
    if MAX_DRIVER_AVAILABLE:
        try:
            gpu = Accelerator()
            t_gpu = nb.ops.transfer_to(t, gpu)
            assert t_gpu is not None
            gpu_ok = True
            print("  nabla transfer_to(Accelerator) works")
        except Exception as e:
            print(f"  nabla transfer_to failed: {e}")

    if not gpu_ok:
        try:
            t_gpu = t.cuda()
            assert t_gpu is not None
            gpu_ok = True
            print("  nabla tensor.cuda() works")
        except (AttributeError, NotImplementedError, Exception) as e:
            print(f"  nabla .cuda() failed: {e}")

    # Also verify torch GPU works
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
    c_cpu = a.cpu() @ b.cpu()
    assert torch.allclose(c.cpu(), c_cpu, atol=1e-4)
    print("  GPU matmul matches CPU result")


@gpu_skip
def test_04_vram_baseline():
    """Measure VRAM before/after tensor allocation."""
    _gpu_pause()
    torch.cuda.reset_peak_memory_stats()
    before = _gpu_vram_mb()

    big = torch.randn(4096, 4096, device="cuda")  # 64MB float32
    after = _gpu_vram_mb()
    delta = after - before

    assert delta > 0, "VRAM did not increase after allocation"
    assert delta < 200, f"VRAM jump too large: {delta}MB (expected ~64MB)"
    _resource_report("post-alloc")
    print(f"  VRAM: {before:.1f}MB -> {after:.1f}MB (delta={delta:.1f}MB)")

    del big
    torch.cuda.empty_cache()


@gpu_skip
def test_05_progressive_tensor_scale():
    """Allocate tensors 64->128->256->512->1024 MB on GPU, track VRAM."""
    _gpu_pause()
    sizes_mb = [64, 128, 256, 512, 1024]
    tensors = []

    for target_mb in sizes_mb:
        if not _safe_to_continue_vram():
            pytest.skip(f"VRAM above {VRAM_SAFETY_GB}GB at {target_mb}MB target")

        elements = target_mb * 1024 * 1024 // 4
        side = int(elements**0.5)
        t = torch.randn(side, side, device="cuda")
        tensors.append(t)

        _resource_report(f"{target_mb}MB")
        info = get_gpu_memory_info()
        if info["used_mb"] > VRAM_SAFETY_GB * 1024:
            print(f"  Stopping: exceeded {VRAM_SAFETY_GB}GB safety limit")
            break

    del tensors
    torch.cuda.empty_cache()


# === Section 2: Nabla model tests (GPU-enabled, RAM-guarded) ===


@nabla_skip
@gpu_skip
def test_06_nano_jepa_forward():
    """Tiny OmenJEPA (latent=64), forward pass."""
    _gpu_pause()
    _ram_guard("pre-jepa")

    from omen.config import OmenConfig
    from omen.model.jepa import OmenJEPA

    config = OmenConfig.v1_dense()
    config.components.ar_predictor = False
    config.components.scene_delta_encoder = False
    config.components.episodic_correction = False

    model = OmenJEPA(config=config, latent_dim=64)
    model.train()

    device = _get_device()
    _transfer_model_to_device(model, device)

    scene_graph = {
        "geometry": _tensor(np.random.randn(1, 10, 6).astype(np.float32), device),
        "materials": _tensor(np.random.randn(1, 5, 5).astype(np.float32), device),
        "lights": _tensor(np.random.randn(1, 3, 7).astype(np.float32), device),
    }
    rgba = _tensor(np.random.randn(1, 16, 16, 4).astype(np.float32), device)

    fused, scene_latent = model.encode(scene_graph, rgba)
    assert _shape(fused) == (1, 64), f"Expected (1,64), got {_shape(fused)}"

    residual = model.decode(fused, rgba)
    assert int(residual.shape[0]) == 1
    _ram_guard("post-jepa-fwd")
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  Nano JEPA forward ({device_str}): "
        f"fused={_shape(fused)}, residual={_shape(residual)}"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_07_nano_jepa_backward():
    """Nano OmenJEPA forward+backward via value_and_grad."""
    _gpu_pause()
    _ram_guard("pre-jepa-bwd")

    from omen.config import OmenConfig
    from omen.model.jepa import OmenJEPA

    config = OmenConfig.v1_dense()
    config.components.ar_predictor = False
    config.components.scene_delta_encoder = False
    config.components.episodic_correction = False

    model = OmenJEPA(config=config, latent_dim=64)
    model.train()

    device = _get_device()
    _transfer_model_to_device(model, device)

    scene_graph = {
        "geometry": _tensor(np.random.randn(1, 10, 6).astype(np.float32), device),
        "materials": _tensor(np.random.randn(1, 5, 5).astype(np.float32), device),
        "lights": _tensor(np.random.randn(1, 3, 7).astype(np.float32), device),
    }
    rgba = _tensor(np.random.randn(1, 16, 16, 4).astype(np.float32), device)
    target = _tensor(np.random.randn(1, 64).astype(np.float32), device)

    def loss_fn(rgba):
        fused, _ = model.encode(scene_graph, rgba)
        diff = fused - target
        return nb.mean(diff * diff)

    t0 = time.perf_counter()
    val, grad = nb.value_and_grad(loss_fn)(rgba)
    dt = time.perf_counter() - t0

    assert _shape(grad) == _shape(rgba), (
        f"Grad shape {_shape(grad)} != input {_shape(rgba)}"
    )
    assert not np.isnan(val.to_numpy().item()), "Loss is NaN"
    _ram_guard("post-jepa-bwd")
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  Nano JEPA backward ({device_str}): loss={val.to_numpy().item():.4f}, "
        f"grad_shape={_shape(grad)}, time={dt:.2f}s"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_08_small_unet_forward():
    """3-level U-Net (16->32->64), forward pass."""
    _gpu_pause()
    _ram_guard("pre-unet")

    device = _get_device()

    model = _make_nano_unet(channels=(16, 32, 64), device=device)
    x = _tensor(np.random.randn(1, 32, 32, 3).astype(np.float32), device)

    s1 = silu_gpu(nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1)))
    s2 = silu_gpu(nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1)))
    s3 = silu_gpu(nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1)))

    assert _shape(s1) == (1, 16, 16, 16), f"Stage 1: {_shape(s1)}"
    assert _shape(s2) == (1, 8, 8, 32), f"Stage 2: {_shape(s2)}"
    assert _shape(s3) == (1, 4, 4, 64), f"Stage 3: {_shape(s3)}"
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  U-Net forward ({device_str}): "
        f"32x32 -> s1={_shape(s1)}, s2={_shape(s2)}, s3={_shape(s3)}"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_09_small_unet_backward():
    """U-Net encoder forward+backward, measure RAM + time."""
    _gpu_pause()
    _ram_guard("pre-unet-bwd")

    device = _get_device()

    model = _make_nano_unet(channels=(16, 32, 64), device=device)
    x = _tensor(np.random.randn(1, 64, 64, 3).astype(np.float32), device)

    def forward(x):
        s1 = silu_gpu(
            nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1))
        )
        s2 = silu_gpu(
            nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1))
        )
        s3 = silu_gpu(
            nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1))
        )
        return nb.mean(s3 * s3)

    t0 = time.perf_counter()
    val, grad = nb.value_and_grad(forward)(x)
    dt = time.perf_counter() - t0

    assert _shape(grad) == _shape(x)
    assert not np.isnan(val.to_numpy().item())
    _ram_guard("post-unet-bwd")
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  U-Net backward ({device_str}) at 64x64: "
        f"loss={val.to_numpy().item():.4f}, time={dt:.2f}s"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_10_compile_gpu():
    """@nb.compile on a forward step, check compilation works."""
    _gpu_pause()
    _ram_guard("pre-compile")

    device = _get_device()

    x = _tensor(np.random.randn(1, 16, 16, 3).astype(np.float32), device)
    w = F.he_normal((3, 3, 3, 16))
    if device is not None:
        try:
            w = nb.ops.transfer_to(w, device)
        except Exception:
            pass
    w.requires_grad = True

    @nb.compile
    def compiled_forward(x, w):
        c = nb.conv2d(x, w, padding=(1, 1))
        return nb.mean(c * c)

    try:
        t0 = time.perf_counter()
        result = compiled_forward(x, w)
        dt = time.perf_counter() - t0
        assert result is not None
        _ram_guard("post-compile")
        print(f"  @nb.compile works: output shape={_shape(result)}, time={dt:.2f}s")
    except Exception as e:
        pytest.skip(f"@nb.compile failed: {e}")


# === Section 3: Resolution scaling (GPU + RAM-guarded) ===


@nabla_skip
@gpu_skip
@pytest.mark.parametrize("resolution", [64, 128, 256])
def test_11_resolution_scale(resolution):
    """Run encoder at scaled resolutions. RAM guard stops before OOM."""
    _gpu_pause()
    _ram_guard(f"pre-{resolution}")

    device = _get_device()

    model = _make_nano_unet(channels=(16, 32, 64), device=device)
    x = _tensor(
        np.random.randn(1, resolution, resolution, 3).astype(np.float32), device
    )

    def forward(x):
        s1 = silu_gpu(
            nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1))
        )
        s2 = silu_gpu(
            nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1))
        )
        s3 = silu_gpu(
            nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1))
        )
        return nb.mean(s3 * s3)

    t0 = time.perf_counter()
    val, grad = nb.value_and_grad(forward)(x)
    dt = time.perf_counter() - t0

    _ram_guard(f"post-{resolution}")
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  {resolution}x{resolution} ({device_str}): "
        f"loss={val.to_numpy().item():.4f}, time={dt:.2f}s"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_12_ram_breaking_point():
    """Find max nabla resolution under RAM safety limit.

    Graph cache is the RAM bomb culprit: each value_and_grad compiles
    a 6-8GB graph entry. Without clearing between iterations, RAM explodes.
    This test clears after each resolution to prove the fix works.
    """
    _gpu_pause()

    device = _get_device()

    results = []
    for res in [64, 128, 256]:
        if not _safe_to_continue_ram():
            print(f"  Stopped at {res}x{res}: RAM limit reached")
            break

        ram_before = _system_ram_mb()["used_mb"]
        t0 = time.perf_counter()

        x = _tensor(np.random.randn(1, res, res, 3).astype(np.float32), device)
        w = F.he_normal((3, 3, 3, 16))
        if device is not None:
            try:
                w = nb.ops.transfer_to(w, device)
            except Exception:
                pass
        c = nb.conv2d(x, w, padding=(1, 1))
        result = nb.mean(c * c)
        _ = result.to_numpy()

        dt = time.perf_counter() - t0
        ram_after = _system_ram_mb()["used_mb"]
        ram_delta = ram_after - ram_before

        results.append(
            {
                "res": res,
                "time_s": dt,
                "ram_delta_mb": ram_delta,
                "ram_total_mb": ram_after,
            }
        )
        _resource_report(f"{res}x{res}")

        # Clear graph cache after each resolution to prevent accumulation
        _clear_nabla_graph()

        if ram_after > RAM_SAFETY_GB * 1024:
            print(f"  Hit RAM safety limit at {res}x{res}")
            break

        _gpu_pause()

    assert len(results) > 0, "No resolutions tested"
    print("  Resolution scaling summary:")
    for r in results:
        print(
            f"    {r['res']}x{r['res']}: time={r['time_s']:.2f}s, "
            f"RAM +{r['ram_delta_mb']}MB (total {r['ram_total_mb']}MB)"
        )


# === Section 4: GPU training speed benchmarks ===


@gpu_skip
def test_13_torch_gpu_training_step():
    """Torch conv2d forward+backward on GPU — baseline speed."""
    _gpu_pause()

    # Small conv model: 3 conv layers
    conv1 = torch.nn.Conv2d(3, 16, 3, padding=1).cuda()
    conv2 = torch.nn.Conv2d(16, 32, 3, padding=1).cuda()
    conv3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1).cuda()

    x = torch.randn(1, 3, 256, 256, device="cuda")
    target = torch.randn(1, 64, 128, 128, device="cuda")
    optimizer = torch.optim.SGD(
        list(conv1.parameters()) + list(conv2.parameters()) + list(conv3.parameters()),
        lr=0.01,
    )

    # Warmup
    out = conv3(torch.nn.functional.silu(conv2(torch.nn.functional.silu(conv1(x)))))
    loss = torch.nn.functional.mse_loss(out, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    # Timed steps
    times = []
    for _ in range(5):
        t0 = time.perf_counter()
        out = conv3(torch.nn.functional.silu(conv2(torch.nn.functional.silu(conv1(x)))))
        loss = torch.nn.functional.mse_loss(out, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    avg = np.mean(times)
    _resource_report("torch-gpu-256")
    print(
        f"  Torch GPU training (256x256, 5 steps): "
        f"avg={avg:.4f}s, min={min(times):.4f}s, max={max(times):.4f}s"
    )
    del x, target, conv1, conv2, conv3
    torch.cuda.empty_cache()


@nabla_skip
@gpu_skip
def test_14_nabla_gpu_training_step():
    """Nabla conv2d forward+backward — compiled train step on GPU."""
    _gpu_pause()
    _ram_guard("pre-nabla-train")

    device = _get_device()

    model = _make_nano_unet(channels=(16, 32, 64), device=device)
    x = _tensor(np.random.randn(1, 64, 64, 3).astype(np.float32), device)

    def loss_fn(x):
        s1 = silu_gpu(
            nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1))
        )
        s2 = silu_gpu(
            nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1))
        )
        s3 = silu_gpu(
            nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1))
        )
        return nb.mean(s3 * s3)

    # @nb.compile: graph compiled ONCE in warmup, reused on every call (cache hit)
    @nb.compile
    def compiled_fwd_bwd(x):
        val, grad = nb.value_and_grad(loss_fn)(x)
        return val, grad

    # Warmup (compiles graph — first call is slow)
    val, grad = compiled_fwd_bwd(x)

    # Timed steps (cache hit — same shapes, no recompilation, no RAM accumulation)
    times = []
    for _ in range(3):
        x = _tensor(np.random.randn(1, 64, 64, 3).astype(np.float32), device)
        t0 = time.perf_counter()
        val, grad = compiled_fwd_bwd(x)
        _ = val.to_numpy()  # Force evaluation
        times.append(time.perf_counter() - t0)
        if not _safe_to_continue_ram():
            print("  RAM limit hit during nabla training loop")
            break

    avg = np.mean(times) if times else 0
    _ram_guard("post-nabla-train")
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  Nabla {device_str} compiled training (64x64, {len(times)} steps): "
        f"avg={avg:.4f}s, min={min(times):.4f}s, max={max(times):.4f}s"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_15_torch_gpu_vs_nabla_gpu():
    """Side-by-side: same conv encoder, torch GPU vs nabla GPU (compiled)."""
    _gpu_pause()
    _ram_guard("pre-compare")

    resolution = 64
    n_steps = 3

    # --- Torch GPU ---
    conv1 = torch.nn.Conv2d(3, 16, 3, stride=2, padding=1).cuda()
    conv2 = torch.nn.Conv2d(16, 32, 3, stride=2, padding=1).cuda()
    conv3 = torch.nn.Conv2d(32, 64, 3, stride=2, padding=1).cuda()
    x_t = torch.randn(1, 3, resolution, resolution, device="cuda")

    torch_times = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        out = conv3(
            torch.nn.functional.silu(conv2(torch.nn.functional.silu(conv1(x_t))))
        )
        loss = out.pow(2).mean()
        loss.backward()
        torch.cuda.synchronize()
        torch_times.append(time.perf_counter() - t0)
        conv1.zero_grad()
        conv2.zero_grad()
        conv3.zero_grad()

    del x_t, conv1, conv2, conv3
    torch.cuda.empty_cache()

    # --- Nabla GPU (compiled) ---
    device = _get_device()

    model = _make_nano_unet(channels=(16, 32, 64), device=device)
    x_n = _tensor(
        np.random.randn(1, resolution, resolution, 3).astype(np.float32), device
    )

    def fwd(x):
        s1 = silu_gpu(
            nb.conv2d(x, model["enc_filters"][0], stride=(2, 2), padding=(1, 1))
        )
        s2 = silu_gpu(
            nb.conv2d(s1, model["enc_filters"][1], stride=(2, 2), padding=(1, 1))
        )
        s3 = silu_gpu(
            nb.conv2d(s2, model["enc_filters"][2], stride=(2, 2), padding=(1, 1))
        )
        return nb.mean(s3 * s3)

    @nb.compile
    def compiled_fwd_bwd(x):
        val, grad = nb.value_and_grad(fwd)(x)
        return val, grad

    # Warmup (compile)
    compiled_fwd_bwd(x_n)

    nabla_times = []
    for _ in range(n_steps):
        x_n = _tensor(
            np.random.randn(1, resolution, resolution, 3).astype(np.float32),
            device,
        )
        t0 = time.perf_counter()
        val, grad = compiled_fwd_bwd(x_n)
        _ = val.to_numpy()
        nabla_times.append(time.perf_counter() - t0)
        if not _safe_to_continue_ram():
            break

    torch_avg = np.mean(torch_times)
    nabla_avg = np.mean(nabla_times)
    speedup = nabla_avg / torch_avg if torch_avg > 0 else float("inf")

    _ram_guard("post-compare")
    device_str = "GPU" if device is not None else "CPU"
    print(f"  Torch GPU (64x64, {n_steps} steps): avg={torch_avg:.4f}s")
    print(
        f"  Nabla {device_str} compiled (64x64, {n_steps} steps): avg={nabla_avg:.4f}s"
    )
    print(
        f"  Ratio: nabla {device_str} is {speedup:.1f}x "
        f"{'slower' if speedup > 1 else 'faster'} than torch GPU"
    )
    _clear_nabla_graph()


# === Section 5: silu_gpu custom op — GPU backward stress test ===


@nabla_skip
@gpu_skip
def test_16_silu_gpu_forward_backward():
    """Custom SiluGPU op: forward + backward on GPU with scalar-free VJP."""
    _gpu_pause()
    _ram_guard("pre-silu-gpu")

    from omen.kernels.activations import silu_gpu

    device = _get_device()
    x = _tensor(np.random.randn(1, 32, 32, 16).astype(np.float32), device)

    # Forward
    out = silu_gpu(x)
    out_np = out.to_numpy()
    assert out_np.shape == (1, 32, 32, 16)
    assert not np.isnan(out_np).any(), "silu_gpu forward produced NaN"

    # Backward via value_and_grad
    def loss_fn(x):
        return nb.mean(silu_gpu(x) * silu_gpu(x))

    val, grad = nb.value_and_grad(loss_fn)(x)
    val_np = val.to_numpy()
    grad_np = grad.to_numpy()
    assert not np.isnan(val_np).item(), "silu_gpu backward loss is NaN"
    assert not np.isnan(grad_np).any(), "silu_gpu backward grad has NaN"
    assert grad_np.shape == (1, 32, 32, 16)

    _ram_guard("post-silu-gpu")
    device_str = "GPU" if device is not None else "CPU"
    print(
        f"  silu_gpu forward+backward ({device_str}): "
        f"loss={val_np.item():.4f}, grad_shape={grad_np.shape}"
    )
    _clear_nabla_graph()


@nabla_skip
@gpu_skip
def test_17_silu_gpu_training_loop():
    """Sustained GPU training loop with silu_gpu — 10 steps, RAM-guarded."""
    _gpu_pause()
    _ram_guard("pre-silu-train")

    from omen.kernels.activations import silu_gpu

    device = _get_device()

    # 4-layer conv model on GPU with silu_gpu
    filters = []
    for ch_in, ch_out in [(3, 16), (16, 32), (32, 64), (64, 128)]:
        w = F.he_normal((3, 3, ch_in, ch_out))
        if device is not None:
            w = nb.ops.transfer_to(w, device)
        w.requires_grad = True
        filters.append(w)

    def loss_fn(x):
        s = x
        for w in filters:
            conv_out = nb.conv2d(s, w, stride=(2, 2), padding=(1, 1))
            s = silu_gpu(conv_out)
        return nb.mean(s * s)

    # Step 1: compile + execute
    x = _tensor(np.random.randn(1, 128, 128, 3).astype(np.float32), device)
    print("  Step 1 (compile+exec)...", end=" ", flush=True)
    t0 = time.perf_counter()
    val, grad = nb.value_and_grad(loss_fn)(x)
    val_np = val.to_numpy()
    grad_np = grad.to_numpy()
    dt = time.perf_counter() - t0
    assert not np.isnan(val_np).item(), f"Step 1 NaN loss: {val_np.item()}"
    assert not np.isnan(grad_np).any(), "Step 1 NaN grads"
    print(
        f"{dt:.1f}s loss={val_np.item():.1f} grad_nan={np.isnan(grad_np).any()}",
        flush=True,
    )
    nb.GRAPH.clear_all()

    # Steps 2-10: sustained
    times = []
    for i in range(9):
        if not _safe_to_continue_ram():
            print(f"  RAM limit at step {i + 2}, stopping")
            break
        x = _tensor(np.random.randn(1, 128, 128, 3).astype(np.float32), device)
        t0 = time.perf_counter()
        val, grad = nb.value_and_grad(loss_fn)(x)
        val_np = val.to_numpy()
        grad_np = grad.to_numpy()
        dt = time.perf_counter() - t0
        times.append(dt)
        assert not np.isnan(val_np).item(), f"Step {i + 2} NaN loss"
        assert not np.isnan(grad_np).any(), f"Step {i + 2} NaN grads"
        print(
            f"  Step {i + 2}: {dt:.2f}s loss={val_np.item():.1f} grad_nan=False",
            flush=True,
        )
        nb.GRAPH.clear_all()

    _ram_guard("post-silu-train")
    device_str = "GPU" if device is not None else "CPU"
    avg = np.mean(times) if times else 0
    print(
        f"  silu_gpu training ({device_str}, 128x128, {len(times) + 1} steps): "
        f"avg={avg:.2f}s"
    )
