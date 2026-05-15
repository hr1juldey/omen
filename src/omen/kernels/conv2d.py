"""Conv2d via im2col + matmul with optional Mojo GPU acceleration.

Two paths:
  1. Pure nabla (default): pad/slice/concat matmul — nabla auto-diffs through
  2. Mojo GPU: call_custom_kernel for im2col/col2im — faster data rearrangement

Both paths use nabla matmul for the compute-heavy part (forward + gradients).
The Mojo GPU path accelerates the memory-bound im2col data rearrangement.

Forward: im2col(x) -> patches, then patches @ filter_flat -> output
Backward: automatic via nabla's built-in autodiff (matmul, reshape, etc.)
"""

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger("omen.kernels.conv2d")

KERNEL_DIR = Path(__file__).parent

try:
    from nabla.ops import UnaryOperation, call_custom_kernel

    NABLA_AVAILABLE = True
except ImportError:
    UnaryOperation = object
    call_custom_kernel = None
    NABLA_AVAILABLE = False


def _conv_out_size(in_size: int, kernel: int, stride: int, pad: int) -> int:
    return (in_size + 2 * pad - kernel) // stride + 1


def _im2col(x, kh, kw, sh, sw, ph, pw):
    """Extract patches from NHWC tensor using nabla ops.

    Input:  x (B, H, W, C_in)
    Output: patches (B*H_out*W_out, Kh*Kw*C_in)
    """
    import nabla as nb

    b, h, w, c_in = (int(d) for d in x.shape)
    h_out = _conv_out_size(h, kh, sh, ph)
    w_out = _conv_out_size(w, kw, sw, pw)

    # Pad spatial dims
    if ph > 0 or pw > 0:
        x = nb.pad(x, ((0, 0), (ph, ph), (pw, pw), (0, 0)))

    # Extract Kh*Kw patches and concatenate along channel dim
    patches = []
    for ki in range(kh):
        for kj in range(kw):
            if sh == 1:
                patch = x[:, ki : ki + h_out, kj : kj + w_out, :]
            else:
                # Strided access: extract then subsample via reshape trick
                rows = x[:, ki : ki + h_out * sh, kj : kj + w_out * sh, :]
                # (B, h_out*sh, w_out*sh, C) -> (B, h_out, sh, w_out, sh, C)
                r = nb.reshape(rows, (b, h_out, sh, w_out, sh, c_in))
                patch = r[:, :, 0, :, 0, :]
            # Flatten spatial dims: (B*H_out*W_out, C_in)
            patch = nb.reshape(patch, (b * h_out * w_out, c_in))
            patches.append(patch)

    # (B*H_out*W_out, Kh*Kw*C_in)
    return nb.concatenate(patches, axis=1)


# ---------------------------------------------------------------------------
# Mojo GPU im2col / col2im Operation wrappers
# ---------------------------------------------------------------------------


class Im2colOp(UnaryOperation):
    """Nabla op wrapping Mojo conv2d_im2col GPU kernel."""

    @property
    def name(self) -> str:
        return "conv2d_im2col"

    def __init__(self, kh, kw, sh, sw, ph, pw):
        self.kh = kh
        self.kw = kw
        self.sh = sh
        self.sw = sw
        self.ph = ph
        self.pw = pw

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        x = args[0]
        b, h, w, cin = (int(d) for d in x.shape)
        hout = (h + 2 * self.ph - self.kh) // self.sh + 1
        wout = (w + 2 * self.pw - self.kw) // self.sw + 1
        return [(b * hout * wout, self.kh * self.kw * cin)], [x.dtype], [x.device]

    def kernel(self, args, kwargs):
        import nabla as nb

        x = args[0]
        b, h, w, cin = (int(d) for d in x.shape)
        hout = (h + 2 * self.ph - self.kh) // self.sh + 1
        wout = (w + 2 * self.pw - self.kw) // self.sw + 1
        params = np.array(
            [
                hout, wout, self.kh, self.kw, cin,
                self.sh, self.sw, self.ph, self.pw, h, w,
            ],
            dtype=np.float32,
        )
        params_t = nb.array(params)
        result = call_custom_kernel(
            "conv2d_im2col", str(KERNEL_DIR), x, params_t,
        )
        return [result]


class Col2imOp(UnaryOperation):
    """Nabla op wrapping Mojo conv2im_col2im GPU kernel."""

    @property
    def name(self) -> str:
        return "conv2im_col2im"

    def __init__(self, kh, kw, sh, sw, ph, pw, b, h, w, cin):
        self.kh = kh
        self.kw = kw
        self.sh = sh
        self.sw = sw
        self.ph = ph
        self.pw = pw
        self.b = b
        self.h = h
        self.w = w
        self.cin = cin

    def compute_physical_shape(self, args, kwargs, output_sharding=None):
        return (
            [(self.b, self.h, self.w, self.cin)],
            [args[0].dtype],
            [args[0].device],
        )

    def kernel(self, args, kwargs):
        import nabla as nb

        col = args[0]
        hout = (self.h + 2 * self.ph - self.kh) // self.sh + 1
        wout = (self.w + 2 * self.pw - self.kw) // self.sw + 1
        params = np.array(
            [
                hout, wout, self.kh, self.kw, self.cin,
                self.sh, self.sw, self.ph, self.pw, self.h, self.w,
            ],
            dtype=np.float32,
        )
        params_t = nb.array(params)
        result = call_custom_kernel(
            "conv2im_col2im", str(KERNEL_DIR), col, params_t,
        )
        return [result]


def _im2col_gpu(x, kh, kw, sh, sw, ph, pw):
    """Extract patches using Mojo GPU im2col kernel."""
    import nabla as nb

    op = Im2colOp(kh=kh, kw=kw, sh=sh, sw=sw, ph=ph, pw=pw)
    return op([x], {})[0]


def conv2d_safe(
    x,
    filter,
    stride=1,
    padding=0,
    bias=None,
    use_gpu=False,
):
    """Drop-in replacement for nb.conv2d using im2col + matmul.

    Args:
        x: (B, H, W, C_in) NHWC tensor
        filter: (Kh, Kw, C_in, C_out) HWIO filter tensor
        stride: int or (sh, sw)
        padding: int or (ph, pw) or (ph_top, ph_bot, pw_left, pw_right)
        bias: optional (C_out,) bias tensor
        use_gpu: if True, use Mojo GPU im2col kernel (requires GPU + nabla)

    Returns:
        (B, H_out, W_out, C_out) output tensor
    """
    import nabla as nb

    if isinstance(stride, int):
        sh, sw = stride, stride
    else:
        sh, sw = stride

    if isinstance(padding, int):
        ph, pw = padding, padding
    elif len(padding) == 2:
        ph, pw = padding
    elif len(padding) == 4:
        ph, pw = padding[0], padding[2]
    else:
        ph, pw = 0, 0

    kh, kw, c_in, c_out = (int(d) for d in filter.shape)
    b, h, w, _ = (int(d) for d in x.shape)
    h_out = _conv_out_size(h, kh, sh, ph)
    w_out = _conv_out_size(w, kw, sw, pw)

    # im2col: (B*H_out*W_out, Kh*Kw*C_in)
    if use_gpu and NABLA_AVAILABLE:
        try:
            patches = _im2col_gpu(x, kh, kw, sh, sw, ph, pw)
        except Exception as exc:
            logger.warning("Mojo GPU im2col failed (%s) — pure nabla fallback", exc)
            patches = _im2col(x, kh, kw, sh, sw, ph, pw)
    else:
        patches = _im2col(x, kh, kw, sh, sw, ph, pw)

    # matmul: (B*H_out*W_out, Kh*Kw*C_in) @ (Kh*Kw*C_in, C_out)
    filt_flat = nb.reshape(filter, (kh * kw * c_in, c_out))
    out_flat = nb.matmul(patches, filt_flat)

    # Reshape to NHWC
    out = nb.reshape(out_flat, (b, h_out, w_out, c_out))

    if bias is not None:
        out = out + bias

    return out
