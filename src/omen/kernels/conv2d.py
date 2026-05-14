"""Conv2d via im2col + matmul using pure nabla operations.

No custom Mojo kernels or Operation subclasses needed. Uses nabla built-ins
(pad, slice, concatenate, matmul, reshape) which nabla auto-diffs through.

The Mojo GPU im2col/col2im kernels (conv2d_im2col.mojo, conv2im.mojo) are
available for future optimization — swap the pure-nabla im2col/col2im with
call_custom_kernel when the MAX kernel loading is properly configured.

Forward: im2col(x) -> patches, then patches @ filter_flat -> output
Backward: automatic via nabla's built-in autodiff (matmul, reshape, etc.)
"""

import logging

logger = logging.getLogger("omen.kernels.conv2d")


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
                patch = x[:, ki:ki + h_out, kj:kj + w_out, :]
            else:
                # Strided access: extract then subsample via reshape trick
                rows = x[:, ki:ki + h_out * sh, kj:kj + w_out * sh, :]
                # (B, h_out*sh, w_out*sh, C) -> (B, h_out, sh, w_out, sh, C)
                r = nb.reshape(rows, (b, h_out, sh, w_out, sh, c_in))
                patch = r[:, :, 0, :, 0, :]
            # Flatten spatial dims: (B*H_out*W_out, C_in)
            patch = nb.reshape(patch, (b * h_out * w_out, c_in))
            patches.append(patch)

    # (B*H_out*W_out, Kh*Kw*C_in)
    return nb.concatenate(patches, axis=1)


def conv2d_safe(
    x, filter, stride=1, padding=0, bias=None,
):
    """Drop-in replacement for nb.conv2d using im2col + matmul.

    Pure nabla implementation — no custom Mojo kernels needed.
    Nabla auto-diffs through all ops (matmul, reshape, pad, etc.),
    so gradients for both input and filter are computed automatically.

    Args:
        x: (B, H, W, C_in) NHWC tensor
        filter: (Kh, Kw, C_in, C_out) HWIO filter tensor
        stride: int or (sh, sw)
        padding: int or (ph, pw) or (ph_top, ph_bot, pw_left, pw_right)
        bias: optional (C_out,) bias tensor

    Returns:
        (B, H_out, W_out, C_out) output tensor

    Note: nabla matmul is the compute-heavy part (forward + gradients).
    im2col is a memory-bound data rearrangement via pad/slice/concat.
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
    patches = _im2col(x, kh, kw, sh, sw, ph, pw)

    # matmul: (B*H_out*W_out, Kh*Kw*C_in) @ (Kh*Kw*C_in, C_out)
    filt_flat = nb.reshape(filter, (kh * kw * c_in, c_out))
    out_flat = nb.matmul(patches, filt_flat)

    # Reshape to NHWC
    out = nb.reshape(out_flat, (b, h_out, w_out, c_out))

    if bias is not None:
        out = out + bias

    return out
