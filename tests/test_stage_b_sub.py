"""Run Stage B in subprocess — pure-nabla first, then Mojo."""
import subprocess
import time


def run_stage(mode):
    silu_name = "silu_mojo" if mode == "mojo" else "silu_gpu"
    sig_name = "sigmoid_mojo" if mode == "mojo" else "sigmoid_gpu"

    script = f"""
import time, numpy as np
import nabla as nb
from nabla.ops import conv2d
from omen.kernels.activations import sigmoid_gpu, silu_gpu, square
from omen.kernels.activations_gpu import sigmoid_mojo, silu_mojo

print("MODE: {mode}")
np.random.seed(42)

def _he(s): return np.random.randn(*s).astype(np.float32) * 0.02
def _z(n): return np.zeros(n, dtype=np.float32)

p = dict()
p["re_f1"], p["re_b1"] = _he((3,3,4,16)), _z(16)
p["re_f2"], p["re_b2"] = _he((3,3,16,32)), _z(32)
p["re_f3"], p["re_b3"] = _he((3,3,32,64)), _z(64)
p["re_pw"], p["re_pb"] = _he((64,256)), _z(256)
p["se_w1"], p["se_b1"] = _he((18,64)), _z(64)
p["se_pw"], p["se_pb"] = _he((64,256)), _z(256)
p["ca_gw"], p["ca_gb"] = _he((256,256)), _z(256)

silu_fn = {silu_name}
sig_fn = {sig_name}

noisy = nb.Tensor.from_dlpack(np.random.rand(1,64,64,4).astype(np.float32))
scene_f = nb.Tensor.from_dlpack(np.random.rand(1,18).astype(np.float32))
gt_lat = nb.Tensor.from_dlpack(np.random.randn(1,256).astype(np.float32)*0.01)

def loss_fn(p, noisy, scene_f, gt):
    x = silu_fn(conv2d(noisy, p["re_f1"], stride=2, padding=1, bias=p["re_b1"]))
    x = silu_fn(conv2d(x, p["re_f2"], stride=2, padding=1, bias=p["re_b2"]))
    x = silu_fn(conv2d(x, p["re_f3"], stride=2, padding=1, bias=p["re_b3"]))
    rl = x.mean(axis=(1,2)) @ p["re_pw"] + p["re_pb"]
    h = silu_fn(scene_f @ p["se_w1"] + p["se_b1"])
    sl = h @ p["se_pw"] + p["se_pb"]
    g = sig_fn(rl @ p["ca_gw"] + p["ca_gb"])
    fused = rl + g * sl
    return nb.mean(square(fused - gt))

print("Calling value_and_grad...")
t0 = time.time()
val, grads = nb.value_and_grad(loss_fn, argnums=0)(p, noisy, scene_f, gt_lat)
dt = time.time() - t0
print(f"RESULT: val={{float(val.to_numpy()):.6f}} time={{dt:.1f}}s")
print("PASS")
"""

    t0 = time.time()
    result = subprocess.run(
        ["uv", "run", "python", "-u", "-c", script],
        capture_output=True, text=True, timeout=120,
        cwd="/home/riju279/Documents/Projects/MOJO/Cycles_mojo/omen",
    )
    dt = time.time() - t0
    rc = result.returncode
    stdout = result.stdout.strip()[-400:]
    stderr = result.stderr.strip()[-300:]

    print(f"\n{'='*50}")
    print(f"MODE: {mode}  RC={rc}  wall={dt:.1f}s")
    if rc == 0:
        print(f"OUT: {stdout}")
    elif rc < 0:
        sig = -rc
        names = {11: "SIGSEGV", 9: "SIGKILL", 6: "SIGABRT"}
        print(f"SIGNAL: {names.get(sig, sig)}")
        print(f"OUT: {stdout}")
        if stderr:
            print(f"ERR: {stderr}")
    else:
        print(f"OUT: {stdout}")
        if stderr:
            print(f"ERR: {stderr}")
    return rc


run_stage("pure")
run_stage("mojo")
