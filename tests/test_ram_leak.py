#!/usr/bin/env python3
"""RAM leak diagnostic — isolate which component leaks."""

import gc
import os
import tracemalloc
import time

import numpy as np
import nabla as nb
from max.driver import CPU

from omen.config import OmenConfig
from omen.model.jepa import OmenJEPA
from omen.training.trainer.compiled_step import compiled_loss_and_grads


def _ram_mb():
    """Process RSS in MB."""
    with open(f"/proc/{os.getpid()}/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    return 0


def _ram_system():
    with open("/proc/meminfo") as f:
        info = {}
        for line in f:
            parts = line.split()
            if len(parts) >= 2:
                info[parts[0].rstrip(":")] = int(parts[1]) // 1024
    return info.get("MemTotal", 0) - info.get("MemAvailable", 0)


def main():
    tracemalloc.start()

    config = OmenConfig.v1_dense()
    model = OmenJEPA(config=config)
    params = model.state_dict()

    # Fake inputs (same shapes as real training)
    noisy = nb.Tensor.from_dlpack(np.random.randn(1, 512, 512, 4).astype(np.float32))
    gt = nb.Tensor.from_dlpack(np.random.randn(1, 512, 512, 4).astype(np.float32))
    scene_latent = nb.Tensor.from_dlpack(np.random.randn(1, 1024).astype(np.float32))

    print(f"Baseline RAM: {_ram_mb()}MB (process), {_ram_system()}MB (system)")
    print("=" * 60)

    # Phase 1: Just compiled_loss_and_grads (no optimizer)
    print("\n--- Phase 1: compiled_loss_and_grads only (no optimizer) ---")
    for i in range(20):
        loss, grads = compiled_loss_and_grads(params, noisy, gt, scene_latent)
        nb.realize_all(loss, *grads.values())
        # Explicitly delete to see if it helps
        del loss, grads
        gc.collect()
        ram = _ram_mb()
        sys_ram = _ram_system()
        if i % 5 == 0 or i < 3:
            print(f"  Step {i+1:3d}: process={ram}MB system={sys_ram}MB")
    print(f"  Phase 1 final: {_ram_mb()}MB (process), {_ram_system()}MB (system)")

    # Check top allocations
    snapshot = tracemalloc.take_snapshot()
    top = snapshot.statistics("lineno")[:5]
    print("\n  Top allocations:")
    for stat in top:
        print(f"    {stat}")

    # Phase 2: compiled_loss_and_grads + eager optimizer update
    print("\n--- Phase 2: compiled fwd+bwd + eager AdamW ---")
    from nabla.nn.optim import adamw_update
    from omen.training.trainer.optimizers import COMPONENT_LRS, COMPONENT_PREFIXES

    # Init optimizer states
    opt_states = {}
    for name, prefixes in COMPONENT_PREFIXES.items():
        subset = {k: v for k, v in params.items() if any(k.startswith(p) for p in prefixes)}
        if not subset:
            continue
        from nabla.nn.optim import adamw_init
        opt_states[name] = adamw_init(subset)

    for i in range(20):
        loss, grads = compiled_loss_and_grads(params, noisy, gt, scene_latent)
        nb.realize_all(loss, *grads.values())

        # Optimizer update (same as compiled_trainer.py)
        new_params = dict(params)
        new_states = {}
        for name in sorted(COMPONENT_LRS.keys()):
            if name not in opt_states:
                continue
            prefixes = COMPONENT_PREFIXES[name]
            subset_p = {k: new_params[k] for k in params if any(k.startswith(p) for p in prefixes)}
            subset_g = {k: grads[k] for k in grads if any(k.startswith(p) for p in prefixes)}
            if not subset_p:
                continue
            updated_p, updated_state = adamw_update(subset_p, subset_g, opt_states[name], lr=5e-6, weight_decay=0.01)
            new_params.update(updated_p)
            new_states[name] = updated_state

        params = new_params
        opt_states = new_states
        nb.realize_all(*params.values())
        for s in opt_states.values():
            nb.realize_all(*s["m"].values(), *s["v"].values())

        del loss, grads, new_params, new_states
        gc.collect()

        ram = _ram_mb()
        sys_ram = _ram_system()
        if i % 5 == 0 or i < 3:
            print(f"  Step {i+1:3d}: process={ram}MB system={sys_ram}MB")

    print(f"  Phase 2 final: {_ram_mb()}MB (process), {_ram_system()}MB (system)")

    snapshot2 = tracemalloc.take_snapshot()
    top2 = snapshot2.statistics("lineno")[:5]
    print("\n  Top allocations:")
    for stat in top2:
        print(f"    {stat}")

    # Phase 2 diff
    diff = snapshot2.compare_to(snapshot, "lineno")[:5]
    print("\n  RAM growth between phases:")
    for stat in diff:
        print(f"    {stat}")

    tracemalloc.stop()


if __name__ == "__main__":
    main()
