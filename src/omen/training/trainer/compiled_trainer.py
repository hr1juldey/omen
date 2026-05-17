"""Compiled OmenTrainer — wraps compiled_step with tiling and checkpoints.

Supports GPU training: detects accelerator, transfers params + opt states
+ tile tensors to GPU. Falls back to CPU if no GPU detected.
"""

import json
import logging
import time

import numpy as np
import nabla as nb
from max.driver import CPU, Accelerator, accelerator_count
from nabla.nn.optim import adamw_init, adamw_update

from omen.training.trainer.compiled_step import compiled_loss_and_grads
from omen.training.trainer.optimizers import COMPONENT_LRS, COMPONENT_PREFIXES

logger = logging.getLogger("omen.training.trainer.compiled")


def _transfer(tree, device):
    """Recursively transfer a pytree of nabla tensors to device."""

    def _move(t):
        if hasattr(t, "to_numpy"):
            return nb.ops.transfer_to(t, device)
        return t

    if isinstance(tree, dict):
        return {k: _transfer(v, device) for k, v in tree.items()}
    return _move(tree)


def _numpy_to_nabla(tree):
    """Convert numpy arrays in a nested dict to nabla tensors."""

    def _convert(t):
        if isinstance(t, np.ndarray):
            return nb.Tensor.from_dlpack(t.astype(np.float32))
        return t

    if isinstance(tree, dict):
        return {k: _numpy_to_nabla(v) for k, v in tree.items()}
    return _convert(tree)


class CompiledOmenTrainer:
    """Training loop using @nb.compile for graph reuse and RAM stability.

    One-time ~300s warmup compiles forward+backward+optimizer into a
    single graph entry. Subsequent steps hit cache (~5ms/step).
    No clear_all() — graph is reused, not destroyed.

    Automatically uses GPU if available, falls back to CPU.
    """

    def __init__(self, model, config, weight_decay=0.01,
                 warmup_steps=0, total_steps=1000):
        self.model = model
        self.config = config
        self.weight_decay = weight_decay
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps

        # Device: CPU for now (GPU @nb.compile uses 25GB RAM during compilation)
        if accelerator_count() > 0:
            self.device = CPU()  # TODO: switch to Accelerator() when RAM is fixed
            logger.info("GPU available but using CPU for compilation RAM safety")
        else:
            self.device = CPU()
            logger.info("No GPU — training on CPU")

        # Transfer model params to device
        cpu_params = model.state_dict()
        self.params = _transfer(cpu_params, self.device)
        n = len(self.params)
        logger.info("Transferred %d param tensors to %s", n, self.device)

        # Optimizer states (zeros_like inherits device from params)
        self.opt_states = self._init_opt_states()
        self.step_count = 0

        # Cache CPU params for eager scene encoding (avoids transfer every step)
        self._cpu_params = _transfer(self.params, CPU())

    def _init_opt_states(self):
        """Create per-component AdamW states for active components."""
        states = {}
        c = self.config.components
        for name, prefixes in COMPONENT_PREFIXES.items():
            subset = {
                k: v
                for k, v in self.params.items()
                if any(k.startswith(p) for p in prefixes)
            }
            if not subset:
                continue
            if name == "ar_predictor" and not c.ar_predictor:
                continue
            if name == "episodic_correction" and not c.episodic_correction:
                continue
            if name.startswith(
                ("shared_", "material_", "light_", "geometry_", "motion_")
            ):
                if not c.moe:
                    continue
            # adamw_init creates zeros_like — same device as params
            states[name] = adamw_init(subset)
        return states

    @property
    def iteration(self):
        return self.step_count

    def _scaled_lr(self, base_lr):
        """Linear warmup then cosine decay to 0 over total_steps."""
        step = self.step_count
        # Warmup: 0 -> base_lr over warmup_steps
        if self.warmup_steps > 0 and step < self.warmup_steps:
            return base_lr * (step + 1) / self.warmup_steps
        # Cosine decay after warmup
        progress = (step - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
        progress = min(progress, 1.0)
        return base_lr * 0.5 * (1.0 + __import__("math").cos(__import__("math").pi * progress))

    def _encode_scene(self, scene_input):
        """Pre-encode scene graph on CPU, then transfer latent to device.

        Scene graph data is numpy — encoding runs eagerly on CPU with CPU
        param copies. The resulting latent is transferred to GPU for the
        compiled train step.
        """
        if isinstance(scene_input, dict):
            from omen.model.functional import _extract_prefix, scene_encoder_fn

            sg = _numpy_to_nabla(scene_input)
            # Use cached CPU params for eager scene encoding
            p = _extract_prefix(self._cpu_params, "scene_encoder.")
            latent = scene_encoder_fn(p, sg)
            # Realize: prevent lazy graph from leaking into @nb.compile
            nb.realize_all(latent)
        else:
            latent = scene_input
        return _transfer(latent, self.device)

    def train_step_tiled(self, noisy, gt, scene_input, tile_size=256):
        """Run compiled train step per tile, accumulate loss.

        Args:
            scene_input: scene_graph dict or pre-encoded (B, latent_dim) tensor.

        Returns:
            metrics dict compatible with OmenTrainer interface.
        """
        scene_latent = self._encode_scene(scene_input)

        # Ensure batch dim + 4 channels: (H,W,C) -> (1,H,W,4) for numpy arrays
        if isinstance(noisy, np.ndarray):
            if noisy.ndim == 3:
                noisy = noisy[np.newaxis]
            if noisy.shape[-1] == 3:
                noisy = np.pad(noisy, ((0, 0), (0, 0), (0, 0), (0, 1)),
                               constant_values=1.0)
            noisy = nb.Tensor.from_dlpack(noisy.astype(np.float32))
        if isinstance(gt, np.ndarray):
            if gt.ndim == 3:
                gt = gt[np.newaxis]
            if gt.shape[-1] == 3:
                gt = np.pad(gt, ((0, 0), (0, 0), (0, 0), (0, 1)),
                            constant_values=1.0)
            gt = nb.Tensor.from_dlpack(gt.astype(np.float32))

        B, H, W, C = (int(d) for d in noisy.shape)
        total_loss = 0.0
        n_tiles = 0
        t0 = time.time()

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
                tile_noisy = noisy[:, y:y2, x:x2, :]
                tile_gt = gt[:, y:y2, x:x2, :]

                # Transfer tiles to device
                tile_noisy = _transfer(tile_noisy, self.device)
                tile_gt = _transfer(tile_gt, self.device)

                # Compiled forward + backward (no optimizer inside — avoids
                # nabla's adamw_step CPU scalar device mismatch on GPU)
                loss, grads = compiled_loss_and_grads(
                    self.params,
                    tile_noisy,
                    tile_gt,
                    scene_latent,
                )

                # Eager per-component AdamW update (CPU scalars OK in eager mode)
                new_params = dict(self.params)
                new_states = {}
                for name in sorted(COMPONENT_LRS.keys()):
                    if name not in self.opt_states:
                        continue
                    prefixes = COMPONENT_PREFIXES[name]
                    subset_p = {
                        k: new_params[k]
                        for k in self.params
                        if any(k.startswith(p) for p in prefixes)
                    }
                    subset_g = {
                        k: grads[k]
                        for k in grads
                        if any(k.startswith(p) for p in prefixes)
                    }
                    if not subset_p:
                        continue
                    lr = self._scaled_lr(COMPONENT_LRS[name])
                    updated_p, updated_state = adamw_update(
                        subset_p,
                        subset_g,
                        self.opt_states[name],
                        lr=lr,
                        weight_decay=self.weight_decay,
                    )
                    new_params.update(updated_p)
                    new_states[name] = updated_state

                self.params = new_params
                self.opt_states = new_states

                # Realize all lazy tensors — breaks computation graph chain
                # that would otherwise accumulate across steps (RAM leak)
                nb.realize_all(*self.params.values())
                for s in self.opt_states.values():
                    nb.realize_all(*s["m"].values(), *s["v"].values())

                # Transfer loss to CPU for logging
                loss_cpu = _transfer(loss, CPU())
                total_loss += float(loss_cpu.to_numpy())
                n_tiles += 1

        self.step_count += 1
        avg_loss = total_loss / max(n_tiles, 1)
        elapsed = time.time() - t0
        logger.info(
            "Step %d: loss=%.2f tiles=%d time=%.1fs device=%s lr=%.2e",
            self.step_count,
            avg_loss,
            n_tiles,
            elapsed,
            self.device,
            self._scaled_lr(COMPONENT_LRS.get("encoder", 5e-5)),
        )
        return {
            "total_loss": avg_loss,
            "num_tiles": n_tiles,
            "iteration": self.step_count,
        }

    def flush_graph_cache(self):
        """No-op: compiled graph is reused, not destroyed."""
        logger.info("Compiled mode: graph cache retained (no flush needed)")

    def save_checkpoint_rotating(self, ckpt_dir):
        """Save rotating checkpoint (compatible with OmenTrainer interface)."""
        import os

        os.makedirs(ckpt_dir, exist_ok=True)
        path = os.path.join(ckpt_dir, f"step_{self.step_count:06d}.omen.npz")
        self.save_checkpoint(path)

    def save_checkpoint(self, path):
        """Save params + optimizer states as .npz (transfers to CPU first)."""
        arrays = {}
        meta = {"step": self.step_count, "components": []}
        for k, v in self.params.items():
            arrays[f"p/{k}"] = _transfer(v, CPU()).to_numpy()
        for name, state in self.opt_states.items():
            meta["components"].append(name)
            for k, v in state["m"].items():
                arrays[f"opt/{name}/m/{k}"] = _transfer(v, CPU()).to_numpy()
            for k, v in state["v"].items():
                arrays[f"opt/{name}/v/{k}"] = _transfer(v, CPU()).to_numpy()
            step_val = state["step"]
            arrays[f"opt/{name}/step"] = np.array(
                float(step_val) if hasattr(step_val, "to_numpy") else step_val
            )
        arrays["_meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        logger.info("Checkpoint saved to %s (step %d)", path, self.step_count)

    def load_checkpoint(self, path):
        """Load params + optimizer states from .npz, transfer to device."""
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["_meta"]))
        self.step_count = meta["step"]
        # Load to CPU first, then transfer to device
        self.params = {}
        for k in data.files:
            if k.startswith("p/"):
                cpu_t = nb.Tensor.from_dlpack(data[k])
                self.params[k[2:]] = _transfer(cpu_t, self.device)
        self.opt_states = {}
        for name in meta["components"]:
            m = {
                k[2 + len(name) + 4 :]: _transfer(
                    nb.Tensor.from_dlpack(data[k]), self.device
                )
                for k in data.files
                if k.startswith(f"opt/{name}/m/")
            }
            v = {
                k[2 + len(name) + 4 :]: _transfer(
                    nb.Tensor.from_dlpack(data[k]), self.device
                )
                for k in data.files
                if k.startswith(f"opt/{name}/v/")
            }
            step = float(data[f"opt/{name}/step"])
            self.opt_states[name] = {"m": m, "v": v, "step": step}
        logger.info(
            "Checkpoint loaded from %s (step %d, device=%s)",
            path,
            self.step_count,
            self.device,
        )
