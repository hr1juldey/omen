"""Compiled OmenTrainer — wraps compiled_step with tiling and checkpoints."""

import json
import logging
import time

import numpy as np
import nabla as nb
from nabla.nn.optim import adamw_init

from omen.training.trainer.compiled_step import compiled_train_step
from omen.training.trainer.optimizers import COMPONENT_PREFIXES

logger = logging.getLogger("omen.training.trainer.compiled")


class CompiledOmenTrainer:
    """Training loop using @nb.compile for graph reuse and RAM stability.

    One-time ~300s warmup compiles forward+backward+optimizer into a
    single graph entry. Subsequent steps hit cache (~5ms/step).
    No clear_all() — graph is reused, not destroyed.
    """

    def __init__(self, model, config, weight_decay=0.01):
        self.model = model
        self.config = config
        self.weight_decay = weight_decay
        self.params = model.state_dict()
        self.opt_states = self._init_opt_states()
        self.step_count = 0

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
            states[name] = adamw_init(subset)
        return states

    @property
    def iteration(self):
        return self.step_count

    def _encode_scene(self, scene_input):
        """Pre-encode scene graph outside compiled function."""
        if isinstance(scene_input, dict):
            return self.model.scene_encoder(scene_input)
        return scene_input

    def train_step_tiled(self, noisy, gt, scene_input, tile_size=256):
        """Run compiled train step per tile, accumulate loss.

        Args:
            scene_input: scene_graph dict or pre-encoded (B, latent_dim) tensor.

        Returns:
            metrics dict compatible with OmenTrainer interface.
        """
        scene_latent = self._encode_scene(scene_input)
        B, H, W, C = (int(d) for d in noisy.shape)
        total_loss = 0.0
        n_tiles = 0
        t0 = time.time()

        for y in range(0, H, tile_size):
            for x in range(0, W, tile_size):
                y2, x2 = min(y + tile_size, H), min(x + tile_size, W)
                tile_noisy = noisy[:, y:y2, x:x2, :]
                tile_gt = gt[:, y:y2, x:x2, :]

                new_p, new_s, loss = compiled_train_step(
                    self.params,
                    tile_noisy,
                    tile_gt,
                    scene_latent,
                    self.opt_states,
                    self.weight_decay,
                )

                self.params = new_p
                self.opt_states = new_s
                total_loss += float(loss.to_numpy())
                n_tiles += 1

        self.step_count += 1
        avg_loss = total_loss / max(n_tiles, 1)
        elapsed = time.time() - t0
        logger.info(
            "Step %d: loss=%.2f tiles=%d time=%.1fs",
            self.step_count,
            avg_loss,
            n_tiles,
            elapsed,
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
        """Save params + optimizer states as .npz (safe serialization)."""
        arrays = {}
        meta = {"step": self.step_count, "components": []}
        for k, v in self.params.items():
            arrays[f"p/{k}"] = v.to_numpy()
        for name, state in self.opt_states.items():
            meta["components"].append(name)
            for k, v in state["m"].items():
                arrays[f"opt/{name}/m/{k}"] = v.to_numpy()
            for k, v in state["v"].items():
                arrays[f"opt/{name}/v/{k}"] = v.to_numpy()
            step_val = state["step"]
            arrays[f"opt/{name}/step"] = np.array(
                float(step_val) if hasattr(step_val, "to_numpy") else step_val
            )
        arrays["_meta"] = np.array(json.dumps(meta))
        np.savez_compressed(path, **arrays)
        logger.info("Checkpoint saved to %s (step %d)", path, self.step_count)

    def load_checkpoint(self, path):
        """Load params + optimizer states from .npz checkpoint."""
        data = np.load(path, allow_pickle=False)
        meta = json.loads(str(data["_meta"]))
        self.step_count = meta["step"]
        self.params = {}
        for k in data.files:
            if k.startswith("p/"):
                self.params[k[2:]] = nb.Tensor.from_dlpack(data[k])
        self.opt_states = {}
        for name in meta["components"]:
            m = {
                k[2 + len(name) + 4 :]: nb.Tensor.from_dlpack(data[k])
                for k in data.files
                if k.startswith(f"opt/{name}/m/")
            }
            v = {
                k[2 + len(name) + 4 :]: nb.Tensor.from_dlpack(data[k])
                for k in data.files
                if k.startswith(f"opt/{name}/v/")
            }
            step = float(data[f"opt/{name}/step"])
            self.opt_states[name] = {"m": m, "v": v, "step": step}
        logger.info("Checkpoint loaded from %s (step %d)", path, self.step_count)
