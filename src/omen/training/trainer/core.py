"""OmenTrainer: functional value_and_grad + per-component AdamW."""

import logging
import os

import numpy as np

from omen.config import OmenConfig
from omen.training.tile import Tile, extract_tiles
from omen.training.trainer.gradient import clip_grad_norm_pytree
from omen.training.trainer.loss import compute_training_loss
from omen.training.trainer.optimizers import create_functional_optimizers

logger = logging.getLogger("omen.training.trainer")

try:
    import nabla as nb
    from nabla.nn.optim import adamw_update

    NABLA_AVAILABLE = True
except (ImportError, RuntimeError):
    NABLA_AVAILABLE = False

DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_GRADIENT_CLIP = 1.0


class OmenTrainer:
    """Functional trainer using ``nb.value_and_grad`` and per-component AdamW.

    Uses custom conv2d_safe (Mojo im2col + nabla matmul) to avoid MAX
    compiler num_groups bug. All 139 params trainable.
    """

    def __init__(
        self, model, config=None, lr=DEFAULT_LR, weight_decay=DEFAULT_WEIGHT_DECAY
    ):
        if not NABLA_AVAILABLE:
            raise ImportError("Nabla required for training")

        self.model = model
        self.config = config or OmenConfig()
        self.weight_decay = weight_decay
        self.iteration = 0

        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        self._components = create_functional_optimizers(
            self.model, self.config, self.weight_decay
        )

    def train_step(self, noisy, ground_truth, scene_graph, z_score=0.0):
        """Single training step via ``nb.value_and_grad``."""
        self.model.train()

        params = self.model.state_dict()
        total_loss, grads = nb.value_and_grad(compute_training_loss, argnums=0)(
            params, self.model, noisy, ground_truth, scene_graph, self.config
        )

        loss_val = float(total_loss.to_numpy().sum())
        realized_grads = self._realize_grads(grads, params)
        clipped_grads = clip_grad_norm_pytree(realized_grads, DEFAULT_GRADIENT_CLIP)
        new_params = self._apply_optimizer_updates(params, clipped_grads, z_score)
        self.model.load_state_dict(new_params)

        self.iteration += 1
        return {
            "total_loss": loss_val,
            "iteration": self.iteration,
            "z_score": z_score,
        }

    def train_step_tiled(
        self, noisy_full, gt_full, scene_graph, tile_size=512, z_score=0.0
    ):
        """Tile-based training: encode scene once, train per-tile."""
        self.model.train()
        scene_latent = self._encode_scene_once(scene_graph)

        noisy_tiles = extract_tiles(noisy_full, tile_size)
        gt_tiles = extract_tiles(gt_full, tile_size)

        total_loss = 0.0
        for tile_noisy, tile_gt in zip(noisy_tiles, gt_tiles):
            total_loss += self._train_single_tile(
                tile_noisy, tile_gt, scene_latent, z_score
            )

        avg_loss = total_loss / len(noisy_tiles)
        self.iteration += 1
        return {
            "total_loss": avg_loss,
            "num_tiles": len(noisy_tiles),
            "iteration": self.iteration,
            "z_score": z_score,
        }

    def _encode_scene_once(self, scene_graph):
        """Encode scene graph into latent (once per full render)."""
        self.model.eval()
        scene_latent = self.model.scene_encoder(scene_graph)
        self.model.train()
        return scene_latent

    def _train_single_tile(self, tile_noisy, tile_gt, scene_latent, z_score):
        """Train on one tile with GPU OOM fallback."""
        noisy_tensor = self._tile_to_tensor(tile_noisy.data)
        gt_tensor = self._tile_to_tensor(tile_gt.data)

        try:
            return self._run_tile_gpu(
                noisy_tensor, gt_tensor, scene_latent, z_score
            )
        except (RuntimeError, Exception) as exc:
            err = str(exc).lower()
            if "out of memory" in err or "cuda" in err:
                logger.warning("GPU OOM, falling back to CPU: %s", exc)
                return self._run_tile_cpu(
                    noisy_tensor, gt_tensor, scene_latent, z_score
                )
            raise

    def _run_tile_gpu(self, noisy, gt, scene_latent, z_score):
        """Run tile on GPU, silently fall back if .cuda() unavailable."""
        try:
            noisy = noisy.cuda()
            gt = gt.cuda()
            if hasattr(scene_latent, "cuda"):
                scene_latent = scene_latent.cuda()
        except Exception:
            pass
        return self._run_tile(noisy, gt, scene_latent, z_score)

    def _run_tile_cpu(self, noisy, gt, scene_latent, z_score):
        """Run tile on CPU."""
        if hasattr(noisy, "cpu"):
            noisy = noisy.cpu()
        if hasattr(gt, "cpu"):
            gt = gt.cpu()
        return self._run_tile(noisy, gt, scene_latent, z_score)

    def _run_tile(self, noisy, gt, scene_latent, z_score):
        """Core training step for one tile."""
        params = self.model.state_dict()
        total_loss, grads = nb.value_and_grad(compute_training_loss, argnums=0)(
            params, self.model, noisy, gt, scene_latent, self.config
        )

        loss_val = float(total_loss.to_numpy().sum())
        realized_grads = self._realize_grads(grads, params)
        clipped_grads = clip_grad_norm_pytree(realized_grads, DEFAULT_GRADIENT_CLIP)
        new_params = self._apply_optimizer_updates(params, clipped_grads, z_score)
        self.model.load_state_dict(new_params)
        return loss_val

    def _tile_to_tensor(self, tile_data):
        """Convert numpy tile to nabla tensor (1, H, W, 4)."""
        arr = tile_data.astype(np.float32)
        if arr.ndim == 2:
            arr = arr[:, :, np.newaxis]
        h, w, c = arr.shape
        if c == 3:
            alpha = np.ones((h, w, 1), dtype=np.float32)
            arr = np.concatenate([arr, alpha], axis=-1)
        return nb.Tensor.from_dlpack(arr[np.newaxis])

    def _realize_grads(self, grads, params):
        """Realize lazy gradient tensors to numpy-backed.

        Single ``nb.realize_all`` compiles the backward graph once.
        Replaced per-tensor ``.to_numpy()`` which caused 139 separate
        compilations leaking ~30GB RAM.
        """
        lazy_grads = [g for g in grads.values() if not g.real]
        if lazy_grads:
            nb.realize_all(*lazy_grads)
        realized = {}
        for name, g in grads.items():
            if g.real:
                realized[name] = g
            else:
                realized[name] = nb.Tensor.from_dlpack(g.to_numpy())
        return realized

    def _compute_lr(self, component_name, z_score=0.0):
        """Compute learning rate with optional surprise modulation."""
        base_lr = self._components[component_name]["lr"]
        if not self.config.training.surprise_lr_modulation or z_score <= 0:
            return base_lr
        scale = self.config.training.surprise_lr_scale
        return base_lr * (1.0 + scale * min(z_score, 5.0))

    def _apply_optimizer_updates(self, params, grads, z_score):
        """Per-component ``adamw_update`` with surprise-modulated LRs."""
        new_params = dict(params)
        for name, comp in self._components.items():
            names = comp["param_names"]
            subset_p = {n: params[n] for n in names}
            subset_g = {n: grads[n] for n in names}
            lr = self._compute_lr(name, z_score)

            updated_p, new_state = adamw_update(
                subset_p,
                subset_g,
                comp["state"],
                lr=lr,
                weight_decay=comp["weight_decay"],
            )

            comp["state"] = new_state
            for k, v in updated_p.items():
                new_params[k] = v
        return new_params

    def save_checkpoint(self, path):
        """Save model state dict and config to disk."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        state_dict = self.model.state_dict()
        save_data = {
            "__iteration__": np.array(self.iteration),
            "__config__": np.array(self.config.to_dict(), dtype=object),
        }
        for key, tensor in state_dict.items():
            save_data[key] = tensor.to_numpy()
        np.savez_compressed(path, **save_data)
        logger.info("Checkpoint saved: %s (iter %d)", path, self.iteration)

    def load_checkpoint(self, path):
        """Load model state dict and config from disk."""
        npz_path = path if path.endswith(".npz") else path + ".npz"
        data = np.load(npz_path, allow_pickle=True)

        if "__config__" in data.files:
            try:
                self.config = OmenConfig.from_dict(data["__config__"].item())
            except Exception as exc:
                logger.warning("Failed to load config, using default: %s", exc)
                self.config = OmenConfig.v1_dense()
        else:
            self.config = OmenConfig.v1_dense()

        state_dict = {}
        for key in data.files:
            if key.startswith("__"):
                continue
            state_dict[key] = nb.Tensor.from_dlpack(data[key])
        self.model.load_state_dict(state_dict)
        self.iteration = int(data.get("__iteration__", 0))

        self._components = create_functional_optimizers(
            self.model, self.config, self.weight_decay
        )
        logger.info("Resumed from iteration %d", self.iteration)
