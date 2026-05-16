"""OmenTrainer: functional value_and_grad + per-component AdamW."""

import logging
import math
import os

import numpy as np

from omen.config import OmenConfig
from omen.training.tile import extract_tiles
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
        self,
        model,
        config=None,
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        warmup_steps=0,
        total_steps=1000,
    ):
        if not NABLA_AVAILABLE:
            raise ImportError("Nabla required for training")

        self.model = model
        self.config = config or OmenConfig()
        self.weight_decay = weight_decay
        self.iteration = 0
        self._warmup_steps = warmup_steps
        self._total_steps = total_steps

        errors = self.config.validate()
        if errors:
            raise ValueError(f"Invalid config: {'; '.join(errors)}")

        from omen.gpu_budget import get_gpu_memory_info

        gpu_info = get_gpu_memory_info()
        self._gpu_available = gpu_info["backend"] != "none"

        self._components = create_functional_optimizers(
            self.model, self.config, self.weight_decay
        )

        self._compiled_step = None
        self._gpu = None
        if self._gpu_available:
            try:
                from max.driver import Accelerator

                self._gpu = Accelerator()
                self._transfer_model_to_gpu(self._gpu)
                logger.info("Model weights transferred to GPU")
            except Exception as e:
                logger.warning("GPU model transfer failed, using CPU: %s", e)
                self._gpu = None

        self._init_compiled_step()

    def _transfer_model_to_gpu(self, gpu):
        """Transfer all model weights to GPU so state_dict returns GPU tensors."""
        for name, param in self.model.state_dict().items():
            transferred = nb.ops.transfer_to(param, gpu)
            # Replace the parameter in the model's internal storage
            parts = name.split(".")
            obj = self.model
            for part in parts[:-1]:
                if part.isdigit():
                    obj = obj[int(part)]
                else:
                    obj = getattr(obj, part)
            leaf_name = parts[-1]
            if hasattr(obj, "weight") and leaf_name == "weight":
                obj.weight = transferred
            elif hasattr(obj, "bias") and leaf_name == "bias":
                obj.bias = transferred
            elif hasattr(obj, leaf_name):
                setattr(obj, leaf_name, transferred)

    def _init_compiled_step(self):
        """Create compiled forward+backward step for graph cache reuse.

        Following nabla's official examples (Transformer, CNN), the entire
        ``value_and_grad`` call is wrapped in ``@nb.compile`` so the graph
        is compiled ONCE and reused for every tile.  Same tile size means
        same input shapes means cache hit — no accumulation, no RAM bomb.

        Falls back to eager mode (with ``clear_all``) if compile fails.
        """
        try:
            model = self.model
            config = self.config

            @nb.compile
            def _compiled_fwd_bwd(params, noisy, gt, scene_latent):
                loss, grads = nb.value_and_grad(compute_training_loss, argnums=0)(
                    params, model, noisy, gt, scene_latent, config
                )
                return grads, loss

            self._compiled_step = _compiled_fwd_bwd
            logger.info("Compiled training step ready (graph will be cached)")
        except Exception as e:
            logger.warning("@nb.compile not available, using eager mode: %s", e)
            self._compiled_step = None

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
        sg_tensor = self._to_nabla_scene_graph(scene_graph)

        noisy_tiles = extract_tiles(noisy_full, tile_size)
        gt_tiles = extract_tiles(gt_full, tile_size)

        total_loss = 0.0
        for tile_noisy, tile_gt in zip(noisy_tiles, gt_tiles):
            total_loss += self._train_single_tile(
                tile_noisy, tile_gt, sg_tensor, z_score
            )
            # Compiled mode: graph cached once, reused per tile (same shapes).
            # Eager fallback: each call leaks a 6-8GB graph entry, must clear.
            if self._compiled_step is None:
                nb.GRAPH.clear_all()

        avg_loss = total_loss / len(noisy_tiles)
        self.iteration += 1
        return {
            "total_loss": avg_loss,
            "num_tiles": len(noisy_tiles),
            "iteration": self.iteration,
            "z_score": z_score,
        }

    def _to_nabla_scene_graph(self, scene_graph):
        """Convert scene_graph numpy arrays to batched nabla tensors.

        Adds leading batch dim (``[np.newaxis]``) to match what
        :class:`SceneGraphEncoder` expects — ``vertices (B, N, 3)``,
        ``params (B, M, D)``.
        """
        if not isinstance(scene_graph, dict):
            return scene_graph
        result = {}
        for section_key, section in scene_graph.items():
            converted = {}
            for k, v in section.items():
                if isinstance(v, np.ndarray):
                    converted[k] = nb.Tensor.from_dlpack(
                        v.astype(np.float32)[np.newaxis]
                    )
                else:
                    converted[k] = v
            result[section_key] = converted
        return result

    def _transfer_scene_latent_to_gpu(self, scene_latent):
        """Transfer scene_latent tensors to GPU."""
        if not isinstance(scene_latent, dict) or self._gpu is None:
            return scene_latent
        result = {}
        for section_key, section in scene_latent.items():
            if isinstance(section, dict):
                converted = {}
                for k, v in section.items():
                    if nb.is_tensor(v):
                        converted[k] = nb.ops.transfer_to(v, self._gpu)
                    else:
                        converted[k] = v
                result[section_key] = converted
            elif nb.is_tensor(section):
                result[section_key] = nb.ops.transfer_to(section, self._gpu)
            else:
                result[section_key] = section
        return result

    def _train_single_tile(self, tile_noisy, tile_gt, scene_latent, z_score):
        """Train on one tile — GPU when available, CPU fallback."""
        noisy_tensor = self._tile_to_tensor(tile_noisy.data)
        gt_tensor = self._tile_to_tensor(tile_gt.data)

        if self._gpu is not None:
            try:
                noisy_tensor = nb.ops.transfer_to(noisy_tensor, self._gpu)
                gt_tensor = nb.ops.transfer_to(gt_tensor, self._gpu)
                scene_latent = self._transfer_scene_latent_to_gpu(scene_latent)
                logger.debug("Tile on GPU via transfer_to")
            except Exception:
                logger.warning("GPU transfer failed, using CPU")

        return self._run_tile(noisy_tensor, gt_tensor, scene_latent, z_score)

    def _run_tile(self, noisy, gt, scene_latent, z_score):
        """Core training step for one tile.

        Uses ``@nb.compile`` when available so the forward+backward graph
        is compiled once and reused (no graph cache accumulation).
        Falls back to eager ``value_and_grad`` when compile is unavailable.
        """
        params = self.model.state_dict()

        if self._compiled_step is not None:
            grads, total_loss = self._compiled_step(params, noisy, gt, scene_latent)
        else:
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

    def _compute_scheduled_lr(self, component_name, z_score=0.0):
        """Compute LR with warmup + cosine decay + optional surprise modulation."""
        base_lr = self._components[component_name]["lr"]
        min_lr = base_lr * 0.01
        step = self.iteration

        if self._warmup_steps > 0 and step < self._warmup_steps:
            scheduled = base_lr * (step / self._warmup_steps)
        elif step >= self._total_steps:
            scheduled = min_lr
        else:
            progress = (step - self._warmup_steps) / max(
                self._total_steps - self._warmup_steps, 1
            )
            scheduled = min_lr + 0.5 * (base_lr - min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        if not self.config.training.surprise_lr_modulation or z_score <= 0:
            return scheduled
        scale = self.config.training.surprise_lr_scale
        return scheduled * (1.0 + scale * min(z_score, 5.0))

    def _apply_optimizer_updates(self, params, grads, z_score):
        """Per-component ``adamw_update`` with surprise-modulated LRs."""
        new_params = dict(params)
        for name, comp in self._components.items():
            names = comp["param_names"]
            subset_p = {n: params[n] for n in names}
            subset_g = {n: grads[n] for n in names}
            lr = self._compute_scheduled_lr(name, z_score)

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

    def flush_graph_cache(self):
        """Flush nabla's compiled graph cache — call between scene transitions."""
        import nabla as nb

        nb.GRAPH.clear_all()
        logger.info("Graph cache flushed (inter-scene)")

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

    def save_checkpoint_rotating(self, base_dir, keep=3):
        """Save checkpoint with rotation — keeps last N, deletes older."""
        import glob

        os.makedirs(base_dir, exist_ok=True)
        path = os.path.join(base_dir, f"step_{self.iteration}.omen")
        self.save_checkpoint(path)
        # Delete oldest checkpoints beyond keep limit
        existing = sorted(glob.glob(os.path.join(base_dir, "step_*.omen.npz")))
        while len(existing) > keep:
            os.remove(existing.pop(0))

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
