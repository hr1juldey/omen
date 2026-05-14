"""OmenTrainer core class with per-component optimizers."""

import logging
import os

import numpy as np

from omen.config import OmenConfig
from omen.training.trainer.gradient import _clip_grad_norm
from omen.training.trainer.optimizers import COMPONENT_LRS, create_optimizers

logger = logging.getLogger("omen.training.trainer")

try:
    import nabla as nb
    from nabla import nn
    NABLA_AVAILABLE = True
except (ImportError, RuntimeError):
    NABLA_AVAILABLE = False

DEFAULT_LR = 5e-5
DEFAULT_WEIGHT_DECAY = 1e-3
DEFAULT_GRADIENT_CLIP = 1.0


class OmenTrainer:
    """Nabla PyTorch-style trainer with per-component optimizers."""

    def __init__(
        self,
        model,
        config: OmenConfig = None,
        lr: float = DEFAULT_LR,
        weight_decay: float = DEFAULT_WEIGHT_DECAY,
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

        self._optimizers, self._component_params = create_optimizers(
            self.model, self.config, self.weight_decay
        )
        logger.info("Created %d optimizers", len(self._optimizers))

    def train_step(
        self,
        noisy,
        ground_truth,
        scene_graph,
        z_score: float = 0.0,
    ):
        """Single training step: JEPA latent loss + decoder noise loss."""
        self.model.train()

        # Encode noisy and GT images to latent space
        predicted_latent, _ = self.model.encode(scene_graph, noisy)
        target_latent, _ = self.model.encode(scene_graph, ground_truth)

        # Decoder predicts noise/residual from (latent, noisy_image)
        noisy_rgb = noisy[:, :, :, :3]
        predicted_noise = self.model.decode(predicted_latent, noisy_rgb)
        gt_residual = ground_truth[:, :, :, :3] - noisy_rgb

        total_loss, pred_loss, reg_loss = self.model.compute_loss(
            predicted_latent, target_latent, config=self.config,
            predicted_noise=predicted_noise, gt_residual=gt_residual,
        )

        total_loss.backward()
        _clip_grad_norm(self.model.parameters(), DEFAULT_GRADIENT_CLIP)

        applied_lrs = {}
        for name, optimizer in self._active_optimizers():
            lr = self._compute_lr(name, z_score)
            optimizer.lr = lr
            self.model = optimizer.step()
            applied_lrs[name] = lr

        self.model.zero_grad()
        self.iteration += 1

        return {
            "total_loss": float(total_loss.to_numpy().sum()),
            "pred_loss": float(pred_loss.to_numpy().sum()),
            "reg_loss": float(reg_loss.to_numpy().sum()),
            "iteration": self.iteration,
            "applied_lrs": applied_lrs,
            "z_score": z_score,
        }

    def _active_optimizers(self):
        """Return list of (name, optimizer) for enabled components."""
        return [(n, opt) for n, opt in self._optimizers.items()]

    def _compute_lr(self, component_name: str, z_score: float = 0.0) -> float:
        """Compute learning rate with surprise modulation."""
        base_lr = COMPONENT_LRS.get(component_name, DEFAULT_LR)
        if not self.config.training.surprise_lr_modulation or z_score <= 0:
            return base_lr
        scale = self.config.training.surprise_lr_scale
        return base_lr * (1.0 + scale * min(z_score, 5.0))

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
                config_dict = data["__config__"].item()
                self.config = OmenConfig.from_dict(config_dict)
            except Exception as e:
                logger.warning("Failed to load config, using default: %s", e)
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

        self._optimizers, self._component_params = create_optimizers(
            self.model, self.config, self.weight_decay
        )
        logger.info("Resumed from iteration %d", self.iteration)
