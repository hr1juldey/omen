"""JEPABridge - Connects Mitsuba renders to Nabla JEPA model.

Bootstrap from scratch, train on every render, DLPack zero-copy.
"""

import logging
import os

import numpy as np

from omen.jepa_inference import JEPAInference
from omen.jepa_tensor import CHECKPOINT_DIR, CHECKPOINT_PATH, NABLA_AVAILABLE, to_nabla

logger = logging.getLogger("omen.jepa_bridge")

try:
    import mitsuba as mi

    MITSUBA_AVAILABLE = True
except ImportError:
    mi = None
    MITSUBA_AVAILABLE = False

CHECKPOINT_SAVE_INTERVAL = 10


class JEPABridge(JEPAInference):
    """Bridge between Mitsuba renders and Nabla JEPA model."""

    def __init__(self, model_path: str | None = None):
        self.available = False
        self.model = None
        self.trainer = None
        self.iteration = 0
        self._is_gpu = False
        self._history = []  # Temporal history buffer (only used when AR enabled)

        if not NABLA_AVAILABLE or not MITSUBA_AVAILABLE:
            logger.warning("Nabla or Mitsuba not installed, AI pipeline disabled")
            return

        self._detect_gpu()
        try:
            self._init_model(model_path)
        except Exception as exc:
            logger.error("Model init failed: %s", exc)
            self.available = False

    def _detect_gpu(self):
        variant = mi.variant() if mi is not None and hasattr(mi, "variant") else None
        self._is_gpu = variant is not None and "cuda" in variant

    def _init_model(self, model_path: str | None):
        try:
            from omen.model.jepa import OmenJEPA
            from omen.training.trainer import OmenTrainer
            from omen.config import OmenConfig
        except ImportError as exc:
            logger.error("Omen model modules not found: %s", exc)
            return

        # Use V1 dense config by default (can be overridden later)
        config = OmenConfig.v1_dense()

        self.model = OmenJEPA(config=config)
        ckpt = model_path or CHECKPOINT_PATH
        if os.path.exists(ckpt):
            self._load_checkpoint(ckpt)
        else:
            logger.info("No checkpoint found, bootstrapping fresh model")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self.trainer = OmenTrainer(self.model, config=config)
        self.model.eval()
        self.available = True
        logger.info("JEPABridge ready (GPU=%s, iter=%d)", self._is_gpu, self.iteration)

    def _load_checkpoint(self, path: str):
        try:
            from omen.training.trainer import OmenTrainer

            tmp = OmenTrainer.__new__(OmenTrainer)
            tmp.model = self.model
            tmp.iteration = 0
            tmp.load_checkpoint(path)
            self.iteration = tmp.iteration
            logger.info("Loaded checkpoint (iter=%d)", self.iteration)
        except Exception as exc:
            logger.warning("Checkpoint load failed (%s), using random weights", exc)

    def train_step(
        self,
        noisy_rgb: np.ndarray,
        gt_rgb: np.ndarray,
        scene_graph: dict,
        z_score: float = 0.0,
    ) -> dict:
        """Run training step with optional z_score for surprise lr modulation."""
        if not self.available or self.trainer is None:
            return {}
        try:
            nb_noisy = to_nabla(noisy_rgb[np.newaxis], self._is_gpu)
            nb_gt = to_nabla(gt_rgb[np.newaxis], self._is_gpu)
            nb_scene = {k: to_nabla(v, self._is_gpu) for k, v in scene_graph.items()}
            losses = self.trainer.train_step(nb_noisy, nb_gt, nb_scene, z_score=z_score)
            self.iteration = self.trainer.iteration

            # Update history buffer only when ARPredictor is enabled
            if self.model.config.components.ar_predictor:
                self._update_history(nb_scene)

            if self.iteration % CHECKPOINT_SAVE_INTERVAL == 0:
                self.save_checkpoint()
            return losses
        except Exception as exc:
            logger.error("train_step failed: %s", exc)
            return {}

    def _update_history(self, scene_graph: dict) -> None:
        """Update history buffer for ARPredictor (only when AR enabled)."""
        # Encode current scene to latent and add to history
        try:
            from omen.jepa_inference import to_nabla
            import numpy as np

            # Create a dummy render for encoding (scene graph only)
            dummy_render = np.zeros((1, 64, 64, 4), dtype=np.float32)
            latent, _ = self.model.encode(scene_graph, to_nabla(dummy_render, self._is_gpu))

            self._history.append(latent)
            # Keep history at reasonable size (e.g., last 10 frames)
            if len(self._history) > 10:
                self._history.pop(0)
        except Exception as e:
            logger.debug("History update failed: %s", e)

    def clear_history(self) -> None:
        """Clear the temporal history buffer."""
        self._history.clear()

    def get_history(self) -> list:
        """Get current temporal history."""
        return self._history.copy()

    def save_checkpoint(self, scene_hash: str | None = None) -> None:
        if not self.available or self.trainer is None:
            return
        path = os.path.join(CHECKPOINT_DIR, f"{scene_hash}.omen") if scene_hash else CHECKPOINT_PATH
        self.trainer.save_checkpoint(path)

    def init_lora(self, scene_hash: str, rank: int = 8) -> None:
        if not self.available or self.trainer is None:
            return
        self.trainer.init_lora(rank=rank)
        logger.info("LoRA adapters initialized for scene %s", scene_hash)
