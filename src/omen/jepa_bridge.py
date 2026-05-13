"""JEPABridge - Connects Mitsuba renders to Nabla JEPA model.

Bootstrap from scratch, train on every render, DLPack zero-copy.
"""

import logging
import os

import numpy as np

from omen.jepa_inference import JEPAInference
from omen.jepa_tensor import NABLA_AVAILABLE, to_nabla

logger = logging.getLogger("omen.jepa_bridge")

try:
    import mitsuba as mi

    MITSUBA_AVAILABLE = True
except ImportError:
    mi = None
    MITSUBA_AVAILABLE = False

CHECKPOINT_DIR = os.path.expanduser("~/.omen/checkpoints")
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "latest.omen")
CHECKPOINT_SAVE_INTERVAL = 10


class JEPABridge(JEPAInference):
    """Bridge between Mitsuba renders and Nabla JEPA model."""

    def __init__(self, model_path: str | None = None):
        self.available = False
        self.model = None
        self.trainer = None
        self.iteration = 0
        self._is_gpu = False

        if not NABLA_AVAILABLE or not MITSUBA_AVAILABLE:
            logger.warning("Nabla or Mitsuba not installed, AI pipeline disabled")
            return

        self._detect_gpu()
        self._init_model(model_path)

    def _detect_gpu(self):
        self._is_gpu = (
            mi is not None
            and hasattr(mi, "variant")
            and "_ad_" in mi.variant()
            and "cuda" in mi.variant()
        )

    def _init_model(self, model_path: str | None):
        try:
            from omen.model.jepa import OmenJEPA
            from omen.training.trainer import OmenTrainer
        except ImportError as exc:
            logger.error("Omen model modules not found: %s", exc)
            return

        self.model = OmenJEPA()
        ckpt = model_path or CHECKPOINT_PATH
        if os.path.exists(ckpt):
            self._load_checkpoint(ckpt)
        else:
            logger.info("No checkpoint found, bootstrapping fresh model")
            os.makedirs(CHECKPOINT_DIR, exist_ok=True)

        self.trainer = OmenTrainer(self.model)
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

    def train_step(self, noisy_rgb: np.ndarray, gt_rgb: np.ndarray,
                   scene_graph: dict) -> dict:
        if not self.available or self.trainer is None:
            return {}
        try:
            nb_noisy = to_nabla(noisy_rgb[np.newaxis], self._is_gpu)
            nb_gt = to_nabla(gt_rgb[np.newaxis], self._is_gpu)
            nb_scene = {k: to_nabla(v, self._is_gpu) for k, v in scene_graph.items()}
            losses = self.trainer.train_step(nb_noisy, nb_gt, nb_scene)
            self.iteration = self.trainer.iteration
            if self.iteration % CHECKPOINT_SAVE_INTERVAL == 0:
                self.save_checkpoint()
            return losses
        except Exception as exc:
            logger.error("train_step failed: %s", exc)
            return {}

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
