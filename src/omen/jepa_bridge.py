"""JEPABridge - Load Nabla model, transfer tensors via DLPack, run inference.

No C ABI, no ctypes, no shared libraries.
Uses Nabla Python API with DLPack zero-copy for GPU tensor transfer.
"""

import logging
import os
import numpy as np

logger = logging.getLogger("omen.jepa_bridge")

# Attempt Nabla import
try:
    import nabla as nb
    NABLA_AVAILABLE = True
except ImportError:
    nb = None
    NABLA_AVAILABLE = False

# Attempt Mitsuba/Dr.Jit import
try:
    import mitsuba as mi
    MITSUBA_AVAILABLE = True
except ImportError:
    mi = None
    MITSUBA_AVAILABLE = False


class JEPABridge:
    """Bridge between Mitsuba renders and Nabla JEPA model.

    Handles:
    - Model loading from Nabla checkpoints
    - DLPack zero-copy tensor transfer (GPU) or numpy fallback (CPU)
    - Inference: denoise, predict_confidence, merge_multires
    - Graceful degradation when Nabla/Mitsuba unavailable
    """

    def __init__(self, model_path: str | None = None):
        self.available = False
        self.model = None
        self._gpu_context = None
        self._is_gpu = False

        if not NABLA_AVAILABLE:
            logger.warning("Nabla not installed. Install with: pip install nabla-ml")
            return

        if not MITSUBA_AVAILABLE:
            logger.warning("Mitsuba not installed. Install with: pip install mitsuba")
            return

        # Detect GPU backend
        self._detect_gpu()

        # Load model
        try:
            from omen.model.jepa import OmenJEPA
            self.model = OmenJEPA()

            if model_path and os.path.exists(model_path):
                self._load_checkpoint(model_path)
            else:
                # Try default base model
                default_path = os.path.expanduser("~/.cache/omen/models/base_v0.omen")
                if os.path.exists(default_path):
                    self._load_checkpoint(default_path)
                else:
                    logger.info("No model checkpoint found. Using random weights.")
                    logger.info("Download base model or train with: omen train")

            self.model.eval()
            self.available = True
            logger.info("JEPABridge initialized (GPU=%s)", self._is_gpu)

        except Exception as e:
            logger.error("Failed to initialize JEPABridge: %s", e)
            self.available = False

    def _detect_gpu(self):
        """Detect if Mitsuba and Nabla share the same GPU."""
        self._is_gpu = False
        try:
            if mi is not None and '_ad_' in mi.variant():
                self._is_gpu = 'cuda' in mi.variant()
        except Exception:
            pass

    def _load_checkpoint(self, path: str):
        """Load model weights from Nabla checkpoint."""
        try:
            state = nb.load(path)
            self.model.load_state_dict(state)
            logger.info("Loaded model from %s", path)
        except Exception as e:
            logger.error("Failed to load checkpoint %s: %s", path, e)
            raise

    def to_nabla(self, array) -> "nb.Tensor":
        """Convert numpy array or Dr.Jit tensor to Nabla tensor.

        Uses DLPack zero-copy when both are on GPU.
        Falls back to numpy copy for CPU.
        """
        if not NABLA_AVAILABLE:
            raise RuntimeError("Nabla not available")

        # Try DLPack zero-copy first
        if hasattr(array, '__dlpack__'):
            try:
                return nb.Tensor.from_dlpack(array)
            except Exception:
                pass

        # Fallback: numpy -> Nabla
        if not isinstance(array, np.ndarray):
            array = np.array(array)
        tensor = nb.ndarray(array)

        # Move to GPU if available
        if self._is_gpu:
            try:
                tensor = tensor.cuda()
            except Exception:
                pass

        return tensor

    def to_numpy(self, tensor) -> np.ndarray:
        """Convert Nabla tensor back to numpy array."""
        if isinstance(tensor, np.ndarray):
            return tensor
        try:
            return tensor.to_numpy()
        except Exception:
            return np.array(tensor)

    def add_alpha(self, render: np.ndarray) -> np.ndarray:
        """Add alpha channel to RGB render -> RGBA."""
        if render.ndim == 3 and render.shape[-1] == 3:
            alpha = np.ones((*render.shape[:2], 1), dtype=render.dtype)
            return np.concatenate([render, alpha], axis=-1)
        return render

    def denoise(self, scene_graph: dict, rgba: np.ndarray, width: int, height: int) -> np.ndarray:
        """Run JEPA denoise inference.

        Args:
            scene_graph: dict of numpy arrays from scene_extractor
            rgba: (H, W, 4) numpy array, noisy render
            width: image width
            height: image height

        Returns:
            (H, W, 4) numpy array, clean render
        """
        if not self.available:
            return rgba

        try:
            nb_rgba = self.to_nabla(rgba.reshape(1, height, width, 4))
            nb_scene = {k: self.to_nabla(v) for k, v in scene_graph.items()}

            latent = self.model.encode(nb_scene, nb_rgba)
            clean = self.model.decode(latent, height, width)

            return self.to_numpy(clean).reshape(height, width, 4)
        except Exception as e:
            logger.error("JEPA denoise failed: %s", e)
            return rgba

    def predict_confidence(self, scene_graph: dict, rgba: np.ndarray, width: int, height: int):
        """Run confidence prediction alongside denoise.

        Returns:
            (clean_preview, confidence) tuple of numpy arrays
        """
        if not self.available:
            return rgba, np.ones((height, width, 1), dtype=np.float32)

        try:
            nb_rgba = self.to_nabla(rgba.reshape(1, height, width, 4))
            nb_scene = {k: self.to_nabla(v) for k, v in scene_graph.items()}

            latent = self.model.encode(nb_scene, nb_rgba)
            clean = self.model.decode(latent, height, width)
            confidence = self.model.predict_confidence(latent, height, width)

            clean_np = self.to_numpy(clean).reshape(height, width, 4)
            conf_np = self.to_numpy(confidence).reshape(height, width, 1)

            return clean_np, conf_np
        except Exception as e:
            logger.error("JEPA confidence prediction failed: %s", e)
            return rgba, np.ones((height, width, 1), dtype=np.float32)

    def merge_multires(self, scene_graph: dict, low_res: np.ndarray,
                       high_res: np.ndarray, scale: int = 4):
        """Merge low-res clean + high-res noisy via JEPA.

        Args:
            scene_graph: dict from scene_extractor
            low_res: (H//scale, W//scale, 4) clean low-res render
            high_res: (H, W, 4) noisy high-res render
            scale: downscale factor

        Returns:
            (H, W, 4) merged output
        """
        if not self.available:
            return high_res

        h, w = high_res.shape[0], high_res.shape[1]
        try:
            nb_low = self.to_nabla(low_res.reshape(1, h // scale, w // scale, 4))
            nb_high = self.to_nabla(high_res.reshape(1, h, w, 4))
            nb_scene = {k: self.to_nabla(v) for k, v in scene_graph.items()}

            merged = self.model.merge(nb_scene, nb_low, nb_high, scale)

            return self.to_numpy(merged).reshape(h, w, 4)
        except Exception as e:
            logger.error("JEPA multires merge failed: %s", e)
            return high_res
