"""JEPA inference methods for denoising, confidence, and multires merge."""

import logging

import numpy as np

from omen.jepa_tensor import to_nabla, to_numpy

logger = logging.getLogger("omen.jepa_inference")


class JEPAInference:
    """Mixin providing JEPA inference methods. Mixed into JEPABridge."""

    def denoise(self, scene_graph: dict, rgba: np.ndarray,
                width: int, height: int) -> np.ndarray:
        if not self.available:
            return rgba
        try:
            nb_rgba = to_nabla(rgba.reshape(1, height, width, 4), self._is_gpu)
            nb_scene = {k: to_nabla(v, self._is_gpu) for k, v in scene_graph.items()}
            latent, _ = self.model.encode(nb_scene, nb_rgba)
            # Decoder predicts noise/residual; clean = noisy - noise
            nb_noisy = nb_rgba[:, :, :, :3]  # RGB only for decoder
            predicted_noise = self.model.decode(latent, nb_noisy)
            clean = nb_noisy - predicted_noise
            # Re-add alpha channel
            alpha = nb_rgba[:, :, :, 3:4]
            clean_rgba = nb.concatenate([clean, alpha], axis=-1)
            return to_numpy(clean_rgba).reshape(height, width, 4)
        except Exception as exc:
            logger.error("JEPA denoise failed: %s", exc)
            return rgba

    def predict_confidence(self, scene_graph: dict, rgba: np.ndarray,
                           width: int, height: int):
        if not self.available:
            return rgba, np.ones((height, width, 1), dtype=np.float32)
        try:
            nb_rgba = to_nabla(rgba.reshape(1, height, width, 4), self._is_gpu)
            nb_scene = {k: to_nabla(v, self._is_gpu) for k, v in scene_graph.items()}
            latent, _ = self.model.encode(nb_scene, nb_rgba)
            # Denoise: predict noise, subtract
            nb_noisy = nb_rgba[:, :, :, :3]
            predicted_noise = self.model.decode(latent, nb_noisy)
            clean = nb_noisy - predicted_noise
            alpha = nb_rgba[:, :, :, 3:4]
            clean_rgba = nb.concatenate([clean, alpha], axis=-1)
            conf = to_numpy(
                self.model.predict_confidence(latent, height, width)
            ).reshape(height, width, 1)
            return to_numpy(clean_rgba).reshape(height, width, 4), conf
        except Exception as exc:
            logger.error("JEPA confidence failed: %s", exc)
            return rgba, np.ones((height, width, 1), dtype=np.float32)

    def merge_multires(self, scene_graph: dict, low_res: np.ndarray,
                       high_res: np.ndarray, scale: int = 4) -> np.ndarray:
        if not self.available:
            return high_res
        h, w = high_res.shape[0], high_res.shape[1]
        try:
            nb_low = to_nabla(low_res.reshape(1, h // scale, w // scale, 4), self._is_gpu)
            nb_high = to_nabla(high_res.reshape(1, h, w, 4), self._is_gpu)
            nb_scene = {k: to_nabla(v, self._is_gpu) for k, v in scene_graph.items()}
            merged = self.model.merge(nb_scene, nb_low, nb_high, scale)
            return to_numpy(merged).reshape(h, w, 4)
        except Exception as exc:
            logger.error("JEPA multires merge failed: %s", exc)
            return high_res
