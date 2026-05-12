"""Mode 4 - Animation pipeline with JEPA temporal prediction.

Pipeline:
Frame 0 (anchor): 1spp render -> encode -> denoise -> store latent
Frames 1..N: 1spp dirty -> encode -> ARPredictor(history, current, delta) -> decode
Surprise detection triggers re-anchor (4spp path trace)
Target: 10-50x speedup, SSIM > 0.90 for predicted frames
"""

import logging
import numpy as np
from collections import deque

logger = logging.getLogger("omen.modes.animation")

HISTORY_SIZE = 3
SURPRISE_THRESHOLD = 2.0
JUMP_CUT_TRANSLATION = 1.0
JUMP_CUT_ROTATION = 0.785  # ~45 degrees
VALIDATION_INTERVAL = 5


class AnimationRenderer:
    """Animation renderer with JEPA temporal prediction."""

    def __init__(self, scene, bridge, history_size: int = HISTORY_SIZE):
        self.scene = scene
        self.bridge = bridge
        self.history_size = history_size
        self.history = deque(maxlen=history_size)
        self.frame_count = 0
        self.surprise_mean = 0.0
        self.surprise_std = 1.0

    def render_frame(self, scene_delta=None):
        """Render a single animation frame.

        Args:
            scene_delta: dict of delta values for this frame, or None

        Returns:
            numpy array (H, W, 4) clean RGBA
        """
        import mitsuba as mi

        if not self.bridge.available:
            return self._render_fallback()

        from omen.scene_extractor import extract_scene_graph

        # Always render 1spp dirty for geometry/occlusion
        dirty = mi.render(self.scene, spp=1)
        dirty_np = np.array(dirty)
        height, width = dirty_np.shape[0], dirty_np.shape[1]

        alpha = np.ones((height, width, 1), dtype=dirty_np.dtype)
        rgba = np.concatenate([dirty_np, alpha], axis=-1)

        scene_graph = extract_scene_graph(self.scene)

        # Check for jump cut
        if self._is_jump_cut(scene_delta):
            logger.info("Jump cut detected at frame %d, re-anchoring", self.frame_count)
            return self._render_anchor(scene_graph, rgba, width, height)

        # Encode current frame
        current_latent = self.bridge.model.encode(
            {k: self.bridge.to_nabla(v) for k, v in scene_graph.items()},
            self.bridge.to_nabla(rgba.reshape(1, height, width, 4))
        )

        if len(self.history) == 0:
            # First frame - anchor
            return self._render_anchor(scene_graph, rgba, width, height)

        # Encode scene delta
        if scene_delta is not None:
            delta_tensor = self._encode_delta(scene_delta)
            delta_emb = self.bridge.model.delta_encoder(delta_tensor)
        else:
            delta_emb = self.bridge.to_nabla(np.zeros((1, 192), dtype=np.float32))

        # Predict using ARPredictor
        predicted_latent = self.bridge.model.predict_temporal(
            list(self.history), current_latent, delta_emb
        )

        # Periodic validation
        if self.frame_count % VALIDATION_INTERVAL == 0:
            self._validate_prediction(predicted_latent, current_latent)

        # Decode prediction
        clean = self.bridge.model.decode(predicted_latent, height, width)
        clean_np = self.bridge.to_numpy(clean).reshape(height, width, 4)

        # Store in history
        self.history.append(predicted_latent)
        self.frame_count += 1

        return clean_np

    def _render_anchor(self, scene_graph, rgba, width, height):
        """Render an anchor frame (4spp + JEPA denoise)."""
        import mitsuba as mi

        # Re-render at 4spp for anchor
        anchor = mi.render(self.scene, spp=4)
        anchor_np = np.array(anchor)
        alpha = np.ones((height, width, 1), dtype=anchor_np.dtype)
        anchor_rgba = np.concatenate([anchor_np, alpha], axis=-1)

        latent = self.bridge.model.encode(
            {k: self.bridge.to_nabla(v) for k, v in scene_graph.items()},
            self.bridge.to_nabla(anchor_rgba.reshape(1, height, width, 4))
        )
        self.history.clear()
        self.history.append(latent)
        self.frame_count += 1

        clean = self.bridge.model.decode(latent, height, width)
        return self.bridge.to_numpy(clean).reshape(height, width, 4)

    def _render_fallback(self):
        """Fallback: render at 4spp without JEPA."""
        import mitsuba as mi
        result = mi.render(self.scene, spp=4)
        result_np = np.array(result)
        h, w = result_np.shape[0], result_np.shape[1]
        alpha = np.ones((h, w, 1), dtype=np.float32)
        return np.concatenate([result_np, alpha], axis=-1)

    def _is_jump_cut(self, delta):
        """Detect jump cut from scene delta."""
        if delta is None:
            return False
        camera_delta = delta.get("camera_translation", 0.0)
        rotation_delta = delta.get("camera_rotation", 0.0)
        return camera_delta > JUMP_CUT_TRANSLATION or rotation_delta > JUMP_CUT_ROTATION

    def _validate_prediction(self, predicted_latent, actual_latent):
        """Check for surprise by comparing predicted vs actual latent."""
        diff = ((predicted_latent - actual_latent) ** 2).mean()
        diff_np = float(self.bridge.to_numpy(diff).sum())

        if self.surprise_std > 0:
            z_score = (diff_np - self.surprise_mean) / self.surprise_std
        else:
            z_score = 0.0

        # Update running stats
        alpha = 0.1
        self.surprise_mean = (1 - alpha) * self.surprise_mean + alpha * diff_np

        if z_score > SURPRISE_THRESHOLD:
            logger.warning(
                "Surprise detected at frame %d (score: %.4f, z: %.1f)",
                self.frame_count, diff_np, z_score
            )
            # Re-anchor on next frame
            self.history.clear()

    def _encode_delta(self, delta_dict):
        """Flatten delta dict into tensor for SceneDeltaEncoder."""
        values = []
        for key in sorted(delta_dict.keys()):
            val = delta_dict[key]
            if isinstance(val, (list, np.ndarray)):
                values.extend(np.array(val).flatten())
            else:
                values.append(float(val))

        # Pad to fixed size
        while len(values) < 50:
            values.append(0.0)

        return self.bridge.to_nabla(np.array(values[:50], dtype=np.float32).reshape(1, 50))
