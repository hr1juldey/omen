"""Mode 4 - Animation pipeline with JEPA temporal prediction.

Frame 0 (anchor): 1spp -> encode -> denoise -> store latent
Frames 1..N: 1spp dirty -> encode -> ARPredictor(history, current, delta) -> decode
Surprise detection triggers re-anchor (4spp path trace).
Target: 10-50x speedup, SSIM > 0.90 for predicted frames.
"""

import logging

import numpy as np
from collections import deque

from omen.temporal import (
    detect_auto_surprise,
    detect_jump_cut,
    detect_surprise,
    flatten_delta,
)

logger = logging.getLogger("omen.modes.animation")

HISTORY_SIZE = 3
VALIDATION_INTERVAL = 5


class AnimationRenderer:
    """Animation renderer with JEPA temporal prediction."""

    def __init__(self, scene, bridge, history_size: int = HISTORY_SIZE):
        self.scene = scene
        self.bridge = bridge
        self.history_size = history_size
        self.history: deque = deque(maxlen=history_size)
        self.prev_graph: dict | None = None
        self.frame_count = 0
        self.surprise_mean = 0.0
        self.surprise_std = 1.0

    def render_frame(self, scene_delta: dict | None = None) -> np.ndarray:
        """Render a single animation frame. Returns (H, W, 4) clean RGBA."""
        import mitsuba as mi

        if not self.bridge.available:
            return self._render_fallback()

        from omen.scene_extractor import extract_scene_graph

        dirty = mi.render(self.scene, spp=1)
        dirty_np = np.array(dirty)
        height, width = dirty_np.shape[0], dirty_np.shape[1]
        alpha = np.ones((height, width, 1), dtype=dirty_np.dtype)
        rgba = np.concatenate([dirty_np, alpha], axis=-1)

        scene_graph = extract_scene_graph(self.scene)

        # Compute delta from previous frame if not provided
        if scene_delta is None and self.prev_graph is not None:
            from omen.temporal import compute_scene_delta
            scene_delta = compute_scene_delta(self.prev_graph, scene_graph)
        self.prev_graph = scene_graph

        # Jump cut -> re-anchor
        if scene_delta and detect_jump_cut(scene_delta):
            logger.info("Jump cut at frame %d, re-anchoring", self.frame_count)
            return self._render_anchor(scene_graph, rgba, width, height)

        # Auto-surprise (structural changes) -> re-anchor
        if scene_delta and detect_auto_surprise(scene_delta):
            return self._render_anchor(scene_graph, rgba, width, height)

        # Encode current frame
        current_latent = self._encode(scene_graph, rgba, height, width)

        if len(self.history) == 0:
            return self._render_anchor(scene_graph, rgba, width, height)

        # Predict using ARPredictor with scene delta
        delta_emb = self._encode_delta(scene_delta)
        predicted_latent = self.bridge.model.predict_temporal(
            list(self.history), current_latent, delta_emb
        )

        # Periodic validation (task 14.9)
        if self.frame_count % VALIDATION_INTERVAL == 0:
            self._validate(predicted_latent, current_latent)

        clean = self.bridge.model.decode(predicted_latent, height, width)
        clean_np = self.bridge.to_numpy(clean).reshape(height, width, 4)
        self.history.append(predicted_latent)
        self.frame_count += 1
        return clean_np

    def _render_anchor(self, scene_graph, rgba, width, height):
        """Render anchor frame (4spp + JEPA denoise)."""
        import mitsuba as mi

        anchor = mi.render(self.scene, spp=4)
        anchor_np = np.array(anchor)
        alpha = np.ones((height, width, 1), dtype=anchor_np.dtype)
        anchor_rgba = np.concatenate([anchor_np, alpha], axis=-1)

        latent = self._encode(scene_graph, anchor_rgba, height, width)
        self.history.clear()
        self.history.append(latent)
        self.frame_count += 1

        clean = self.bridge.model.decode(latent, height, width)
        return self.bridge.to_numpy(clean).reshape(height, width, 4)

    def _encode(self, scene_graph, rgba, height, width):
        """Encode frame into latent space."""
        return self.bridge.model.encode(
            {k: self.bridge.to_nabla(v) for k, v in scene_graph.items()},
            self.bridge.to_nabla(rgba.reshape(1, height, width, 4)),
        )

    def _validate(self, predicted_latent, actual_latent):
        """Periodic validation: check for surprise (task 14.9)."""
        pred_np = self.bridge.to_numpy(predicted_latent)
        actual_np = self.bridge.to_numpy(actual_latent)
        is_surprise, mse, z, self.surprise_mean, self.surprise_std = detect_surprise(
            pred_np, actual_np, self.surprise_mean, self.surprise_std,
        )
        if is_surprise:
            logger.warning("Surprise at frame %d (mse=%.4f, z=%.1f)",
                           self.frame_count, mse, z)
            self.history.clear()

    def _encode_delta(self, delta):
        """Flatten delta dict into tensor for SceneDeltaEncoder."""
        if delta is None:
            return self.bridge.to_nabla(np.zeros((1, 50), dtype=np.float32))
        vec = flatten_delta(delta).reshape(1, -1)
        return self.bridge.to_nabla(vec)

    def _render_fallback(self):
        """Fallback: render at 4spp without JEPA."""
        import mitsuba as mi
        result = mi.render(self.scene, spp=4)
        result_np = np.array(result)
        h, w = result_np.shape[0], result_np.shape[1]
        alpha = np.ones((h, w, 1), dtype=np.float32)
        return np.concatenate([result_np, alpha], axis=-1)
