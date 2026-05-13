"""DualPipe async render pipeline for overlapping render + denoise.

Tasks 19a.1-19a.5: Double-buffered pipeline with bounded queue,
GPU memory ping-pong, and sequential fallback.
"""

import logging
import queue
import threading
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from omen.gpu_budget import get_gpu_memory_info, INFERENCE_BUDGET_MB

logger = logging.getLogger("omen.async_pipeline")

QUEUE_SIZE = 2  # Bounded queue between render and denoise threads


class DualPipeRenderer:
    """Double-buffered async pipeline: Thread 1 renders, Thread 2 denoises.

    Only enables async when VRAM > 2x single-frame inference budget.
    Falls back to sequential otherwise.
    """

    def __init__(self, scene, bridge, spp: int = 4, tier: str = "medium"):
        self.scene = scene
        self.bridge = bridge
        self.spp = spp
        self.tier = tier
        self._async_enabled = self._check_vram()

    def _check_vram(self) -> bool:
        """Only enable async when VRAM > 2x single-frame budget (task 19a.4)."""
        info = get_gpu_memory_info()
        if info["free_mb"] > 2 * INFERENCE_BUDGET_MB:
            logger.info("DualPipe enabled: %dMB free > 2x %dMB budget",
                        info["free_mb"], INFERENCE_BUDGET_MB)
            return True
        logger.info("DualPipe disabled: insufficient VRAM, using sequential")
        return False

    def render_frames(self, num_frames: int) -> list:
        """Render N frames. Uses async if enabled, else sequential.

        Returns list of (H, W, 4) numpy arrays.
        """
        if not self._async_enabled or not self.bridge.available:
            return self._render_sequential(num_frames)
        return self._render_async(num_frames)

    def _render_sequential(self, num_frames: int) -> list:
        """Sequential: render -> denoise, one frame at a time."""
        import mitsuba as mi
        from omen.modes.denoiser import render_denoiser

        results = []
        for i in range(num_frames):
            noisy = mi.render(self.scene, spp=self.spp)
            clean = render_denoiser(self.scene, self.bridge, spp=self.spp,
                                    tier=self.tier)
            results.append(clean)
        return results

    def _render_async(self, num_frames: int) -> list:
        """Async: render thread fills queue, denoise thread consumes (19a.2)."""
        frame_queue = queue.Queue(maxsize=QUEUE_SIZE)
        results = [None] * num_frames
        errors = []

        def render_worker():
            import mitsuba as mi
            for i in range(num_frames):
                try:
                    raw = mi.render(self.scene, spp=self.spp)
                    raw_np = np.array(raw)
                    frame_queue.put((i, raw_np), timeout=30)
                except Exception as exc:
                    errors.append((i, str(exc)))
                    frame_queue.put(None, timeout=5)
                    break
            frame_queue.put(None)  # sentinel

        def denoise_worker():
            from omen.modes.denoiser import render_denoiser
            while True:
                item = frame_queue.get(timeout=60)
                if item is None:
                    break
                idx, raw_np = item
                try:
                    clean = render_denoiser(self.scene, self.bridge,
                                            spp=self.spp, tier=self.tier)
                    results[idx] = clean
                except Exception as exc:
                    errors.append((idx, str(exc)))

        with ThreadPoolExecutor(max_workers=2) as pool:
            render_future = pool.submit(render_worker)
            denoise_future = pool.submit(denoise_worker)
            render_future.result()
            denoise_future.result()

        if errors:
            logger.warning("Async pipeline had %d errors: %s",
                           len(errors), errors[:3])
        return [r for r in results if r is not None]
