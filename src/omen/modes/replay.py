"""Stratified replay buffer for continual learning.

Replaces flat deque with per-scene sub-buffers.
Ensures diversity across scenes during replay.
"""

import random
from collections import deque
from typing import Any


class StratifiedReplayBuffer:
    """Stratified replay buffer for continual learning.

    Maintains per-scene sub-buffers. Sampling ensures diversity
    across scenes (sleep-like replay). Only replays from OTHER scenes.
    """

    def __init__(
        self,
        max_size: int = 500,
        replay_ratio: float = 0.5,
        max_per_scene: int | None = None,
    ):
        """Initialize stratified replay buffer.

        Args:
            max_size: Total capacity across all scenes
            replay_ratio: Fraction of replay vs new data (0.5 = 1:1)
            max_per_scene: Max items per scene (default: max_size / 10)
        """
        self.max_size = max_size
        self.replay_ratio = replay_ratio
        self.max_per_scene = max_per_scene or (max_size // 10)
        self._buffers: dict[str, deque] = {}

    def add(self, scene_hash: str, noisy: Any, gt: Any) -> None:
        """Add a training sample to the appropriate scene buffer.

        Args:
            scene_hash: Unique identifier for the scene
            noisy: Noisy render data
            gt: Ground truth clean render data
        """
        if scene_hash not in self._buffers:
            self._buffers[scene_hash] = deque(maxlen=self.max_per_scene)

        self._buffers[scene_hash].append((noisy, gt))
        self._trim()

    def sample(self, current_scene: str, count: int) -> list[tuple[Any, Any]]:
        """Sample from OTHER scenes for replay interleaving.

        Args:
            current_scene: Hash of scene currently being trained on (excluded)
            count: Number of samples to return

        Returns:
            List of (noisy, gt) tuples from other scenes
        """
        other_scenes = [h for h in self._buffers if h != current_scene]
        if not other_scenes:
            return []

        samples = []
        for _ in range(count):
            scene = random.choice(other_scenes)
            buf = self._buffers[scene]
            if buf:
                samples.append(random.choice(list(buf)))
        return samples

    def replay_ratio_count(self, new_count: int) -> int:
        """How many replay samples for N new samples (1:1 ratio).

        For replay_ratio=0.5: 50% new, 50% replay → 1 new = 1 replay
        Formula: replay = new * ratio / (1 - ratio)
        """
        # Avoid division by zero, handle edge case where ratio >= 1
        if self.replay_ratio >= 1.0:
            return 0
        return int(new_count * self.replay_ratio / (1.0 - self.replay_ratio))

    def _trim(self) -> None:
        """Trim buffers to maintain total max_size."""
        total = sum(len(buf) for buf in self._buffers.values())
        if total > self.max_size:
            # Trim from largest buffer
            largest = max(self._buffers.items(), key=lambda x: len(x[1]))
            while total > self.max_size and self._buffers[largest[0]]:
                self._buffers[largest[0]].popleft()
                total -= 1

    def scene_count(self) -> int:
        """Return number of scenes in buffer."""
        return len(self._buffers)

    def total_count(self) -> int:
        """Return total number of items across all scenes."""
        return sum(len(buf) for buf in self._buffers.values())

    def clear(self) -> None:
        """Clear all buffers."""
        self._buffers.clear()
