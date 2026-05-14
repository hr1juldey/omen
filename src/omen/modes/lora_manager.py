"""Per-scene LoRA fine-tuning with stratified replay buffer.

Compatibility shim: uses StratifiedReplayBuffer internally while
exporting the same interface for backward compatibility.
"""

import hashlib
import logging
import os
import random

from omen.jepa_tensor import CHECKPOINT_DIR
from omen.modes.replay import StratifiedReplayBuffer

logger = logging.getLogger("omen.modes.lora_manager")

LORA_TRIGGER_COUNT = 3
LORA_FINETUNE_ITERS = 50
REPLAY_BUFFER_SIZE = 500
REPLAY_SAMPLES_PER_STEP = 3
REPLAY_RATIO = 0.5

_render_counts: dict[str, int] = {}
_training_cache: dict[str, list] = {}
_replay_buffer = StratifiedReplayBuffer(
    max_size=REPLAY_BUFFER_SIZE,
    replay_ratio=REPLAY_RATIO,
)
_consolidated: set[str] = set()


def _scene_hash(scene_graph: dict) -> str:
    """Hash scene topology from scene graph keys for render tracking."""
    keys = sorted(scene_graph.keys())
    raw = ",".join(keys)
    return hashlib.md5(raw.encode()).hexdigest()[:12]


def _add_to_replay(shash: str, noisy, gt):
    """Add training pair to stratified replay buffer."""
    _replay_buffer.add(shash, noisy, gt)


def _sample_replay(shash: str, count: int) -> list:
    """Sample diverse pairs from OTHER scenes for replay interleaving."""
    samples = _replay_buffer.sample(shash, count)
    # Return in format expected by legacy code: (shash, noisy, gt)
    return [(shash, n, g) for n, g in samples]


def train_on_scene(bridge, scene, scene_graph: dict):
    """Train on scene pair with stratified replay buffer and consolidation."""
    from omen.training.data_gen import generate_denoiser_pair

    noisy, gt = generate_denoiser_pair(scene, spp_noisy=4, spp_gt=256)
    shash = _scene_hash(scene_graph)
    _render_counts[shash] = _render_counts.get(shash, 0) + 1

    # Consolidated scenes: adapter frozen, no further training
    if shash in _consolidated:
        return

    # Load existing LoRA adapter for this scene
    lora_path = os.path.join(CHECKPOINT_DIR, f"{shash}.omen")
    if os.path.exists(lora_path) and bridge.available:
        bridge.init_lora(shash)

    # Train on current pair + replay from diverse scenes (sleep-like interleaving)
    bridge.train_step(noisy, gt, scene_graph)
    for _, rep_noisy, rep_gt in _sample_replay(shash, REPLAY_SAMPLES_PER_STEP):
        bridge.train_step(rep_noisy, rep_gt, scene_graph)

    # Add to replay buffer and per-scene cache
    _add_to_replay(shash, noisy, gt)
    _training_cache.setdefault(shash, []).append((noisy, gt))
    if len(_training_cache[shash]) > 5:
        _training_cache[shash] = _training_cache[shash][-5:]

    # Trigger LoRA fine-tuning after threshold renders
    if _render_counts[shash] == LORA_TRIGGER_COUNT and bridge.available:
        _run_lora_finetune(bridge, shash, scene_graph)


def _run_lora_finetune(bridge, shash: str, scene_graph: dict):
    """Run LoRA fine-tuning with replay, then freeze (consolidate) adapter."""
    bridge.init_lora(shash)
    pairs = _training_cache.get(shash, [])
    replay = _sample_replay(shash, REPLAY_SAMPLES_PER_STEP)

    for _ in range(LORA_FINETUNE_ITERS):
        for pn, pg in pairs:
            bridge.train_step(pn, pg, scene_graph)
        for _, rn, rg in replay:
            bridge.train_step(rn, rg, scene_graph)

    bridge.save_checkpoint(scene_hash=shash)
    _consolidated.add(shash)
    logger.info("LoRA consolidated for scene %s (%d iters)", shash, LORA_FINETUNE_ITERS)
