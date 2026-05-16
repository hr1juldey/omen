#!/usr/bin/env python3
"""CLI entry point for tiled GPU training on Mitsuba scenes."""

import argparse
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("omen.train_cli")

from omen.config import OmenConfig
from omen.gpu_budget import can_fit_tiles, get_gpu_memory_info
from omen.model.jepa import OmenJEPA
from omen.scenes import SCENE_REGISTRY
from omen.training.online_gen import TrainingDataGenerator
from omen.training.trainer.core import OmenTrainer

ANIMATION_TYPES = ("camera_orbit", "mesh", "material", "light")


def _train_on_data(trainer, gen, noisy, gt, sg, tile_size, step_idx):
    """Run one tiled training step and log results."""
    t0 = time.perf_counter()
    metrics = trainer.train_step_tiled(noisy, gt, sg, tile_size=tile_size)
    dt = time.perf_counter() - t0
    logger.info(
        "Step %d: loss=%.6f tiles=%d iter=%d (%.1fs)",
        step_idx + 1,
        metrics["total_loss"],
        metrics["num_tiles"],
        metrics["iteration"],
        dt,
    )
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Omen tiled GPU training")
    parser.add_argument("--scene", default="cornell", choices=sorted(SCENE_REGISTRY))
    parser.add_argument("--resolution", default="512x512", help="WxH")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--camera", default="default", help="Camera name or 'all'")
    parser.add_argument(
        "--animation",
        default=None,
        choices=ANIMATION_TYPES,
        help="Train on animation frames instead of static views",
    )
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    w, h = map(int, args.resolution.split("x"))
    builder = SCENE_REGISTRY[args.scene]

    # GPU check
    gpu_info = get_gpu_memory_info()
    logger.info("GPU: %s (%dMB free)", gpu_info["backend"], gpu_info["free_mb"])
    tile_check = can_fit_tiles(1, args.tile_size)
    logger.info(
        "Tile memory: %dMB per tile, sufficient=%s",
        tile_check["per_tile_mb"],
        tile_check["sufficient"],
    )

    # Setup
    config = OmenConfig.v1_dense()
    model = OmenJEPA(config=config)
    trainer = OmenTrainer(model, config=config)
    gen = TrainingDataGenerator(
        resolution=(w, h), gt_spp=256, noisy_spp=4, save_images=args.save_images
    )

    logger.info(
        "Scene: %s | Resolution: %dx%d | Tiles: %d",
        args.scene, w, h, args.tile_size,
    )

    losses = []

    # Animation mode: train on sequential frames
    if args.animation:
        _run_animation(trainer, gen, builder, args, losses, w, h)
    else:
        _run_static(trainer, gen, builder, args, losses)

    # Save checkpoint
    ckpt_path = os.path.expanduser("~/.cache/omen/checkpoints/latest.omen")
    trainer.save_checkpoint(ckpt_path)

    logger.info("Done. Loss: %.6f -> %.6f", losses[0], losses[-1])
    unique = len(set(f"{l:.4f}" for l in losses))
    if unique > 1:
        logger.info("Loss changed across %d unique values — model is learning.", unique)
    else:
        logger.warning("Loss unchanged — check gradient flow.")


def _run_static(trainer, gen, builder, args, losses):
    """Static camera training — single or multi-camera."""
    logger.info(
        "Mode: static | Camera: %s | Steps: %d", args.camera, args.steps,
    )
    for step_idx in range(args.steps):
        for step_data in gen.train_step(builder, camera=args.camera):
            _train_on_data(
                trainer, gen,
                step_data["noisy_image"], step_data["gt_image"],
                step_data["scene_graph"], args.tile_size, step_idx,
            )
            losses.append(trainer.iteration)


def _run_animation(trainer, gen, builder, args, losses, w, h):
    """Animation training — sequential frames with temporal variation."""
    from omen.scenes import cornell_animations

    logger.info("Mode: animation | Type: %s", args.animation)

    # Build base scene to get scene_graph (shared across all frames)
    _, sg = builder(resolution=(w, h))
    animations = cornell_animations(base_resolution=(w, h))
    frames = animations[args.animation]

    step_idx = 0
    for step_data in gen.train_animation(frames, sg):
        metrics = _train_on_data(
            trainer, gen,
            step_data["noisy_image"], step_data["gt_image"],
            sg, args.tile_size, step_idx,
        )
        losses.append(metrics["total_loss"])
        step_idx += 1
        if step_idx >= args.steps:
            break


if __name__ == "__main__":
    main()
