#!/usr/bin/env python3
"""CLI entry point for tiled GPU training on Mitsuba scenes."""

import argparse
import logging
import os
import time

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("omen.train_cli")

# Don't set NABLA_DEFAULT_DEVICE — nabla defaults to CPU safely.
# GPU activated via .cuda() when _has_accelerator() returns True.

from omen.config import OmenConfig
from omen.gpu_budget import can_fit_tiles, get_gpu_memory_info
from omen.model.jepa import OmenJEPA
from omen.scenes import SCENE_REGISTRY
from omen.training.online_gen import TrainingDataGenerator
from omen.training.trainer.core import OmenTrainer


def main():
    parser = argparse.ArgumentParser(description="Omen tiled GPU training")
    parser.add_argument("--scene", default="cornell", choices=sorted(SCENE_REGISTRY))
    parser.add_argument("--resolution", default="512x512", help="WxH")
    parser.add_argument("--tile-size", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--device", default="gpu", choices=["gpu", "cpu"])
    parser.add_argument("--save-images", action="store_true")
    args = parser.parse_args()

    # Override device
    if args.device == "cpu":
        os.environ["NABLA_DEFAULT_DEVICE"] = "cpu"

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

    logger.info("Scene: %s | Resolution: %dx%d | Tiles: %d", args.scene, w, h, args.tile_size)
    logger.info("Running %d training steps...", args.steps)

    losses = []
    for step_idx in range(args.steps):
        t0 = time.perf_counter()
        for step_data in gen.train_step(builder):
            noisy = step_data["noisy_image"]
            gt = step_data["gt_image"]
            sg = step_data["scene_graph"]

            metrics = trainer.train_step_tiled(
                noisy, gt, sg, tile_size=args.tile_size
            )
            losses.append(metrics["total_loss"])
            dt = time.perf_counter() - t0
            logger.info(
                "Step %d: loss=%.6f tiles=%d iter=%d (%.1fs)",
                step_idx + 1,
                metrics["total_loss"],
                metrics["num_tiles"],
                metrics["iteration"],
                dt,
            )
            t0 = time.perf_counter()

    # Save checkpoint
    ckpt_path = os.path.expanduser("~/.cache/omen/checkpoints/latest.omen")
    trainer.save_checkpoint(ckpt_path)

    logger.info("Done. Loss: %.6f -> %.6f", losses[0], losses[-1])
    if len(set(f"{l:.4f}" for l in losses)) > 1:
        logger.info("Loss changed across steps — model is learning.")
    else:
        logger.warning("Loss unchanged — check gradient flow.")


if __name__ == "__main__":
    main()
