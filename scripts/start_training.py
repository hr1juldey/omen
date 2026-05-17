#!/usr/bin/env python3
"""CLI entry point for tiled GPU training on Mitsuba scenes."""

import argparse
import glob
import logging
import os
import time

from omen.config import OmenConfig
from omen.gpu_budget import can_fit_tiles, get_gpu_memory_info
from omen.model.jepa import OmenJEPA
from omen.scenes import SCENE_REGISTRY
from omen.training.online_gen import TrainingDataGenerator
from omen.training.trainer.core import OmenTrainer

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
logger = logging.getLogger("omen.train_cli")

ANIMATION_TYPES = ("camera_orbit", "mesh", "material", "light")
CKPT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), ".cache", "checkpoints"
)


def _find_latest_checkpoint(ckpt_dir):
    """Find newest step_*.omen checkpoint, return path or None."""
    files = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.omen.npz")))
    return files[-1] if files else None


def _system_ram_mb():
    """System RAM used in MB via /proc/meminfo."""
    try:
        with open("/proc/meminfo") as f:
            info = {}
            for line in f:
                parts = line.split()
                if len(parts) >= 2:
                    info[parts[0].rstrip(":")] = int(parts[1]) // 1024
        total = info.get("MemTotal", 0)
        avail = info.get("MemAvailable", 0)
        return total - avail
    except Exception:
        return 0


DEFAULT_RAM_LIMIT_GB = 24  # Leave ~8GB for OS on 32GB system


def _train_on_data(
    trainer,
    gen,
    noisy,
    gt,
    sg,
    tile_size,
    step_idx,
    steps_per_frame=1,
    checkpoint_every=0,
    ram_limit_gb=DEFAULT_RAM_LIMIT_GB,
    flush_graph=False,
):
    """Run tiled training step(s) and log results.

    With ``@nb.compile``, the compiled graph depends on tensor shapes only —
    not pixel values — so it's reusable across ALL frames at the same
    resolution.  Set ``flush_graph=True`` only when switching scenes (different
    model architecture) or if RAM is critically high.
    """
    # RAM guard — abort before OOM kills the process
    ram_mb = _system_ram_mb()
    if ram_mb > ram_limit_gb * 1024:
        raise MemoryError(
            f"RAM {ram_mb}MB exceeds {ram_limit_gb}GB limit — stopping before OOM"
        )

    all_metrics = []
    for sub_idx in range(steps_per_frame):
        # RAM guard inside loop — each tile can grow ~2GB if lazy tensors leak
        ram_pre = _system_ram_mb()
        if ram_pre > ram_limit_gb * 1024:
            raise MemoryError(
                f"RAM {ram_pre}MB exceeds {ram_limit_gb}GB limit at sub-step "
                f"{sub_idx + 1}/{steps_per_frame} — stopping before OOM"
            )

        t0 = time.perf_counter()
        metrics = trainer.train_step_tiled(noisy, gt, sg, tile_size=tile_size)
        dt = time.perf_counter() - t0
        ram_now = _system_ram_mb()
        if steps_per_frame > 1:
            logger.info(
                "Frame step %d/%d: loss=%.6f tiles=%d iter=%d (%.1fs) ram=%dMB",
                sub_idx + 1,
                steps_per_frame,
                metrics["total_loss"],
                metrics["num_tiles"],
                metrics["iteration"],
                dt,
                ram_now,
            )
        else:
            logger.info(
                "Step %d: loss=%.6f tiles=%d iter=%d (%.1fs) ram=%dMB",
                step_idx + 1,
                metrics["total_loss"],
                metrics["num_tiles"],
                metrics["iteration"],
                dt,
                ram_now,
            )
        all_metrics.append(metrics)
        # Periodic checkpoint
        if checkpoint_every > 0 and trainer.iteration % checkpoint_every == 0:
            trainer.save_checkpoint_rotating(CKPT_DIR)

    # Only flush graph cache when explicitly requested (scene transitions).
    # With @nb.compile, the graph is shape-dependent and reusable across frames.
    if flush_graph:
        trainer.flush_graph_cache()
    return all_metrics[-1]


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
    parser.add_argument(
        "--steps-per-frame",
        type=int,
        default=1,
        help="Optimizer steps per rendered frame",
    )
    parser.add_argument("--lr-warmup", type=int, default=0, help="Linear warmup steps")
    parser.add_argument(
        "--total-steps",
        type=int,
        default=1000,
        help="Total steps for cosine decay schedule",
    )
    parser.add_argument(
        "--scenes",
        choices=["single", "all"],
        default="single",
        help="Train on single scene or all scenes",
    )
    parser.add_argument(
        "--checkpoint-every",
        type=int,
        default=50,
        help="Save checkpoint every N optimizer steps (0=disabled)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from latest checkpoint",
    )
    parser.add_argument(
        "--log-file",
        default=None,
        help="Write logs to this file in addition to stdout",
    )
    parser.add_argument(
        "--ram-limit",
        type=int,
        default=DEFAULT_RAM_LIMIT_GB,
        help=f"Abort if system RAM exceeds this many GB (default: {DEFAULT_RAM_LIMIT_GB})",
    )
    parser.add_argument(
        "--gt-spp",
        type=int,
        default=512,
        help="Ground-truth samples per pixel (default: 512)",
    )
    parser.add_argument(
        "--noisy-spp",
        type=int,
        default=4,
        help="Noisy samples per pixel (default: 4)",
    )
    args = parser.parse_args()

    ram_limit_gb = args.ram_limit

    # Log file handler
    if args.log_file:
        log_dir = os.path.dirname(args.log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
        fh = logging.FileHandler(args.log_file)
        fh.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
        logging.getLogger().addHandler(fh)
        logger.info("Logging to %s", args.log_file)

    w, h = map(int, args.resolution.split("x"))

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
    trainer = OmenTrainer(
        model,
        config=config,
        warmup_steps=args.lr_warmup,
        total_steps=args.total_steps,
    )

    # Resume from checkpoint
    if args.resume:
        latest = _find_latest_checkpoint(CKPT_DIR)
        if latest:
            trainer.load_checkpoint(latest.replace(".npz", ""))
            logger.info("Resumed from %s (iter %d)", latest, trainer.iteration)
        else:
            logger.warning("--resume set but no checkpoint found in %s", CKPT_DIR)

    gen = TrainingDataGenerator(
        resolution=(w, h),
        gt_spp=args.gt_spp,
        noisy_spp=args.noisy_spp,
        save_images=args.save_images,
    )

    ckpt_kw = {"checkpoint_every": args.checkpoint_every, "ram_limit_gb": ram_limit_gb}
    losses = []

    if args.scenes == "all":
        _run_multi_scene(trainer, gen, args, losses, w, h, **ckpt_kw)
    elif args.animation:
        builder = SCENE_REGISTRY[args.scene]
        _run_animation(trainer, gen, builder, args, losses, w, h, **ckpt_kw)
    else:
        builder = SCENE_REGISTRY[args.scene]
        _run_static(trainer, gen, builder, args, losses, **ckpt_kw)

    # Final checkpoint
    trainer.save_checkpoint(os.path.join(CKPT_DIR, "latest.omen"))

    logger.info("Done. Loss: %.6f -> %.6f", losses[0], losses[-1])
    unique = len(set(f"{loss:.4f}" for loss in losses))
    if unique > 1:
        logger.info("Loss changed across %d unique values — model is learning.", unique)
    else:
        logger.warning("Loss unchanged — check gradient flow.")


def _run_static(
    trainer,
    gen,
    builder,
    args,
    losses,
    checkpoint_every=0,
    ram_limit_gb=DEFAULT_RAM_LIMIT_GB,
):
    """Static camera training — single or multi-camera."""
    logger.info(
        "Mode: static | Camera: %s | Steps: %d",
        args.camera,
        args.steps,
    )
    for step_idx in range(args.steps):
        for step_data in gen.train_step(builder, camera=args.camera):
            metrics = _train_on_data(
                trainer,
                gen,
                step_data["noisy_image"],
                step_data["gt_image"],
                step_data["scene_graph"],
                args.tile_size,
                step_idx,
                steps_per_frame=args.steps_per_frame,
                checkpoint_every=checkpoint_every,
                ram_limit_gb=ram_limit_gb,
            )
            losses.append(metrics["total_loss"])


def _run_multi_scene(
    trainer,
    gen,
    args,
    losses,
    w,
    h,
    checkpoint_every=0,
    ram_limit_gb=DEFAULT_RAM_LIMIT_GB,
):
    """Curriculum training — master one scene completely, then move to next.

    Per scene: static cameras -> all animation types.
    Graph cache flushed between scenes. Checkpoint saved on scene transitions.
    """
    from omen.scenes import get_animation_generator

    scene_names = sorted(SCENE_REGISTRY.keys())
    logger.info(
        "Mode: curriculum | Scenes: %s | Steps/scene: %d", scene_names, args.steps
    )

    for scene_idx, scene_name in enumerate(scene_names):
        logger.info("=" * 60)
        logger.info(
            "CURRICULUM: Scene %d/%d — %s", scene_idx + 1, len(scene_names), scene_name
        )
        logger.info("=" * 60)
        builder = SCENE_REGISTRY[scene_name]

        # Phase 1: static multi-camera
        logger.info("[%s] Phase 1: static cameras", scene_name)
        for step_idx in range(args.steps):
            for step_data in gen.train_step(builder, camera=args.camera):
                metrics = _train_on_data(
                    trainer,
                    gen,
                    step_data["noisy_image"],
                    step_data["gt_image"],
                    step_data["scene_graph"],
                    args.tile_size,
                    step_idx,
                    steps_per_frame=args.steps_per_frame,
                    checkpoint_every=checkpoint_every,
                    ram_limit_gb=ram_limit_gb,
                )
                losses.append(metrics["total_loss"])

        # Phase 2: all animation types
        animations = get_animation_generator(scene_name, base_resolution=(w, h))
        if animations is None:
            logger.info("[%s] No animations, skipping to next scene", scene_name)
            # Scene checkpoint before flush
            trainer.save_checkpoint_rotating(CKPT_DIR)
            trainer.flush_graph_cache()
            continue

        _, sg = builder(resolution=(w, h))
        for anim_type in ANIMATION_TYPES:
            if anim_type not in animations:
                continue
            logger.info("[%s] Phase 2: animation %s", scene_name, anim_type)
            frames = animations[anim_type]
            for step_data in gen.train_animation(frames, sg):
                metrics = _train_on_data(
                    trainer,
                    gen,
                    step_data["noisy_image"],
                    step_data["gt_image"],
                    sg,
                    args.tile_size,
                    0,
                    steps_per_frame=args.steps_per_frame,
                    checkpoint_every=checkpoint_every,
                    ram_limit_gb=ram_limit_gb,
                )
                losses.append(metrics["total_loss"])

        # Scene mastered — checkpoint then flush cache before next scene
        trainer.save_checkpoint_rotating(CKPT_DIR)
        trainer.flush_graph_cache()
        logger.info("[%s] Mastered. Checkpoint saved. Cache flushed.", scene_name)


def _run_animation(
    trainer,
    gen,
    builder,
    args,
    losses,
    w,
    h,
    checkpoint_every=0,
    ram_limit_gb=DEFAULT_RAM_LIMIT_GB,
):
    """Animation training — sequential frames with temporal variation."""
    from omen.scenes import get_animation_generator

    logger.info("Mode: animation | Scene: %s | Type: %s", args.scene, args.animation)

    _, sg = builder(resolution=(w, h))
    animations = get_animation_generator(args.scene, base_resolution=(w, h))
    if animations is None:
        logger.warning(
            "No animation generator for scene '%s', falling back to multi-camera",
            args.scene,
        )
        for step_data in gen.train_step(builder, camera="all"):
            metrics = _train_on_data(
                trainer,
                gen,
                step_data["noisy_image"],
                step_data["gt_image"],
                step_data["scene_graph"],
                args.tile_size,
                0,
                steps_per_frame=args.steps_per_frame,
                checkpoint_every=checkpoint_every,
                ram_limit_gb=ram_limit_gb,
            )
            losses.append(metrics["total_loss"])
        return
    if args.animation not in animations:
        avail = list(animations.keys())
        raise ValueError(
            f"No animation '{args.animation}' for scene '{args.scene}'. Available: {avail}"
        )
    frames = animations[args.animation]

    step_idx = 0
    for step_data in gen.train_animation(frames, sg):
        metrics = _train_on_data(
            trainer,
            gen,
            step_data["noisy_image"],
            step_data["gt_image"],
            sg,
            args.tile_size,
            step_idx,
            steps_per_frame=args.steps_per_frame,
            checkpoint_every=checkpoint_every,
            ram_limit_gb=ram_limit_gb,
        )
        losses.append(metrics["total_loss"])
        step_idx += 1
        if step_idx >= args.steps:
            break


if __name__ == "__main__":
    main()
