"""Test pattern generation for Omen render engine validation.

Also serves as the Blender demo scene camera animation module
for generating training data from Blender demo files.
"""

import math
from typing import List


def generate_gradient(width: int, height: int) -> List[List[float]]:
    """Generate horizontal red-to-blue gradient.

    Args:
        width: Image width in pixels
        height: Image height in pixels

    Returns:
        Flat list of [r, g, b, a] values for each pixel (row-major order)
    """
    pixels = []
    for y in range(height):
        for x in range(width):
            t = x / max(1, width - 1)
            r = 1.0 - t
            g = 0.0
            b = t
            a = 1.0
            pixels.append([r, g, b, a])
    return pixels


def generate_checkerboard(width: int, height: int, block_size: int = 32) -> List[List[float]]:
    """Generate checkerboard pattern for denoiser validation.

    Useful for testing tile-based MoE routing: alternating blocks
    create clear material boundaries at tile edges.
    """
    pixels = []
    for y in range(height):
        for x in range(width):
            bx = (x // block_size) % 2
            by = (y // block_size) % 2
            val = 1.0 if (bx ^ by) else 0.2
            pixels.append([val, val, val, 1.0])
    return pixels


# ---------------------------------------------------------------------------
# Camera animation patterns for Blender demo file training data generation
# ---------------------------------------------------------------------------

def camera_orbit(frame_idx: int, total_frames: int = 60,
                 radius: float = 3.0, elevation: float = 0.6,
                 target: tuple = (0, 0, 0)):
    """Compute camera position for orbital animation.

    Args:
        frame_idx: Current frame number
        total_frames: Total frames in orbit
        radius: Distance from target
        elevation: Height angle in radians (0=horizon, pi/2=top)
        target: Look-at target (x, y, z)

    Returns:
        Tuple of (origin, target, up) for Mitsuba look_at transform.
    """
    theta = 2 * math.pi * frame_idx / total_frames
    x = radius * math.cos(elevation) * math.cos(theta) + target[0]
    y = radius * math.sin(elevation) + target[1]
    z = radius * math.cos(elevation) * math.sin(theta) + target[2]
    return ([x, y, z], list(target), [0, 1, 0])


def camera_dolly(frame_idx: int, total_frames: int = 60,
                 start_dist: float = 5.0, end_dist: float = 1.5,
                 direction: tuple = (0, 0, -1)):
    """Compute camera position for dolly (push-in/pull-out) animation.

    Args:
        frame_idx: Current frame number
        total_frames: Total frames
        start_dist: Starting distance
        end_dist: Ending distance
        direction: Camera movement direction (normalized)

    Returns:
        Tuple of (origin, target, up).
    """
    t = frame_idx / max(1, total_frames - 1)
    dist = start_dist + (end_dist - start_dist) * t
    origin = [dist * d for d in direction]
    return (origin, [0, 0, 0], [0, 1, 0])


def camera_pan(frame_idx: int, total_frames: int = 60,
               start_angle: float = -0.5, end_angle: float = 0.5,
               radius: float = 3.0, elevation: float = 0.4):
    """Compute camera position for pan animation.

    Args:
        frame_idx: Current frame number
        total_frames: Total frames
        start_angle: Start horizontal angle (radians)
        end_angle: End horizontal angle (radians)
        radius: Distance from center
        elevation: Fixed height angle (radians)

    Returns:
        Tuple of (origin, target, up).
    """
    t = frame_idx / max(1, total_frames - 1)
    theta = start_angle + (end_angle - start_angle) * t
    x = radius * math.cos(elevation) * math.cos(theta)
    y = radius * math.sin(elevation)
    z = radius * math.cos(elevation) * math.sin(theta)
    return ([x, y, z], [0, 0, 0], [0, 1, 0])


def camera_flythrough(waypoints: list, frame_idx: int,
                      total_frames: int = 60):
    """Compute camera position for flythrough between waypoints.

    Args:
        waypoints: List of (origin, target, up) tuples
        frame_idx: Current frame number
        total_frames: Total frames

    Returns:
        Tuple of (origin, target, up) with linear interpolation.
    """
    t = frame_idx / max(1, total_frames - 1)
    n_segments = len(waypoints) - 1
    segment_t = t * n_segments
    seg_idx = min(int(segment_t), n_segments - 1)
    local_t = segment_t - seg_idx

    o0, t0, u0 = waypoints[seg_idx]
    o1, t1, u1 = waypoints[seg_idx + 1]

    origin = [o0[i] + (o1[i] - o0[i]) * local_t for i in range(3)]
    target = [t0[i] + (t1[i] - t0[i]) * local_t for i in range(3)]
    up = u0  # Keep up vector constant

    return (origin, target, up)
