## ADDED Requirements

### Requirement: Generic animation dispatch
The training CLI SHALL resolve the animation generator for any scene using the naming convention `{scene_name}_animations` from `omen.scenes`.

#### Scenario: Animation dispatch for known scene
- **WHEN** user passes `--scene veach --animation camera_orbit`
- **THEN** the trainer imports `veach_animations` from `omen.scenes` and calls it with `base_resolution=(w, h)` to get animation frames

#### Scenario: Animation dispatch for scene without generator
- **WHEN** user passes `--animation camera_orbit` for a scene with no `{scene_name}_animations` function
- **THEN** the trainer SHALL log a warning and fall back to static multi-camera training with `camera="all"`

### Requirement: Animation type validation
The available animation types SHALL be determined dynamically from the scene's animation generator keys, not hardcoded.

#### Scenario: Scene-specific animation types
- **WHEN** user passes `--scene cornell --animation camera_orbit`
- **THEN** the trainer uses cornell_animations()["camera_orbit"] as the frame generator

#### Scenario: Invalid animation type for scene
- **WHEN** user passes `--animation invalid_type` and the scene's generator does not have that key
- **THEN** the trainer SHALL raise a ValueError listing available animation types for that scene
