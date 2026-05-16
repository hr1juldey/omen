## ADDED Requirements

### Requirement: Multi-scene training loop
The training CLI SHALL support `--scenes all` to cycle through all scenes in SCENE_REGISTRY in sorted order during a single training run.

#### Scenario: All scenes mode
- **WHEN** user passes `--scenes all`
- **THEN** the trainer iterates through cornell, foggy, shaderball, studio, veach in order, training on each scene's rendered pairs before moving to the next

#### Scenario: Single scene mode (default)
- **WHEN** user passes `--scene cornell` (or omits the flag)
- **THEN** training runs on cornell only, matching current behavior

### Requirement: Per-scene scene graph encoding
For each scene in the multi-scene loop, the trainer SHALL build the scene, extract the scene graph, and encode it once before tile extraction.

#### Scenario: Scene graph lifecycle
- **WHEN** the trainer starts training on a new scene
- **THEN** it calls `build_*(resolution)` to get (scene, scene_graph), encodes the scene graph to a nabla tensor once, and shares it across all tiles for that scene

### Requirement: Scene cycle logging
The trainer SHALL log scene name, scene index, and total scene count at the start of each scene's training round.

#### Scenario: Scene transition logging
- **WHEN** the trainer transitions from scene A to scene B
- **THEN** logger outputs "Scene 2/5: veach" with the scene name and progress
