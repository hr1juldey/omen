## 1. Scene Infrastructure

- [ ] 1.1 Add `_tf()` helper and `_build_sensor_multi()` to `src/omen/scenes.py` — multi-camera sensor builder that places N cameras at configurable positions around the scene, returning a list of (camera_name, sensor) pairs
- [ ] 1.2 Add `_build_scene_graph()` helper — extracts geometry vertices, material params, light params from scene definition dicts into numpy arrays matching `SceneGraphEncoder` input format
- [ ] 1.3 Add `SCENE_REGISTRY` dict mapping scene names to their builder functions
- [ ] 1.4 Add `SceneAnimation` class — takes a base scene dict + a list of animation channels (camera, mesh, material, light), generates per-frame scene dicts by interpolating parameters

## 2. Cornell Box

- [ ] 2.1 Build `build_cornell_box()` — 6-wall box room: red left wall (diffuse, 0.63,0.065,0.05), green right wall (diffuse, 0.14,0.45,0.091), white floor/ceiling/back (diffuse, 0.725,0.71,0.68), tall box and short box on floor (white diffuse)
- [ ] 2.2 Place area light on ceiling — rectangle emitter (0.8,0.8,0.8) at y=1.325
- [ ] 2.3 Place 5 cameras: front, left-45, right-45, top-down, close-up on tall box
- [ ] 2.4 Build scene_graph metadata — geometry (12 vertices from 6 walls + 2 boxes), materials (3 types), lights (1 area)
- [ ] 2.5 Camera animation: 12-frame orbit around scene center at fixed radius and height
- [ ] 2.6 Mesh animation: 8-frame sequence rotating the tall box 90°, translating the short box across the floor
- [ ] 2.7 Material animation: 6-frame sequence shifting the red wall color toward orange (0.63→0.8), green wall toward teal (0.14→0.3)
- [ ] 2.8 Light animation: 8-frame sequence dimming the area light from full (1.0) to half (0.5) and shifting color temperature from neutral to warm

## 3. Veach Ajar Door

- [ ] 3.1 Build `build_veach_ajar()` — dark room (diffuse black walls) with slightly open door gap on one wall
- [ ] 3.2 Place glass sphere (dielectric, ior=1.5), metal sphere (conductor, Au), matte sphere (diffuse, white) on floor
- [ ] 3.3 Place 3 lights: point light (warm, through door gap), spot light (from above), area light (behind camera)
- [ ] 3.4 Place 5 cameras: through door gap, side view, above, close on glass sphere, close on metal sphere
- [ ] 3.5 Build scene_graph metadata — 3 material types (dielectric, conductor, diffuse), 3 light types (point, spot, area)
- [ ] 3.6 Camera animation: 10-frame dolly moving from far corridor through the door gap into the room
- [ ] 3.7 Mesh animation: 8-frame sequence opening the door from 5° to 45°, moving glass sphere 0.3 units to the right
- [ ] 3.8 Material animation: 6-frame sequence changing glass IOR from 1.3 to 1.8 (visible refraction shift), metal roughness from 0.02 to 0.3
- [ ] 3.9 Light animation: 8-frame sequence moving point light from behind door to room center, dimming spot light from 80 to 20

## 4. Shaderball

- [ ] 4.1 Build `build_shaderball()` — ground plane (checkerboard roughplastic), 5 spheres in a row: conductor (mirror), roughconductor (Cu, α=0.15), plastic (SSS-like, skin tone), roughplastic (clay, α=0.2), dielectric (glass, ior=1.5)
- [ ] 4.2 Place area light above + constant environment emitter (0.4,0.4,0.45)
- [ ] 4.3 Place 4 cameras: front, 45-degree angle, top-down, close-up on material row
- [ ] 4.4 Build scene_graph metadata — 5 material types, 1 area light + 1 env light
- [ ] 4.5 Camera animation: 12-frame circular orbit around the material row at 45° elevation
- [ ] 4.6 Mesh animation: 8-frame sequence scaling each sphere from 0.5x to 1.5x size (one at a time), plus a bounce on the central sphere
- [ ] 4.7 Material animation: 10-frame sequence sweeping roughness on roughconductor from 0.0 to 0.5, roughplastic from 0.1 to 0.4, then cycling plastic IOR from 1.3 to 1.7
- [ ] 4.8 Light animation: 8-frame sequence rotating area light position 180° around the spheres (simulating studio light sweep)

## 5. Studio Product

- [ ] 5.1 Build `build_studio_product()` — 3 objects: gold sphere (roughconductor Au, α=0.05), copper cylinder (roughconductor Cu, α=0.12), matte vase (roughplastic, α=0.25) on a dark ground plane
- [ ] 5.2 Place 3-point studio lighting: key area light (warm, 45° right-above), fill area light (cool, 45° left), rim area light (behind/above)
- [ ] 5.3 Place 4 cameras: product front, product 3/4 view, overhead, low-angle hero
- [ ] 5.4 Build scene_graph metadata — roughconductor + roughplastic materials, 3 area lights
- [ ] 5.5 Camera animation: 12-frame turntable orbit at 3/4 view height (full 360° sweep)
- [ ] 5.6 Mesh animation: 8-frame sequence lifting gold sphere up 0.5 units, rotating copper cylinder 180°, scaling vase from 0.8x to 1.2x
- [ ] 5.7 Material animation: 6-frame sequence changing gold roughness 0.02→0.2, copper roughness 0.1→0.3, vase color shift from matte gray to warm terracotta
- [ ] 5.8 Light animation: 8-frame sequence dimming key light from 100% to 30%, brightening fill to compensate, shifting rim light color from neutral to blue

## 6. Foggy Corridor

- [ ] 6.1 Build `build_foggy_corridor()` — L-shaped corridor (diffuse gray walls), null-BSDF volume boundary (homogeneous medium, σ_t=0.2, albedo=0.9, Henyey-Greenstein g=0.3)
- [ ] 6.2 Place point light at corridor junction, spot light at one end
- [ ] 6.3 Place 4 cameras: corridor entrance, junction looking both ways, down the long arm
- [ ] 6.4 Build scene_graph metadata — diffuse + null BSDF, volume params (σ_t, albedo, g), 2 lights
- [ ] 6.5 Camera animation: 10-frame walkthrough from corridor entrance to junction to end of long arm
- [ ] 6.6 Mesh animation: 6-frame sequence moving a diffuse box obstacle along the corridor (blocking/unblocking light path)
- [ ] 6.7 Material animation: 8-frame sequence changing wall color from gray to warm white, floor from dark to checkerboard
- [ ] 6.8 Light animation: 8-frame sequence varying fog density (σ_t 0.05→0.5), moving point light along corridor, spot light cone narrowing from 45° to 15°

## 7. Training Data Generator

- [ ] 7.1 Implement `TrainingDataGenerator.__init__()` — accepts max_resolution, gpu flag, output_dir
- [ ] 7.2 Implement `generate_pair()` — renders noisy (low SPP) + clean (high SPP) images from any scene builder, returns (noisy, clean, scene_graph) as numpy arrays
- [ ] 7.3 Implement multi-camera support — generate_pair renders from all camera positions, returning a list of pairs per scene
- [ ] 7.4 Implement `generate_batch()` — renders N pairs with different seeds, saves to output_dir as .npz files
- [ ] 7.5 Implement `generate_animation_pairs()` — renders temporal frame sequences from all 4 animation channels (camera, mesh, material, light) for ARPredictor training; each frame pair gets (noisy, clean) renders
- [ ] 7.6 Integrate with `StratifiedReplayBuffer` — `generate_and_store()` method that populates replay buffer keyed by scene_graph hash

## 8. CLI Entry Point

- [ ] 8.1 Add `__main__.py` or update `scenes.py` with argparse CLI: `--scene`, `--spp`, `--spp-pair`, `--count`, `--output`, `--output-dir`, `--list`, `--camera`, `--animate`, `--animate-type` (camera|mesh|material|light|all)
- [ ] 8.2 Implement `--list` — print SCENE_REGISTRY names + descriptions
- [ ] 8.3 Implement `--spp-pair` mode — generate training pairs and save as .npz
- [ ] 8.4 Implement `--animate` flag — render animation frames for temporal data
- [ ] 8.5 Implement `--animate-type` flag — select which animation channels to render (camera, mesh, material, light, or all)
- [ ] 8.6 Implement `--camera all` flag — render from all camera positions

## 9. Validation

- [ ] 9.1 Verify all 5 scenes render at 64spp without errors (CPU + GPU)
- [ ] 9.2 Verify scene_graph metadata has correct shapes for SceneGraphEncoder
- [ ] 9.3 Verify TrainingDataGenerator produces valid (noisy, clean) pairs
- [ ] 9.4 Verify all 4 animation types produce valid temporal sequences (camera, mesh, material, light)
- [ ] 9.5 Verify animation frame-to-frame coherence (no sudden jumps, smooth parameter interpolation)
- [ ] 9.6 Run full test suite to ensure no regressions
