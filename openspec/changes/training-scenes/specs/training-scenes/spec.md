## ADDED Requirements

### Requirement: Cornell Box scene builder
The system SHALL provide `build_cornell_box()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: a box room (6 walls), red left wall, green right wall, white floor/ceiling/back, a small area light on the ceiling, and two diffuse boxes on the floor. All materials SHALL be diffuse BSDF. The scene_graph SHALL contain geometry vertices (8+ faces), 3 material types (red diffuse, green diffuse, white diffuse), and 1 area light.

#### Scenario: Cornell Box renders without error
- **WHEN** `build_cornell_box()` is called
- **THEN** it SHALL return a valid Mitsuba scene and a scene_graph dict
- **AND** rendering at 64spp SHALL produce an image with visible color bleeding from red/green walls

#### Scenario: Cornell Box scene_graph has correct structure
- **WHEN** `build_cornell_box()` returns `(_, scene_graph)`
- **THEN** `scene_graph["geometry"]["vertices"]` SHALL be a numpy array with shape (N, 3) where N > 0
- **AND** `scene_graph["materials"]["params"]` SHALL have at least 3 rows (red, green, white)
- **AND** `scene_graph["lights"]["params"]` SHALL have exactly 1 row (area light)

### Requirement: Veach Ajar Door scene builder
The system SHALL provide `build_veach_ajar()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: a dark room with a slightly open door letting light in, a glass sphere (dielectric), a metal sphere (conductor), a matte sphere (diffuse), and 3 light sources (point, spot, area from door). The scene_graph SHALL list dielectric, conductor, and diffuse materials plus 3 light types.

#### Scenario: Veach scene renders with multiple light types
- **WHEN** `build_veach_ajar()` is called and rendered
- **THEN** the render SHALL show caustics from the glass sphere
- **AND** metallic reflections from the conductor sphere
- **AND** contributions from all 3 light source types

#### Scenario: Veach scene_graph lists all BSDF types
- **WHEN** `build_veach_ajar()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain "dielectric", "conductor", and "diffuse"
- **AND** `scene_graph["lights"]["types"]` SHALL contain "point", "spot", and "area"

### Requirement: Shaderball scene builder
The system SHALL provide `build_shaderball()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: a central sphere displayed on a checkerboard plane, with material variants (conductor, roughconductor, plastic, roughplastic, dielectric) rendered as separate spheres or as a configurable single sphere. The scene_graph SHALL list all 5+ material types.

#### Scenario: Shaderball renders all material types
- **WHEN** `build_shaderball()` is called and rendered at 64spp
- **THEN** the render SHALL show distinct material appearances for each BSDF type
- **AND** no NaN or inf values in the output

#### Scenario: Shaderball scene_graph has comprehensive material coverage
- **WHEN** `build_shaderball()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain at least 5 distinct BSDF types
- **AND** the types SHALL include "conductor", "roughconductor", "plastic", "roughplastic", and "dielectric"

### Requirement: Studio Product scene builder
The system SHALL provide `build_studio_product()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: 2-3 product objects (spheres/cylinders) with conductor and roughplastic materials, a ground plane, and 3-point studio lighting (key, fill, rim area lights). No external HDRI files required — use constant environment + area lights.

#### Scenario: Studio scene renders with 3-point lighting
- **WHEN** `build_studio_product()` is called and rendered
- **THEN** the render SHALL show well-lit product objects with key/fill/rim light contributions
- **AND** metallic and plastic materials SHALL be visually distinguishable

#### Scenario: Studio scene_graph lists conductor and roughplastic
- **WHEN** `build_studio_product()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain "roughconductor" and "roughplastic"
- **AND** `scene_graph["lights"]["types"]` SHALL contain "area" at least 3 times

### Requirement: Foggy Corridor scene builder
The system SHALL provide `build_foggy_corridor()` that returns a `(mi.Scene, scene_graph)` tuple. The scene SHALL contain: an L-shaped corridor with diffuse walls, a null-BSDF volume boundary containing a homogeneous medium (fog), and 2 lights (point, spot). The scene_graph SHALL include the null BSDF type and volume parameters.

#### Scenario: Foggy corridor renders with volumetric scattering
- **WHEN** `build_foggy_corridor()` is called with `integrator="volpath"` and rendered
- **THEN** the render SHALL show visible fog/light scattering in the corridor
- **AND** no NaN or black pixels from failed volume integration

#### Scenario: Foggy corridor scene_graph includes volume metadata
- **WHEN** `build_foggy_corridor()` returns `(_, scene_graph)`
- **THEN** `scene_graph["materials"]["types"]` SHALL contain "null" (volume boundary)
- **AND** the scene_graph SHALL include a "volume" key with sigma_t and albedo parameters

### Requirement: Decoder is a residual noise predictor
The decoder SHALL NOT reconstruct the full image. It SHALL take (JEPA latent, noisy image) as input and output a noise/residual map. The clean image SHALL be recovered as `clean = noisy - predicted_noise`. The decoder SHALL use a U-Net architecture with skip connections, Pixel Shuffle (or DySample) upsampling (NOT Conv2dTranspose), and JEPA latent injection at the bottleneck. Skip connections at higher resolutions SHALL use MLA-style compression.

#### Scenario: Decoder outputs residual, not full image
- **WHEN** `decoder.forward(jepa_latent, noisy_image)` is called
- **THEN** the output SHALL have the same spatial dimensions as `noisy_image`
- **AND** the output SHALL be a 3-channel (RGB) residual/noise map
- **AND** `noisy_image - output` SHALL produce the denoised image

#### Scenario: Decoder uses U-Net with skip connections
- **WHEN** the decoder processes a noisy image
- **THEN** the encoder path SHALL downsample the noisy image through 4 stages (64→128→256 channels)
- **AND** the decoder path SHALL upsample with skip connections from each encoder stage
- **AND** upsampling SHALL use Pixel Shuffle or DySample (NOT Conv2dTranspose)
- **AND** the 1024-dim JEPA latent SHALL be injected at the bottleneck

#### Scenario: No checkerboard artifacts from upsampling
- **WHEN** the decoder outputs a residual map
- **THEN** the output SHALL NOT exhibit checkerboard artifacts characteristic of Conv2dTranspose

### Requirement: Training data generator
The system SHALL provide `TrainingDataGenerator` class that runs online training using a diffusion-like pipeline: render GT at full HD + high SPP → encode to target latent, render noisy at full HD + low SPP → encode with scene_graph → JEPA loss + decoder noise prediction loss → backprop → free images. Images SHALL NOT be saved to disk by default. A `save_images` toggle SHALL be available for debugging.

#### Scenario: Online training step (no disk I/O)
- **WHEN** `TrainingDataGenerator.train_step_online(build_cornell_box)` is called
- **THEN** it SHALL render one GT image at 256 SPP at full HD resolution (1920x1080)
- **AND** render one noisy image at 4 SPP at the same full HD resolution
- **AND** encode both through the model to compute JEPA loss
- **AND** compute decoder noise prediction loss: MSE(predicted_noise, gt - noisy)
- **AND** backprop through ALL components (encoder + decoder + SIGReg + episodic)
- **AND** free both image arrays from memory after loss computation
- **AND** NOT write any files to disk

#### Scenario: Full HD ground truth render
- **WHEN** `TrainingDataGenerator` is initialized with `resolution=(1920, 1080)` and `gt_spp=256`
- **THEN** the GT render SHALL be exactly 1920x1080 at 256 SPP
- **AND** the noisy render SHALL be the same 1920x1080 at the configured low SPP

#### Scenario: Debug save toggle
- **WHEN** `TrainingDataGenerator` is initialized with `save_images=True` and `output_dir="./debug/"`
- **THEN** each rendered pair SHALL be saved as .exr files to the output directory
- **AND** a log message SHALL indicate the file path

#### Scenario: Multi-camera training step
- **WHEN** `TrainingDataGenerator.train_step_online(build_cornell_box, camera="all")` is called
- **THEN** it SHALL run one train_step per camera position (5 cameras for Cornell Box)
- **AND** each step SHALL use independent random seeds for noisy renders

### Requirement: Extended AOV passes with per-pass light path denoising
The system SHALL capture 7+ AOV passes beyond the basic 3 (albedo, normal, depth): diffuse_direct, glossy_direct, transmission_direct, and volume_direct. Each light path type SHALL be available as a separate channel for the MoE expert routing and decoder. The system SHALL use these separate passes to preserve caustics (transmission), specular highlights (glossy), and volumetric scattering (volume) that traditional denoisers (OIDN, OptiX) destroy.

#### Scenario: Extended AOV capture
- **WHEN** a scene is rendered via `_render_with_aov()`
- **THEN** the returned AOV dict SHALL contain keys for `albedo`, `normal`, `depth`, `diffuse_direct`, `glossy_direct`, `transmission_direct`, and `volume_direct`
- **AND** each pass SHALL be a numpy array with shape (H, W, 3) matching the render resolution

#### Scenario: Caustic preservation via transmission pass
- **WHEN** a scene with a dielectric (glass) object is rendered and denoised
- **THEN** caustic patterns from the glass object SHALL be preserved in the denoised output
- **AND** the caustic region PSNR SHALL be within 2dB of the non-caustic region PSNR

#### Scenario: Volumetric preservation via volume pass
- **WHEN** the Foggy Corridor scene is rendered and denoised
- **THEN** volumetric scattering (fog) SHALL be preserved in the denoised output
- **AND** fog density SHALL NOT be flattened or removed

### Requirement: Graceful degradation for missing AOV passes
The system SHALL classify AOV passes into tiers: MUST (albedo, normal, depth — always required) and NICE (diffuse_direct, glossy_direct, transmission_direct, volume_direct — optional). The model SHALL train with all passes but gracefully degrade when NICE passes are missing at inference. Missing NICE passes SHALL be zero-filled. Missing MUST passes SHALL raise an error. Scene graph material metadata SHALL provide fallback knowledge when AOV passes are unavailable.

#### Scenario: Must passes missing → error
- **WHEN** a render does not provide albedo, normal, or depth passes
- **THEN** the system SHALL raise a `ValueError` indicating which MUST pass is missing
- **AND** SHALL NOT proceed with denoising

#### Scenario: Nice passes missing → graceful degradation
- **WHEN** a render provides only MUST passes (albedo, normal, depth) without any NICE passes
- **THEN** the system SHALL zero-fill missing NICE passes
- **AND** proceed with denoising using scene_graph metadata as fallback
- **AND** output quality SHALL degrade by less than 2dB PSNR compared to full-pass denoising

#### Scenario: Random pass dropout during training
- **WHEN** `TrainingDataGenerator` is training
- **THEN** 20% of training steps SHALL randomly drop NICE-2/NICE-3 passes
- **AND** 10% of training steps SHALL drop ALL NICE passes
- **AND** 0% of training steps SHALL drop MUST passes
- **AND** the model SHALL learn to use scene_graph metadata when AOV passes are unavailable

#### Scenario: Scene graph fallback for caustics without transmission pass
- **WHEN** `transmission_direct` pass is missing but `scene_graph["materials"]["types"]` contains `"dielectric"`
- **THEN** the model SHALL still preserve caustics in the denoised output using scene_graph knowledge
- **AND** caustic region quality SHALL be within 3dB PSNR of full-pass denoising

### Requirement: CLI entry point for scene rendering
The system SHALL provide a CLI via `python -m omen.scenes` that can render scenes and run online training. By default, images are NOT saved (online training mode). The `--save-images` flag enables saving to disk for debugging.

#### Scenario: Render Cornell Box via CLI
- **WHEN** `python -m omen.scenes --scene cornell --spp 64 --save-images --output cornell.exr` is executed
- **THEN** the system SHALL render the Cornell Box at 64spp and save to `cornell.exr`
- **AND** print the render time and output path

#### Scenario: Run online training (no saves)
- **WHEN** `python -m omen.scenes --scene cornell --noisy-spp 4 --gt-spp 256 --count 5 --resolution 1920x1080` is executed
- **THEN** the system SHALL run 5 online training steps (render GT + noisy, train, discard)
- **AND** NOT save any images to disk
- **AND** print training loss per step

#### Scenario: List available scenes
- **WHEN** `python -m omen.scenes --list` is executed
- **THEN** the system SHALL print all 5 scene names with brief descriptions
