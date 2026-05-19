## ADDED Requirements

### Requirement: Tile-based AOV processing
The system SHALL split render images into 256x256 tiles with 16px overlap and process each tile independently through the encoder pipeline.

#### Scenario: 512x512 image tiling
- **WHEN** a 512x512 render is processed
- **THEN** the system produces 4 tiles (2x2 grid) of 256x256 each, with 16px overlap

#### Scenario: 1024x1024 image tiling
- **WHEN** a 1024x1024 render is processed
- **THEN** the system produces 16 tiles (4x4 grid) of 256x256 each

### Requirement: Deep residual scene encoder
The system SHALL encode scene graph features through a 12-layer residual MLP (18→128→...→128) producing a 128-dimensional scene latent vector.

#### Scenario: Scene encoding produces 128d latent
- **WHEN** scene graph features (geometry 6d + materials 5d + lights 7d = 18d) are provided
- **THEN** the encoder outputs a (1, 128) latent vector through 12 residual layers

#### Scenario: Residual connections maintain gradient flow
- **WHEN** backward pass runs through the 12-layer encoder
- **THEN** gradients SHALL NOT vanish (all finite, no NaN) after 1000 training steps

### Requirement: FiLM conditioning from scene latent
The system SHALL inject scene context at every convolutional layer via Feature-wise Linear Modulation: `output = γ * conv_output + β` where γ and β are computed from scene_latent.

#### Scenario: FiLM modulation per conv layer
- **WHEN** a 256x256 AOV tile passes through Conv1 and Conv2
- **THEN** each conv output is modulated by FiLM(γ, β) generated from the same scene_latent before activation

#### Scenario: Different scenes produce different modulations
- **WHEN** cornell_box scene_latent and shaderball scene_latent modulate the same tile
- **THEN** γ and β values SHALL differ between the two scenes

### Requirement: Tile position encoding
The system SHALL append 2 positional encoding channels (sin of normalized x,y tile coordinates) to the AOV input, making the total input 12 channels.

#### Scenario: Center tile vs edge tile encoding
- **WHEN** tile (1,1) and tile (0,0) of a 4x4 grid are processed
- **THEN** the positional encoding channels differ between the two tiles

### Requirement: 10-channel AOV input
The system SHALL accept 10-channel AOV input: albedo(3) + normal(3) + depth(1) + material_id(1) + motion_vectors(2).

#### Scenario: AOV packing from Mitsuba render
- **WHEN** a Mitsuba AOV render produces albedo, sh_normal, and depth buffers
- **THEN** the system packs them into a (H, W, 10) tensor with missing channels zeroed

### Requirement: Multi-term loss function
The system SHALL compute training loss as: `total = L_mse + λ_sigreg * L_sigreg + λ_energy * L_energy` where L_mse is JEPA prediction loss, L_sigreg is variance regularization, and L_energy is energy conservation loss.

#### Scenario: Loss components are finite
- **WHEN** training runs for 1000 steps
- **THEN** all three loss components remain finite at every step

#### Scenario: Loss converges to zero
- **WHEN** training runs for 1000 steps with AdamW lr=1e-3 on a single scene
- **THEN** L_mse converges to approximately 0.000

### Requirement: SIGReg variance regularization
The system SHALL prevent latent collapse by computing `L_sigreg = -mean(log(std(latent) + eps))` on the fused latent.

#### Scenario: Latent variance maintained
- **WHEN** training runs for 1000 steps
- **THEN** the standard deviation of the fused latent SHALL remain above 1e-3

### Requirement: Energy conservation physics loss
The system SHALL compute `L_energy = mean(square(sum(abs(render_latent)) - sum(abs(target_latent))))` to enforce that the predicted latent preserves total scene energy.

#### Scenario: Energy loss is non-negative
- **WHEN** any training step computes the energy loss
- **THEN** L_energy SHALL be >= 0

### Requirement: GPU rendering with scene selection
The system SHALL render scenes using Mitsuba with cuda_ad_rgb variant (fallback to llvm_ad_rgb, scalar_rgb) and randomly select from 5 scenes with different cameras.

#### Scenario: Random scene per training epoch
- **WHEN** training begins a new phase
- **THEN** a random scene and camera are selected from (cornell, veach, shaderball, studio, foggy)

### Requirement: Kill switch for memory safety
The system SHALL monitor RSS memory and exit with code 99 when RSS exceeds 28GB, with a warning at 24GB.

#### Scenario: Memory exceeds kill threshold
- **WHEN** RSS exceeds 28GB during training
- **THEN** the process exits with code 99

### Requirement: Sustained training mode
The system SHALL support a --sustain CLI flag that runs training for a specified number of minutes with cosine LR decay.

#### Scenario: 30-minute sustained training
- **WHEN** --sustain 30 is passed
- **THEN** training runs for 30 minutes with linearly decaying learning rate

### Requirement: Cross-attention fusion
The system SHALL fuse render_latent and scene_latent via gated cross-attention: `gate = sigmoid(render @ W_g + b_g)`, `fused = LayerNorm(render + gate * scene)`.

#### Scenario: Fusion produces 128d output
- **WHEN** render_latent (1, 128) and scene_latent (1, 128) are fused
- **THEN** the output is (1, 128)
