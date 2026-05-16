## ADDED Requirements

### Requirement: Tmux-based training session
The system SHALL provide a shell script (`scripts/start_training.sh`) that launches training in a detached tmux session. The script SHALL activate the project venv, set `NABLA_DEFAULT_DEVICE=gpu`, and run the training script.

#### Scenario: Start training in tmux
- **WHEN** `bash scripts/start_training.sh` is executed
- **THEN** a tmux session named "omen_training" SHALL be created
- **AND** `NABLA_DEFAULT_DEVICE=gpu` SHALL be set in the session environment
- **AND** the training script SHALL run inside the session
- **AND** stdout SHALL indicate how to attach: "tmux attach -t omen_training"

#### Scenario: Training survives IDE disconnect
- **WHEN** VSCode or the terminal is closed during training
- **THEN** the tmux session SHALL continue running
- **AND** training SHALL proceed uninterrupted
- **AND** the user can reattach with `tmux attach -t omen_training`

### Requirement: Training CLI arguments
The training script SHALL accept command-line arguments for scene selection, resolution, tile size, number of steps, and GPU/CPU mode.

#### Scenario: Custom scene and resolution
- **WHEN** `python scripts/start_training.py --scene cornell --resolution 1920x1080 --tile-size 512 --steps 100` is run
- **THEN** training SHALL use the Cornell Box scene at 1920x1080
- **AND** tiles SHALL be 512x512
- **AND** training SHALL run for 100 steps
- **AND** a checkpoint SHALL be saved at the end

#### Scenario: CPU-only mode
- **WHEN** `python scripts/start_training.py --device cpu` is run
- **THEN** `NABLA_DEFAULT_DEVICE` SHALL NOT be set to gpu
- **AND** all tensors SHALL remain on CPU
