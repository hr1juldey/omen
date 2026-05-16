## ADDED Requirements

### Requirement: Log file output
The training CLI SHALL support `--log-file PATH` to write all training output to a log file in addition to stdout.

#### Scenario: Log file writing
- **WHEN** user passes `--log-file training.log`
- **THEN** all logger output is written to `training.log` and also printed to stdout

### Requirement: Tmux launcher script
A `scripts/train_tmux.sh` script SHALL launch training in a detached tmux session with log file output.

#### Scenario: Launch tmux training
- **WHEN** user runs `bash scripts/train_tmux.sh --scenes all --steps 10 --steps-per-frame 100`
- **THEN** a tmux session named "omen-training" is created, training runs detached, logs written to `logs/training_YYYYMMDD_HHMM.log`

#### Scenario: Reattach to training
- **WHEN** user runs `tmux attach -t omen-training`
- **THEN** the training session output is visible

### Requirement: Tail log for monitoring
The log file SHALL be tail-able for progress monitoring without attaching to tmux.

#### Scenario: Monitor progress
- **WHEN** user runs `tail -f logs/training_*.log`
- **THEN** training progress (loss, scene, step) is visible in real-time
