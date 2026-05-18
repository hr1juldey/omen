## ADDED Requirements

### Requirement: Progressive scale-up test
The test file `tests/test_gpu_mojo_conv2d_backward.py` SHALL execute a sequence of 5 phases with increasing resolution and channel count. Each phase SHALL verify forward correctness and backward gradient correctness before proceeding.

#### Scenario: Phase progression
- **WHEN** the test runs
- **THEN** it SHALL execute phases in order: (16x16, 4→8, 1 conv), (16x16, 4→8→16, 2 conv), (32x32, 4→16→32, 2 conv), (64x64, 4→16→32, 2 conv), (64x64, 2+ conv training loop)

#### Scenario: RAM cleanup between phases
- **WHEN** each phase completes
- **THEN** the test SHALL run `gc.collect()`, delete all tensors, and verify RSS < 20GB before proceeding to next phase

### Requirement: Numerical gradient verification
The test SHALL compare analytical gradients from `Conv2dMojoOp.vjp_rule` against numerical gradients computed via finite differences (eps=1e-3) on a subset of parameters.

#### Scenario: Gradient accuracy within tolerance
- **WHEN** analytical and numerical gradients are compared for a single conv2d layer
- **THEN** `max(|analytical - numerical|)` SHALL be < 0.1

#### Scenario: Multi-layer gradient accuracy
- **WHEN** analytical and numerical gradients are compared for 2 chained conv2d layers
- **THEN** `max(|analytical - numerical|)` SHALL be < 0.1 for both layers

### Requirement: RAM guard
The test SHALL monitor process RSS and terminate with exit code 99 if RSS exceeds 20GB at any point.

#### Scenario: RSS exceeds 20GB
- **WHEN** process RSS > 20,480 MB
- **THEN** the test SHALL print `KILL: RSS=Xmb > 20GB <phase>` and call `sys.exit(99)`

### Requirement: GPU-only execution
The test SHALL require an NVIDIA GPU and SHALL NOT run on CPU-only systems.

#### Scenario: No GPU available
- **WHEN** `accelerator_count()` returns 0
- **THEN** the test SHALL print "No GPU — aborting" and return without error

### Requirement: Training loop proof
Phase 5 SHALL run a 10-step training loop with 2+ conv2d layers, AdamW optimizer, and verify loss decreases or stays stable (no NaN, no crash).

#### Scenario: Training completes without crash
- **WHEN** 10 training steps are executed with 2 conv2d layers on GPU
- **THEN** all steps SHALL complete without SIGABRT, NaN loss, or OOM error

#### Scenario: Loss is finite
- **WHEN** loss values are recorded across 10 steps
- **THEN** all loss values SHALL be finite (not NaN, not inf)
