## ADDED Requirements

### Requirement: Linear warmup schedule
The trainer SHALL support a linear LR warmup from near-zero to base LR over a configurable number of warmup steps.

#### Scenario: Warmup phase
- **WHEN** `--lr-warmup 100` is passed and current step < 100
- **THEN** the effective LR SHALL be `base_lr * (step / warmup_steps)`

#### Scenario: No warmup (default)
- **WHEN** `--lr-warmup` is not passed or is 0
- **THEN** LR starts at base_lr immediately with no warmup phase

### Requirement: Cosine decay schedule
After warmup, the trainer SHALL apply cosine decay from base LR to a minimum LR over the remaining training steps.

#### Scenario: Decay phase
- **WHEN** current step > warmup_steps
- **THEN** the effective LR SHALL be `min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * progress))` where progress = (step - warmup) / (total_steps - warmup)

### Requirement: Schedule composes with surprise modulation
The LR schedule SHALL compose with existing surprise-modulated LR as a multiplicative factor.

#### Scenario: Surprise on top of schedule
- **WHEN** z_score > 0 during warmup
- **THEN** the LR SHALL be `scheduled_lr * (1 + surprise_scale * min(z_score, 5.0))`
