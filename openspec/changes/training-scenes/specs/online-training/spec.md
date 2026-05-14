## ADDED Requirements

### Requirement: Training data generator SHALL integrate with StratifiedReplayBuffer
The `TrainingDataGenerator` SHALL accept a `StratifiedReplayBuffer` instance and populate it with generated training pairs. Each scene SHALL be identified by a hash of its scene_graph metadata, enabling per-scene stratified sampling.

#### Scenario: Generated pairs fill replay buffer
- **WHEN** `generator.generate_and_store(build_cornell_box, count=10)` is called with a StratifiedReplayBuffer
- **THEN** 10 training pairs SHALL be added to the buffer under the scene's topology hash
- **AND** subsequent `buffer.sample(other_scene_hash)` SHALL return pairs from this scene

#### Scenario: Multiple scenes create separate sub-buffers
- **WHEN** pairs are generated from 3 different scene builders
- **THEN** the replay buffer SHALL have 3 separate scene entries
- **AND** stratified sampling SHALL return diverse pairs across all scenes
