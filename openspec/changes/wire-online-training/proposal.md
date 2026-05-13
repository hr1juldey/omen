## Why

Omen's training infrastructure (OmenTrainer, data_gen, checkpoint, JEPA compute_loss) is fully implemented but completely disconnected from the render pipeline. The denoiser runs inference-only — it never learns from the scenes it renders. The design doc planned "per-scene LoRA fine-tuning after 3+ renders" but no code invokes it. The user's vision: no external dataset, learn purely from usage — "if model is not there, make one from the scene given."

## What Changes

- Wire `OmenTrainer.train_step()` into the denoiser render loop so every progressive batch is a training opportunity
- Add model initialization from scratch when no checkpoint exists (first-ever render creates the model)
- Use progressive render signal: early low-spp batches = noisy input, final accumulated result = pseudo-ground-truth
- Trigger LoRA fine-tuning after N renders of the same scene (detected via topology hash)
- Save checkpoints after training steps so learned weights persist across sessions
- `jepa_bridge.py` gains training mode: initialize model fresh if no weights found, train before inference
- `denoiser.py` gains a training phase before denoising: render training pair, train, then denoise

## Capabilities

### New Capabilities
- `online-training`: Self-supervised JEPA training wired into the render pipeline — no external dataset needed, learns from progressive render batches
- `model-bootstrap`: Initialize JEPA model from scratch on first render when no checkpoint exists, train on the scene being rendered

### Modified Capabilities
- `jepa-inference`: JEPABridge must support both training and inference mode, not just pretrained weight loading

## Impact

- **Modified files**: `src/omen/jepa_bridge.py` (add training/init mode), `src/omen/modes/denoiser.py` (add training phase), `src/omen_engine/session.py` (pass training config)
- **Dependencies**: Nabla must be installed for training (graceful fallback to inference-only if unavailable)
- **Model files**: New `.omen` checkpoint files created in `~/.omen/checkpoints/`
- **No breaking changes**: Existing inference-only path preserved as fallback when Nabla unavailable
