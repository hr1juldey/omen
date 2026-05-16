# Mitsuba Differentiable Rendering for Omen Learning

## What Is Differentiable Rendering?

Mitsuba 3's `cuda_ad_rgb` variant provides **automatic differentiation through the entire rendering pipeline**. Every pixel is a differentiable function of scene parameters (material reflectance, light intensity, geometry position, camera pose).

This is "reverse rendering" — not just rendering a 3D scene to a 2D image, but computing how changes in scene parameters would change the image.

## Relation to Omen

Omen's pipeline: `noisy image → encode → latent → decode → predicted noise`

The noise in a path-traced image is NOT random — it follows physical laws governed by:
- BRDF sampling distributions
- Light source characteristics
- Scene geometry (occlusion, indirect bouncing)
- Sample count (variance ∝ 1/√SPP)

Differentiable rendering gives us the **gradient of image noise with respect to scene parameters**. This is physics-based knowledge that the model currently has to learn from scratch.

## Three Applications for Omen

### 1. Ground Truth Noise Maps (Near-term)

Render the same frame at SPP=4 and SPP=256. The pixel-level difference IS the noise, computed from first principles (photon statistics, BRDF sampling). This gives per-pixel noise magnitude and distribution — a direct supervision signal.

```
noise_gt = abs(render(spp=4) - render(spp=256))
predicted_noise = model.decode(latent, noisy)
loss += MSE(predicted_noise, noise_gt)  # physics-supervised
```

**Benefit**: Model learns actual noise distributions instead of just residual prediction.

### 2. Physics-Informed Loss (Medium-term)

After denoising, check physical plausibility through the renderer:

1. Denoise: `clean = noisy - model.predicted_noise`
2. Render reference: `ref = render(scene, spp=1)` (one sample)
3. Compare: the denoised image should be consistent with the scene's light transport

The differentiable renderer can compute: "given this denoised image, what scene parameters would produce it?" If the recovered parameters don't match the known scene, the denoised image has artifacts.

```
# Pseudocode — physics consistency loss
recovered_params = inverse_render(denoised_image)
physics_loss = MSE(recovered_params, known_scene_params)
```

**Benefit**: Catches hallucination — the model can't invent features that don't exist in the scene.

### 3. Scene Encoder Supervision (Long-term)

The scene encoder learns `scene_graph → latent` from scratch. Mitsuba's AD provides direct supervision:

- Compute sensitivity: `∂pixel_ij / ∂material_k` — how much does each pixel depend on each material?
- These sensitivities ARE what the scene encoder should capture
- Train the encoder to predict these sensitivities from the scene graph

```
# Ground truth from Mitsuba AD
pixel_sensitivities = d(render(scene)) / d(scene_params)

# Scene encoder should predict these
predicted_sensitivities = scene_encoder.encode(scene_graph)
loss += MSE(predicted_sensitivities, pixel_sensitivities)
```

**Benefit**: The scene encoder learns physically meaningful representations, not just statistical correlations.

## Implementation Considerations

### Performance
- Differentiable rendering through Mitsuba is ~2-3x slower than forward-only rendering
- For training data generation (offline), this is acceptable
- For real-time inference, use the forward-only variant

### Architecture Integration
```
                    ┌──────────────────┐
                    │  Mitsuba AD      │
                    │  (cuda_ad_rgb)   │
                    └────┬─────────────┘
                         │ gradients
                         ▼
┌──────────┐    ┌──────────────┐    ┌──────────┐
│ Scene     │───▶│ Scene        │───▶│ Fusion   │
│ Graph     │    │ Encoder      │    │          │
└──────────┘    └──────────────┘    └──────────┘
       │                ▲                   │
       │  physics       │  supervision      ▼
       │  sensitivities │            ┌──────────┐
       └────────────────┘            │ Decoder  │
                                     └──────────┘
                                          │
                                          ▼
                                    predicted_noise
```

### Bridging Nabla AD and Mitsuba AD
The two autodiff systems (nabla's trace-based, Mitsuba's Enoki-based) don't directly interop. The bridge is through **numpy**:
1. Mitsuba AD computes gradients → numpy arrays
2. Numpy arrays → nabla tensors (via `from_dlpack`)
3. Nabla's `value_and_grad` handles the rest

No need to unify the AD systems — they communicate through concrete values.

## Priority

This is **Phase 2** after basic training is stable. The immediate path:
1. Phase 1 (current): GPU rendering + multi-camera + animation training
2. Phase 2a: Ground truth noise maps from differentiable rendering
3. Phase 2b: Physics-informed loss
4. Phase 2c: Scene encoder supervision

## References

- Mitsuba 3 paper: "Mitsuba 3: A Retargetable Forward and Inverse Renderer" (Nimier-David et al., 2022)
- Differentiable path tracing: the `cuda_ad_rgb` variant uses Enoki for GPU-accelerated reverse-mode AD through the full path tracer
- Related work: "Physics-aware Denoising" uses similar principles for Monte Carlo denoising
