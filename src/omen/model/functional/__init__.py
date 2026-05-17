"""Functional (JAX-style) forward passes for OmenJEPA sub-modules.

Each function takes a flat params dict (subset) and inputs, returning
outputs using only nabla tensor operations — no nn.Module state mutation.
"""

from omen.model.functional.ar_predictor import ar_predictor_fn
from omen.model.functional.confidence import confidence_fn
from omen.model.functional.cross_attn import cross_attn_fn
from omen.model.functional.decoder import decoder_fn
from omen.model.functional.episodic import episodic_fn
from omen.model.functional.render_encoder import render_encoder_fn
from omen.model.functional.scene_encoder import scene_encoder_fn
from omen.model.functional.sigreg import sigreg_fn


def _extract_prefix(params, prefix):
    """Return subset dict of params starting with *prefix*, stripping it."""
    return {k[len(prefix) :]: v for k, v in params.items() if k.startswith(prefix)}


__all__ = [
    "_extract_prefix",
    "scene_encoder_fn",
    "render_encoder_fn",
    "cross_attn_fn",
    "decoder_fn",
    "sigreg_fn",
    "confidence_fn",
    "ar_predictor_fn",
    "episodic_fn",
]
