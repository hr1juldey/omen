"""EpisodicCorrection network for fast per-scene adaptation.

Separate 2-layer MLP (~100K params) with own optimizer at 400x higher lr.
Replaces LoRA as the adaptation mechanism.

Architecture: main_output + scene_context → hidden → correction → add to output.
"""

from nabla import nn
import nabla as nb


class EpisodicCorrection(nn.Module):
    """Separate fast-adaptation network (~100K params).

    Own optimizer, own (higher) learning rate (2e-2 vs base 5e-5).
    Architecturally independent from main model.

    When disabled (enabled=False), returns main_output unchanged (identity passthrough).
    """

    def __init__(self, dim: int = 192, hidden: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim * 2, hidden),
            nn.SiLU(),
            nn.Linear(hidden, dim),
        )

    def forward(self, main_output, scene_context, enabled: bool = True):
        """Apply episodic correction if enabled.

        Args:
            main_output: (batch, dim) main model output
            scene_context: (batch, dim) scene context embedding
            enabled: if False, identity passthrough

        Returns:
            corrected output (main_output + correction) or main_output unchanged
        """
        if not enabled:
            return main_output

        # Concatenate main output and scene context
        combined = nb.concat([main_output, scene_context], axis=-1)
        correction = self.net(combined)
        return main_output + correction
