import torch
import torch.nn as nn

from xfold.nn.diffusion_transformer import DiffusionTransformer

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()

        self.c_act = 768
        self.transformer = DiffusionTransformer()
        self.output_norm = nn.LayerNorm(self.c_act, bias=False)

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']

        return single_cond, pair_cond

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool
    ) -> torch.Tensor:
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA /
            torch.sqrt(noise_level**2 + SIGMA_DATA**2)
        )

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask[..., None]
