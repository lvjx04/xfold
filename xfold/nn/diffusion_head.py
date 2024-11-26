import torch
import torch.nn as nn

from xfold import feat_batch
from xfold.nn import featurization
from xfold.nn.diffusion_transformer import DiffusionTransformer, DiffusionTransition

# Carefully measured by averaging multimer training set.
SIGMA_DATA = 16.0


def fourier_embeddings(x: torch.Tensor, dim: int) -> torch.Tensor:
    weight = torch.randn([dim], dtype=x.dtype, device=x.device)
    bias = torch.rand([dim], dtype=x.dtype, device=x.device)
    return torch.cos(2 * torch.pi * (x[..., None] * weight + bias))


class DiffusionHead(nn.Module):
    def __init__(self):
        super(DiffusionHead, self).__init__()

        self.c_act = 768
        self.pair_channel = 128
        self.seq_channel = 384
        self.transformer = DiffusionTransformer()
        self.output_norm = nn.LayerNorm(self.c_act, bias=False)

        self.pair_cond_initial_norm = nn.LayerNorm(self.c_act, bias=False)
        self.pair_cond_initial_projection = nn.Linear(
            self.c_act, self.pair_channel, bias=False)

        self.pair_transition_0 = DiffusionTransition(self.pair_channel)
        self.pair_transition_1 = DiffusionTransition(self.pair_channel)

        self.single_cond_initial_norm = nn.LayerNorm(self.c_act, bias=False)
        self.single_cond_initial_projection = nn.Linear(
            self.c_act, self.seq_channel, bias=False)
        
        self.noise_embedding_initial_norm = nn.LayerNorm(self.c_act, bias=False)
        self.noise_embedding_initial_projection = nn.Linear(
            self.c_act, self.seq_channel, bias=False)
        
        self.single_transition_0 = DiffusionTransition(self.pair_channel)
        self.single_transition_1 = DiffusionTransition(self.pair_channel)

    def _conditioning(
        self,
        batch,
        embeddings: dict[str, torch.Tensor],
        noise_level: torch.Tensor,
        use_conditioning: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        single_embedding = use_conditioning * embeddings['single']
        pair_embedding = use_conditioning * embeddings['pair']

        rel_features = featurization.create_relative_encoding(
            batch.token_features, max_relative_idx=32, max_relative_chain=2
        ).to(dtype=pair_embedding.dtype)
        features_2d = torch.concatenate([pair_embedding, rel_features], dim=-1)

        pair_cond = self.pair_cond_initial_projection(
            self.pair_cond_initial_norm(single_embedding)
        )

        pair_cond += self.pair_transition_0(pair_cond)
        pair_cond += self.pair_transition_1(pair_cond)

        target_feat = embeddings['target_feat']
        features_1d = torch.concatenate(
            [single_embedding, target_feat], dim=-1)
        single_cond = self.single_cond_initial_projection(
            self.single_cond_initial_norm(features_1d))

        noise_embedding = fourier_embeddings(
            (1 / 4) * torch.log(noise_level / SIGMA_DATA), dim=256
        )

        single_cond += self.noise_embedding_initial_projection(
            self.noise_embedding_initial_norm(noise_embedding)
        )

        single_cond += self.single_transition_0(single_cond)
        single_cond += self.single_transition_1(single_cond)

        return single_cond, pair_cond

    def forward(
        self,
        positions_noisy: torch.Tensor,
        noise_level: torch.Tensor,
        batch: feat_batch.Batch,
        embeddings: dict[str, torch.Tensor],
        use_conditioning: bool
    ) -> torch.Tensor:
        # Get conditioning
        trunk_single_cond, trunk_pair_cond = self._conditioning(
            batch=batch,
            embeddings=embeddings,
            noise_level=noise_level,
            use_conditioning=use_conditioning,
        )

        # Extract features
        sequence_mask = batch.token_features.mask
        atom_mask = batch.predicted_structure_info.atom_mask

        # Position features
        act = positions_noisy * atom_mask[..., None]
        act = act / torch.sqrt(noise_level**2 + SIGMA_DATA**2)

        skip_scaling = SIGMA_DATA**2 / (noise_level**2 + SIGMA_DATA**2)
        out_scaling = (
            noise_level * SIGMA_DATA /
            torch.sqrt(noise_level**2 + SIGMA_DATA**2)
        )

        return (
            skip_scaling * positions_noisy + out_scaling * position_update
        ) * atom_mask[..., None]
