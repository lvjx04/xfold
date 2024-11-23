import torch
import torch.nn as nn
import einops

from xfold.constants import atom_types
from xfold.nn import template, atom_layout, pairformer

_CONTACT_THRESHOLD = 8.0
_CONTACT_EPSILON = 1e-3


class DistogramHead(nn.Module):
    def __init__(self,
                 c_pair: int = 128,
                 num_bins: int = 64,
                 first_break: float = 2.3125,
                 last_break: float = 21.6875) -> None:
        super(DistogramHead, self).__init__()

        self.c_pair = c_pair
        self.num_bins = num_bins
        self.first_break = first_break
        self.last_break = last_break

        self.half_logits = nn.Linear(self.c_pair, self.num_bins, bias=False)

        breaks = torch.linspace(
            self.first_break,
            self.last_break,
            self.num_bins - 1,
        )

        self.register_buffer('breaks', breaks)

        bin_tops = torch.cat(
            (breaks, (breaks[-1] + (breaks[-1] - breaks[-2])).reshape(1)))
        threshold = _CONTACT_THRESHOLD + _CONTACT_EPSILON
        is_contact_bin = 1.0 * (bin_tops <= threshold)

        self.register_buffer('is_contact_bin', is_contact_bin)

    def forward(self, pair: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pair (torch.Tensor): pair embedding
                [*, N_token, N_token, C_z]

        Returns:
            torch.Tensor: distogram probability distribution
                [*, N_token, N_token, num_bins]
        """
        left_half_logits = self.half_logits(pair)
        right_half_logits = left_half_logits
        logits = left_half_logits + right_half_logits.transpose(-2, -3)
        probs = torch.softmax(logits, dim=-1)
        # precision=jax.lax.Precision.HIGHEST
        contact_probs = torch.einsum('ijk,k->ij', probs, self.is_contact_bin)

        return {
            'bin_edges': self.breaks,
            'contact_probs': contact_probs,
        }


class ConfidenceHead(nn.Module):
    """
    Implements Algorithm 31 in AF3
    """

    def __init__(self, c_single: int = 384, c_pair: int = 128, c_target_feat: int = 447, n_pairformer_layers=4):
        super(ConfidenceHead, self).__init__()

        self.c_single = c_single
        self.c_pair = c_pair
        self.c_target_feat = c_target_feat

        self.dgram_features_config = template.DistogramFeaturesConfig()

        self.left_target_feat_project = nn.Linear(
            self.c_target_feat, self.c_pair, bias=False)
        self.right_target_feat_project = nn.Linear(
            self.c_target_feat, self.c_pair, bias=False)
        self.distogram_feat_project = nn.Linear(
            self.dgram_features_config.num_bins, self.c_pair, bias=False)

        self.confidence_pairformer = nn.ModuleList([
            pairformer.PairformerBlock(
                c_single=self.c_single,
                c_pair=self.c_pair,
                with_single=True,
            ) for _ in range(n_pairformer_layers)
        ])

        self.num_plddt_bins = 50
        self.num_atom = atom_types.DENSE_ATOM_NUM
        self.bin_width = 1.0 / self.num_plddt_bins

        self.register_buffer('bin_centers', torch.arange(
            0.5 * self.bin_width, 1.0, self.bin_width))

        self.plddt_logits_ln = nn.LayerNorm(self.c_single)
        self.plddt_logits = nn.Linear(
            self.c_single, self.num_atom * self.num_plddt_bins, bias=False)

        self.experimentally_resolved_ln = nn.LayerNorm(self.c_single)
        self.experimentally_resolved_logits = nn.Linear(
            self.c_single, self.num_atom * 2, bias=False)

    def _embed_features(
        self,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo,
        pair_mask: torch.Tensor,
        target_feat: torch.Tensor,
    ) -> torch.Tensor:

        out = self.left_target_feat_project(target_feat)[..., None, :, :] \
            + self.right_target_feat_project(target_feat)[..., None, :]

        positions = atom_layout.convert(
            token_atoms_to_pseudo_beta,
            dense_atom_positions,
            layout_axes=(-3, -2),
        )

        dgram = template.dgram_from_positions(
            positions, self.dgram_features_config
        )

        dgram *= pair_mask[..., None]

        out += self.distogram_feat_project(dgram)

        return out

    def forward(
        self,
        target_feat: torch.Tensor,
        single: torch.Tensor,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        dense_atom_positions: torch.Tensor,
        token_atoms_to_pseudo_beta: atom_layout.GatherInfo
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            target_feat (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            dense_atom_positions (torch.Tensor): array of positions.
                [N_tokens, N_atom, 3] 
            pair_mask (torch.Tensor): pair mask
                [..., N_token, N_token]
            token_atoms_to_pseudo_beta (atom_layout.GatherInfo): Pseudo beta info for atom tokens.
        """
        pair += self._embed_features(
            dense_atom_positions, token_atoms_to_pseudo_beta, pair_mask, target_feat)

        # pairformer stack
        pair, single = self.confidence_pairformer(
            pair, single, pair_mask=pair_mask)

        # pLDDT
        plddt_logits = self.plddt_logits(self.plddt_logits_ln(single))
        plddt_logits = einops.rearrange(
            plddt_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=self.num_plddt_bins)
        predicted_lddt = torch.sum(
            torch.softmax(plddt_logits, dim=-1) * self.bin_centers, dim=-1
        )
        predicted_lddt = predicted_lddt * 100.0

        # Experimentally resolved
        experimentally_resolved_logits = self.experimentally_resolved_logits(
            self.experimentally_resolved_ln(single))
        experimentally_resolved_logits = einops.rearrange(
            experimentally_resolved_logits, '... (n_atom n_bins) -> ... n_atom n_bins', n_bins=2)

        predicted_experimentally_resolved = torch.softmax(
            experimentally_resolved_logits, dim=-1
        )[..., 1]

        return {
            'predicted_lddt': predicted_lddt,
            'predicted_experimentally_resolved': predicted_experimentally_resolved,
        }
