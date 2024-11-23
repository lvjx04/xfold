import torch
import torch.nn as nn

from xfold.nn.pairformer import EvoformerBlock, PairformerBlock
from xfold.nn.head import DistogramHead, ConfidenceHead


class Evoformer(nn.Module):
    def __init__(self, msa_channel: int = 64):
        super(Evoformer, self).__init__()

        self.msa_channel = msa_channel
        self.pairformer_num_layer = 48

        self.msa_activations = nn.Linear(777, self.msa_channel)
        self.extra_msa_target_feat = nn.Linear(777, self.msa_channel)
        self.msa_stack = nn.ModuleList([EvoformerBlock() for _ in range(4)])

        self.trunk_pairformer = nn.ModuleList(
            [PairformerBlock(with_single=True) for _ in range(self.pairformer_num_layer)])

    def _relative_encoding(self):
        pass

    def _seq_pair_embedding(self,
                            token_features: features.TokenFeatures,
                            target_feat: torch.Tensor):
        pass

    def _embed_bonds(self):
        pass

    def _embed_template_pair(self):
        pass

    def _embed_process_msa(self):
        pass

    def forward(
        self,
        batch: dict[str, torch.Tensor],
        prev: dict[str, torch.Tensor],
        target_feat: torch.Tensor
    ) -> dict[str, torch.Tensor]:

        num_residues = target_feat.shape[0]

        pair_activations, pair_mask = self._seq_pair_embedding(
            batch.token_features, target_feat
        )

        for pairformer_b in self.trunk_pairformer:
            pair_activations, single_activations = pairformer_b(
                pair_activations, pair_mask, single_activations, seq_mask)

        output = {
            'single': single_activations,
            'pair': pair_activations,
            'target_feat': target_feat,
        }

        return output


class AlphaFold3(nn.Module):
    def __init__(self):
        super(AlphaFold3, self).__init__()

        self.evoformer_pair_channel = 128
        self.evoformer_seq_channel = 384

        self.evoformer = Evoformer()

        self.distogram_head = DistogramHead()
        self.confidence_head = ConfidenceHead()

    def forward(self, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:

        num_res = batch.num_res

        embeddings = {
            'pair': torch.zeros(
                [num_res, num_res, self.evoformer_pair_channel], device=target_feat.device,
                dtype=torch.float32,
            ),
            'single': torch.zeros(
                [num_res, self.evoformer_seq_channel], dtype=torch.float32, device=target_feat.device,
            ),
            'target_feat': target_feat,  # type: ignore
        }

        return {
            'diffusion_samples': samples,
            'distogram': distogram,
            **confidence_output,
        }
