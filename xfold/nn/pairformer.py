from typing import Optional

import torch
import torch.nn as nn

from xfold.nn.primitives import Transition
from xfold.nn.triangle_multiplication import TriangleMultiplication
from xfold.nn.attention import GridSelfAttention, AttentionPairBias


class PairformerBlock(nn.Module):
    """Implements Algorithm 17 [Line2-Line8] in AF3
    Ref to: openfold/model/evoformer.py and protenix/model/modules/pairformer.py
    """

    def __init__(
        self,
        n_heads: int = 16,
        c_pair: int = 128,
        c_single: int = 384,
        c_hidden_mul: int = 128,
        n_heads_pair: int = 4,
        with_single: bool = True,
    ) -> None:
        """
        Args:
            n_heads (int, optional): number of head [for AttentionPairBias]. Defaults to 16.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_hidden_mul (int, optional): hidden dim [for TriangleMultiplicationOutgoing].
                Defaults to 128.
            n_heads_pair (int, optional): number of head [for TriangleAttention]. Defaults to 4.
        """
        super(PairformerBlock, self).__init__()
        self.n_heads = n_heads
        self.with_single = with_single
        self.triangle_multiplication_outgoing = TriangleMultiplication(
            c_pair=c_pair, c_hidden=c_hidden_mul, _outgoing=True
        )
        self.triangle_multiplication_incoming = TriangleMultiplication(
            c_pair=c_pair, c_hidden=c_hidden_mul, _outgoing=False)
        self.pair_attention1 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=False
        )
        self.pair_attention2 = GridSelfAttention(
            c_pair=c_pair, num_head=n_heads_pair, transpose=True
        )
        self.pair_transition = Transition(c_in=c_pair)
        self.c_single = c_single
        if self.with_single is True:
            self.single_pair_logits_norm = nn.LayerNorm(c_pair)
            self.single_pair_logits_projection = nn.Linear(
                c_pair, n_heads, bias=False)
            self.single_attention_ = AttentionPairBias(
                c_single=c_single, num_head=n_heads, use_single_cond=False)
            self.single_transition = Transition(c_in=self.c_single)

    def forward(
        self,
        pair: torch.Tensor,
        pair_mask: torch.Tensor,
        single: Optional[torch.Tensor] = None,
        seq_mask: Optional[torch.Tensor] = None,
    ) -> tuple[Optional[torch.Tensor], torch.Tensor]:
        """
        Forward pass of the PairformerBlock.

        Args:
            pair (torch.Tensor): [..., N_token, N_token, c_pair]
            pair_mask (torch.Tensor): [..., N_token, N_token]
            single (torch.Tensor, optional): [..., N_token, c_single]
            seq_mask (torch.Tensor, optional): [..., N_token]

        Returns:
            tuple[torch.Tensor, Optional[torch.Tensor]]: pair, single
        """
        pair += self.triangle_multiplication_outgoing(pair, mask=pair_mask)
        pair += self.triangle_multiplication_incoming(pair, mask=pair_mask)
        pair += self.pair_attention1(pair, mask=pair_mask)
        pair += self.pair_attention2(pair, mask=pair_mask)

        pair += self.pair_transition(pair)
        if self.with_single is True:
            pair_logits = self.single_pair_logits_projection(
                self.single_pair_logits_norm(pair))

            pair_logits = pair_logits.permute(2, 0, 1)

            single += self.single_attention_(single,
                                             seq_mask,
                                             pair_logits=pair_logits)

            single += self.single_transition(single)
            return pair, single
        return pair
