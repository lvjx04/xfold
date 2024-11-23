# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at

#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F


class AdaptiveLayerNorm(nn.Module):
    """
    Implements Algorithm 26 in AF3
    """

    def __init__(self, c_a: int = 768, c_s: int = 384) -> None:
        """
        Args:
            c_a (int, optional): the embedding dim of a(single feature aggregated atom info). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
        """
        super(AdaptiveLayerNorm, self).__init__()
        self.layernorm_a = nn.LayerNorm(
            c_a, elementwise_affine=False, bias=False)
        # The pytorch version should be newer than 2.1
        self.layernorm_s = nn.LayerNorm(c_s, bias=False)
        self.linear_s = nn.Linear(in_features=c_s, out_features=c_a)
        self.linear_nobias_s = nn.Linear(
            in_features=c_s, out_features=c_a, bias=False)

    def forward(self, a: torch.Tensor, s: torch.Tensor) -> torch.Tensor:
        """
        Args:
            a (torch.Tensor): the single feature aggregate per-atom representation
                [..., N_token, c_a]
            s (torch.Tensor): single embedding
                [..., N_token, c_s]

        Returns:
            torch.Tensor: the updated a from AdaLN
                [..., N_token, c_a]
        """
        a = self.layernorm_a(a)
        s = self.layernorm_s(s)
        a = torch.sigmoid(self.linear_s(s)) * a + self.linear_nobias_s(s)
        return a


class Transition(nn.Module):
    """
    Implements Algorithm 11 in AF3
    """

    def __init__(self, c_in: int, num_intermediate_factor: int = 4) -> None:
        """
        Args:
            c_in (int, optional): the input dimension.
            num_intermediate_factor (int, optional): factor by which c_in is multiplied to obtain hidden dimension.
        """
        super(Transition, self).__init__()
        self.num_intermediate_factor = num_intermediate_factor
        self.c_in = c_in
        self.input_layer_norm = nn.LayerNorm(c_in)
        self.transition1 = nn.Linear(
            in_features=c_in, out_features=self.num_intermediate_factor * c_in * 2, bias=False)
        self.transition2 = nn.Linear(
            in_features=self.num_intermediate_factor * c_in, out_features=c_in, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): the input tensor
                [..., c]

        Returns:
            torch.Tensor: the output tensor as the same shape of x
                [..., c]
        """
        x = self.input_layer_norm(x)
        x = self.transition1(x)
        a, b = torch.chunk(x, 2, dim=-1)
        c = F.silu(a) * b
        return self.transition2(c)
