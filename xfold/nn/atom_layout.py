import dataclasses
from typing import Any

import numpy as np
import torch


@dataclasses.dataclass(frozen=True)
class GatherInfo:
    """Gather indices to translate from one atom layout to another.

    All members are np or jnp ndarray (usually 1-dim or 2-dim) with the same
    shape, e.g.
    - [num_atoms]
    - [num_residues, max_atoms_per_residue]
    - [num_fragments, max_fragments_per_residue]

    Attributes:
      gather_idxs: np or jnp ndarray of int: gather indices into a flattened array
      gather_mask: np or jnp ndarray of bool: mask for resulting array
      input_shape: np or jnp ndarray of int: the shape of the unflattened input
        array
      shape: output shape. Just returns gather_idxs.shape
    """

    gather_idxs: torch.Tensor
    gather_mask: torch.Tensor
    input_shape: torch.Tensor

    def __post_init__(self):
        if self.gather_mask.shape != self.gather_idxs.shape:
            raise ValueError(
                'All arrays must have the same shape. Got\n'
                f'gather_idxs.shape = {self.gather_idxs.shape}\n'
                f'gather_mask.shape = {self.gather_mask.shape}\n'
            )

    def __getitem__(self, key: Any) -> 'GatherInfo':
        return GatherInfo(
            gather_idxs=self.gather_idxs[key],
            gather_mask=self.gather_mask[key],
            input_shape=self.input_shape,
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.gather_idxs.shape

    def as_dict(
        self,
        key_prefix: str | None = None,
    ) -> dict[str, torch.Tensor]:
        prefix = f'{key_prefix}:' if key_prefix else ''
        return {
            prefix + 'gather_idxs': self.gather_idxs,
            prefix + 'gather_mask': self.gather_mask,
            prefix + 'input_shape': self.input_shape,
        }

    @classmethod
    def from_dict(
        cls,
        d: dict[str, torch.Tensor],
        key_prefix: str | None = None,
    ) -> 'GatherInfo':
        """Creates GatherInfo from a given dictionary."""
        prefix = f'{key_prefix}:' if key_prefix else ''
        return cls(
            gather_idxs=d[prefix + 'gather_idxs'],
            gather_mask=d[prefix + 'gather_mask'],
            input_shape=d[prefix + 'input_shape'],
        )


def convert(
    gather_info: GatherInfo,
    arr: torch.Tensor,
    *,
    layout_axes: tuple[int, ...] = (0,),
) -> torch.Tensor:
    """Convert an array from one atom layout to another."""
    # Translate negative indices to the corresponding positives.
    layout_axes = tuple(i if i >= 0 else i + arr.ndim for i in layout_axes)

    # Ensure that layout_axes are continuous.
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1

    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f'layout_axes must be continuous. Got {layout_axes}.')
    layout_shape = arr.shape[layout_axes_begin:layout_axes_end]
    gather_info_input_shape = gather_info.input_shape.numpy()

    # Ensure that the layout shape is compatible
    # with the gather_info. I.e. the first axis size must be equal or greater
    # than the gather_info.input_shape, and all subsequent axes sizes must match.
    if (len(layout_shape) != gather_info_input_shape.size) or (
        isinstance(gather_info_input_shape, np.ndarray)
        and (
            (layout_shape[0] < gather_info_input_shape[0])
            or (np.any(layout_shape[1:] != gather_info_input_shape[1:]))
        )
    ):
        raise ValueError(
            'Input array layout axes are incompatible. You specified layout '
            f'axes {layout_axes} with an input array of shape {arr.shape}, but '
            f'the gather info expects shape {gather_info.input_shape}. '
            'Your first axis size must be equal or greater than the '
            'gather_info.input_shape, and all subsequent axes sizes must '
            'match.'
        )

    # Compute the shape of the input array with flattened layout.
    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    arr_flattened_shape = batch_shape + \
        (np.prod(layout_shape),) + features_shape

    # Flatten input array and perform the gather.
    arr_flattened = arr.reshape(arr_flattened_shape)
    if layout_axes_begin == 0:
        out_arr = arr_flattened[gather_info.gather_idxs, ...]
    elif layout_axes_begin == 1:
        out_arr = arr_flattened[:, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 2:
        out_arr = arr_flattened[:, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 3:
        out_arr = arr_flattened[:, :, :, gather_info.gather_idxs, ...]
    elif layout_axes_begin == 4:
        out_arr = arr_flattened[:, :, :, :, gather_info.gather_idxs, ...]
    else:
        raise ValueError(
            'Only 4 batch axes supported. If you need more, the code '
            'is easy to extend.'
        )

    # Broadcast the mask and apply it.
    broadcasted_mask_shape = (
        (1,) * len(batch_shape)
        + gather_info.gather_mask.shape
        + (1,) * len(features_shape)
    )
    out_arr *= gather_info.gather_mask.reshape(broadcasted_mask_shape)
    return out_arr
