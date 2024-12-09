import torch

import triton
import triton.language as tl

from xfold.fastnn import config as fastnn_config


@triton.jit
def silu_mul_fused_kernel(a_ptr,
                          b_ptr,
                          output_ptr,
                          n_elements,
                          BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(a_ptr + offsets, mask=mask).to(tl.float32)
    y = tl.load(b_ptr + offsets, mask=mask)
    output = x * tl.sigmoid(x) * y
    tl.store(output_ptr + offsets, output, mask=mask)


def triton_silu_mul(a, b):

    a = a.contiguous()
    b = b.contiguous()

    c = torch.empty_like(a)
    n_elements = c.numel()

    def grid(meta): return (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    silu_mul_fused_kernel[grid](a, b, c, n_elements, BLOCK_SIZE=1024)
    return c


def silu_mul(a, b):

    def _silu_mul(a, b):
        return torch.nn.functional.silu(a) * b

    _silu_mul_func = _silu_mul
    if fastnn_config.silu_implementation == "triton" and a.numel() > fastnn_config.silu_threshold:
        _silu_mul_func = torch.compile(_silu_mul)

    return _silu_mul_func(a, b)
