"""
SwiGLU kernel vs PyTorch, including the MLP's layout: x and gate are the two halves of each
row of the fused gate_up output, so for batch > 1 their rows are strided, not contiguous.

To run:
    pytest tests/test_swiglu.py -v
"""

import pytest
import torch

from kernels import swiglu, swiglu_out
from kernels.swiglu import swiglu_pytorch

DEVICE = torch.device("cuda")


@pytest.mark.parametrize("batch", [1, 3, 8])
@pytest.mark.parametrize("intermediate", [14336, 1000])
def test_swiglu_out_on_fused_halves(batch, intermediate):
    gate_up = torch.randn(batch, 1, 2 * intermediate, device=DEVICE, dtype=torch.float16)
    gate, up = torch.split(gate_up, [intermediate, intermediate], dim=-1)
    out = torch.empty(batch, 1, intermediate, device=DEVICE, dtype=torch.float16)
    out_ptr = out.data_ptr()

    swiglu_out(up, gate, out)

    torch.testing.assert_close(out.float(), swiglu_pytorch(up.float(), gate.float()), rtol=2e-3, atol=2e-3)
    assert out.data_ptr() == out_ptr  # written in place: no hidden copy of the output


def test_swiglu_contiguous():
    x = torch.randn(4, 1, 2048, device=DEVICE, dtype=torch.float16)
    gate = torch.randn_like(x)
    torch.testing.assert_close(swiglu(x, gate).float(), swiglu_pytorch(x.float(), gate.float()), rtol=2e-3, atol=2e-3)


def test_swiglu_rejects_non_contiguous_last_dim():
    x = torch.randn(4, 2048, 2, device=DEVICE, dtype=torch.float16)[..., 0]  # last-dim stride 2
    with pytest.raises(AssertionError):
        swiglu_out(x, x, torch.empty(4, 2048, device=DEVICE, dtype=torch.float16))
