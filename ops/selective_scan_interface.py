import torch
import selective_scan
from typing import Tuple

from ..utils.tensor_utils import ensure_contiguous

class SelectiveScanFn(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx, 
        u: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor, 
        D: torch.Tensor, 
        delta: torch.Tensor, 
        delta_bias: torch.Tensor,
        h_init: torch.Tensor | None = None,
        length: torch.Tensor | None = None,
        padding_size: str = "left"
    ) -> Tuple[torch.Tensor, torch.Tensor] | torch.Tensor:
        """
        Args:
            u: (batch_size, num_channels, seq_len)
            A: (num_channels, state_dim)
            B: (batch_size, state_dim, seq_len)
            C: (batch_size, state_dim, seq_len)
            D: (num_channels,)
            delta: (batch_size, num_channels, seq_len)
            delta_bias: (num_channels,)
            h_init: (batch_size, num_channels, state_dim)
            length: (batch_size,)
        
        Returns:
            out: (batch_size, num_channels, seq_len)
            h_last: (batch_size, num_channels, state_dim)

                
        """
        u, A, B, C, D, delta, delta_bias, h_init, length = map(
            ensure_contiguous,
            (u, A, B, C, D, delta, delta_bias, h_init, length)
        )

        out, h = selective_scan.forward(
            u, 
            A, B, C, D, 
            delta, delta_bias, 
            h_init, length, padding_size
        )

        if any(ctx.needs_input_grad):
            ctx.save_for_backward(
                u, 
                A, B, C, D, delta, delta_bias, 
                h, h_init, length, padding_size
            )
        
        return out, h[..., -1, 1::2]

    @staticmethod
    def backward(ctx, dout, dh_last):
        if not hasattr(ctx, "saved_tensors") or len(ctx.saved_tensors) == 0:
            return (None,) * 8
        
        dout = ensure_contiguous(dout)
        dh_last = ensure_contiguous(dh_last)

        u, A, B, C, D, delta, delta_bias, h, h_init, length, padding_side = ctx.saved_tensors
        du, dA, dB, dC, dD, ddelta, ddelta_bias, dh_init = selective_scan.backward(
            u, 
            A, B, C, D, 
            delta, delta_bias, 
            h, h_init, 
            dout, dh_last, length, padding_side
        )

        return du, dA, dB, dC, dD, ddelta, ddelta_bias, dh_init, None, None