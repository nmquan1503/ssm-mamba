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
        return_last_hidden: bool = False
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
        
        Returns:
            If return_last_hidden is True:
                out: (batch_size, num_channels, seq_len)
                h: (batch_size, num_channels, state_dim)
            
            If return_last_hidden is False:
                out: (batch_size, num_channels, seq_len)
                
        """
        u = ensure_contiguous(u);
        A = ensure_contiguous(A)
        B = ensure_contiguous(B)
        C = ensure_contiguous(C)
        D = ensure_contiguous(D)
        delta = ensure_contiguous(delta)
        delta_bias = ensure_contiguous(delta_bias)

        out, h = selective_scan.forward(u, A, B, C, D, delta, delta_bias)

        if any(ctx.needs_input_grad[:7]):
            ctx.save_for_backward(u, A, B, C, D, delta, delta_bias, h)
        
        if return_last_hidden:
            return out, h[:, :, -1, 1::2]
        
        return out

    @staticmethod
    def backward(ctx, dout):
        if not hasattr(ctx, "saved_tensors") or len(ctx.saved_tensors) == 0:
            return (None,) * 8
        
        dout = ensure_contiguous(dout)
        u, A, B, C, D, delta, delta_bias, h = ctx.saved_tensors
        du, dA, dB, dC, dD, ddelta,ddelta_bias = selective_scan.backward(
            u, 
            A, B, C, D, 
            delta, delta_bias, 
            h, dout
        )
        return du, dA, dB, dC, dD, ddelta, ddelta_bias, None