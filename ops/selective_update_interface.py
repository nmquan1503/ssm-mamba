import torch
import selective_update

from ..utils.tensor_utils import ensure_contiguous

class SelectiveUpdateFn:
    @staticmethod
    def apply(
        u: torch.Tensor, 
        A: torch.Tensor, 
        B: torch.Tensor, 
        C: torch.Tensor, 
        D: torch.Tensor, 
        delta: torch.Tensor, 
        delta_bias: torch.Tensor, 
        h: torch.Tensor
    ):
        """
        Args:
            u: (batch_size, num_channels)
            A: (num_channels, state_dim)
            B: (batch_size, state_dim)
            C: (batch_size, state_dim)
            D: (num_channels,)
            delta: (batch_size, num_channels)
            delta_bias: (num_channels,)
            h: (batch_size, num_channels, state_dim)
        Returns:
            out: (batch_size, num_channels)
            h: (batch_size, num_channels, state_dim)
        """

        u = ensure_contiguous(u)
        A = ensure_contiguous(A)
        B = ensure_contiguous(B)
        C = ensure_contiguous(C)
        D = ensure_contiguous(D)
        delta = ensure_contiguous(delta)
        delta_bias = ensure_contiguous(delta_bias)
        h = ensure_contiguous(h)

        out, h = selective_update.apply(u, A, B, C, D, delta, delta_bias, h)

        return out, h