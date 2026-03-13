import torch
import torch.nn as nn

from .mamba import Mamba
from .rms_norm import RMSNorm
from .hidden_bridge import HiddenBridge

class Block(nn.Module):
    def __init__(
        self,
        model_dim: int = 512,
        state_dim: int = 16,
        expansion_factor: int = 2,
        conv_kernel: int = 4,
        conv_bias: bool = False,
        bias: bool = False,
        delta_rank: int | str = "auto",
        delta_min: float = 0.001,
        delta_max: float = 0.1,
        delta_init: str = "random",
        delta_scale: float = 1.0,
        delta_init_floor: float = 1e-4,
        dropout_rate: float = 0.15,
        use_hidden_bridge: bool = False,
        device: str | None = None
    ):
        super().__init__()

        self.norm = RMSNorm(model_dim)

        self.ssm = Mamba(
            model_dim=model_dim,
            state_dim=state_dim,
            expansion_factor=expansion_factor,
            bias=bias,
            conv_kernel=conv_kernel,
            conv_bias=conv_bias,
            delta_rank=delta_rank,
            delta_min=delta_min,
            delta_max=delta_max,
            delta_init=delta_init,
            delta_scale=delta_scale,
            delta_init_floor=delta_init_floor,
            device=device
        )

        if use_hidden_bridge:
            self.hidden_bridge = HiddenBridge(
                channels=model_dim * expansion_factor
            )
        else:
            self.hidden_bridge = None

        self.dropout = nn.Dropout(dropout_rate)

    def forward(
        self,
        hidden_states: torch.Tensor,
        lengths: torch.Tensor | None = None,
        ssm_hiddens: torch.Tensor | None = None,
        use_cache: bool = False
    ):
        """
        Args:
            hidden_states: (batch_size, seq_len, model_dim)
            lengths: (batch_size,)
            ssm_hiddens: (batch_size, inner_dim, state_dim)
        Returns:
            (batch_size, seq_len, model_dim)
        """

        residual = hidden_states
        hidden_states = self.norm(hidden_states)

        if self.hidden_bridge is not None and ssm_hiddens is not None:
            ssm_hiddens = self.hidden_bridge(ssm_hiddens)

        hidden_states, last_ssm_hiddens = self.ssm(
            hidden_states,
            ssm_hiddens=ssm_hiddens,
            lengths=lengths,
            use_cache=use_cache
        )

        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states, last_ssm_hiddens

    def step(self, hidden_states: torch.Tensor):
        """
        Args: (batch_size, model_dim)
        Returns: (batch_size, model_dim)
        """

        residual = hidden_states
        hidden_states = self.norm(hidden_states)
        hidden_states = self.ssm.step(hidden_states)
        return residual + hidden_states