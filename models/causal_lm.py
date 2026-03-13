from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn

from ..modules import RMSNorm, Block

@dataclass
class CausalLMConfig:
    vocab_size: int = 32000
    model_dim: int = 512
    state_dim: int = 16
    expansion_factor: int = 2
    conv_kernel: int = 4
    conv_bias: bool = False
    bias: bool = False
    delta_rank: int | str = "auto"
    delta_min: float = 0.001
    delta_max: float = 0.1
    delta_init: Literal["random", "constant"] = "random"
    delta_scale: float = 1.0
    delta_init_floor: float = 1e-4
    dropout_rate: float = 0.15
    num_layers: int = 5
    tie_embeddings: bool = True
    device: str | None = None
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

    
class CausalLM(nn.Module):
    def __init__(self, config: CausalLMConfig | None = None):
        super().__init__()

        if config is None:
            config = CausalLMConfig()

        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)
        self.layers = nn.ModuleList([
            Block(
                model_dim=config.model_dim,
                state_dim=config.state_dim,
                expansion_factor=config.expansion_factor,
                conv_kernel=config.conv_kernel,
                conv_bias=config.conv_bias,
                bias=config.bias,
                delta_rank=config.delta_rank,
                delta_min=config.delta_min,
                delta_max=config.delta_max,
                delta_init=config.delta_init,
                delta_scale=config.delta_scale,
                delta_init_floor=config.delta_init_floor,
                dropout_rate=config.dropout_rate,
                use_hidden_bridge=False,
                device=config.device
            )
            for _ in range(config.num_layers)
        ])
        self.norm = RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embedding.weight
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        lengths: torch.Tensor | None = None,
        use_cache: bool = False
    ):
        """
        Args: (batch_size, seq_len)
        Returns: (batch_size, seq_len, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states, _ = layer(
                hidden_states, 
                lengths=lengths, 
                use_cache=use_cache
            )
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def step(self, input_ids: torch.Tensor):
        """
        Args: (batch_size,)
        Returns: (batch_size, vocab_size)
        """
        hidden_states = self.embedding(input_ids)
        for layer in self.layers:
            hidden_states = layer.step(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=100, temperature=1.0):
        """
        Args: (batch_size, seq_len)
        Returns: (batch_size, seq_len + new_token)
        """
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            eos_id = self.config.eos_token_id
            pad_id = self.config.pad_token_id

            lengths = (input_ids != pad_id).sum(dim=1)
            last_indices = lengths - 1

            logits = self.forward(input_ids, lengths, use_cache=True)
            logits = logits[torch.arange(batch_size, device=device), last_indices]
            
            seq_ids = input_ids
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            
            for _ in range(max_new_tokens):
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                seq_ids = torch.cat([seq_ids, next_token], dim=1)
                finished |= (next_token.squeeze(1) == eos_id)
                if finished.all():
                    break

                logits = self.step(next_token.squeeze(1))
            
            eos_mask = (seq_ids == eos_id)
            first_eos = eos_mask.float().cumsum(dim=1) >= 1
            seq_ids = torch.where(first_eos, eos_id, seq_ids)
            return seq_ids