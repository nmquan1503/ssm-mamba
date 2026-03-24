from dataclasses import dataclass
from typing import Literal
import torch
import torch.nn as nn

from ..modules import Block, RMSNorm

@dataclass
class Seq2SeqModelConfig:
    vocab_size: int = 32000
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 2

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

    num_layers: int = 4
    tie_embeddings: bool = True

    dropout_rate: float = 0.15

    device: str | None = None


class Seq2SeqModel(nn.Module):
    def __init__(self, config: Seq2SeqModelConfig | None = None):
        super().__init__()
        
        if config is None:
            config = Seq2SeqModelConfig()

        self.config = config
        
        self.embedding = nn.Embedding(config.vocab_size, config.model_dim)

        self.encoder_layers = nn.ModuleList([
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
                device=config.device,
            )
            for _ in range(config.num_layers)
        ])
        
        self.decoder_layers = nn.ModuleList([
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
                device=config.device,
            )
            for _ in range(config.num_layers)
        ])

        self.norm = RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tgt_embedding.weight
    
    def forward(
        self,
        input_ids: torch.Tensor,
        lengths: torch.Tensor,
        tgt_ids: torch.Tensor,
        use_cache: bool = False
    ):
        """
        Args:
            input_ids: (batch_size, seq_len)
            lengths: (batch_size,)
            tgt_ids: (batch_size, out_seq_len)
        
        Returns: (batch_size, out_seq_len, tgt_vocab_size)
        """
        enc_hidden_states = self.embedding(input_ids)
        dec_hidden_states = self.embedding(tgt_ids)

        if use_cache:
            dec_input_lengths = torch.full(
                (tgt_ids.size(0),),
                fill_value=tgt_ids.size(1),
                device=tgt_ids.device,
                dtype=torch.long
            )
        else:
            dec_input_lengths = None

        for enc_layer, dec_layer in zip(self.encoder_layers, self.decoder_layers):
            enc_hidden_states, enc_ssm_hiddens = enc_layer(enc_hidden_states, lengths=lengths)
            dec_hidden_states, _ = dec_layer(
                dec_hidden_states,
                ssm_hiddens=enc_ssm_hiddens,
                use_cache=use_cache,
                lengths=dec_input_lengths
            )
        
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        return logits

    def step(self, input_ids: torch.Tensor):
        """
        Args: (batch_size,)

        Returns: (batch_size, tgt_vocab_size)
        """
        hidden_states = self.tgt_embedding(input_ids)
        for layer in self.decoder_layers:
            hidden_states = layer.step(hidden_states)
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)
        return logits

    def generate(self, input_ids: torch.Tensor, max_new_tokens=100, temperature=1.0):
        """
        Args: (batch_size, seq_len)
        Returns: (batch_size, out_seq_len)
        """
        with torch.no_grad():
            batch_size = input_ids.size(0)
            device = input_ids.device
            bos_id = self.config.bos_token_id
            eos_id = self.config.eos_token_id
            pad_id = self.config.pad_token_id

            lengths = (input_ids != pad_id).sum(dim=1)

            seq_ids = torch.full((batch_size, 1), bos_id, dtype=torch.long, device=device)
            finished = torch.zeros(batch_size, dtype=torch.bool, device=device)
            logits = self.forward(input_ids, lengths, seq_ids, use_cache=True)
            logits = logits[:, -1, :]

            for _ in range(max_new_tokens):
                if temperature != 1.0:
                    logits = logits / temperature
                probs = torch.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

                seq_ids = torch.cat([seq_ids, next_token],dim=1)
                finished |= (next_token.squeeze(1) == eos_id)
                if finished.all():
                    break

                logits = self.step(next_token.squeeze(1))

            eos_mask = (seq_ids == eos_id)
            first_eos = eos_mask.float().cumsum(dim=1) >= 1
            seq_ids = torch.where(first_eos, eos_id, seq_ids)
            return seq_ids