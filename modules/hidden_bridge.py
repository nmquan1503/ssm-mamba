import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenBridge(nn.Module):
    def __init__(self, channels: int, states: int):
        super().__init__()
        self.channels = channels
        self.states = states
        self.fc1 = nn.Linear(channels * states, channels)
        self.fc2 = nn.Linear(channels, channels * states)
    
    def forward(self, hiddens: torch.Tensor):
        """
        Args: (batch_size, num_channels, state_dim)
        Returns: (batch_size, num_channels, state_dim)
        """
        batch_size = hiddens.size(0)
        hiddens = hiddens.reshape(batch_size, self.channels * self.states)
        hiddens = F.silu(self.fc1(hiddens))
        hiddens = self.fc2(hiddens)
        hiddens = hiddens.reshape(batch_size, self.channels, self.states)
        return hiddens
