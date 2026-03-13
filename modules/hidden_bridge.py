import torch
import torch.nn as nn
import torch.nn.functional as F

class HiddenBridge(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        self.l1 = nn.Linear(channels, channels)
        self.l2 = nn.Linear(channels, channels)
        self.l3 = nn.Linear(channels, channels)

    def _linear(self, hiddens: torch.Tensor, layer: nn.Linear):
        hiddens = hiddens.transpose(1, 2)
        hiddens = layer(hiddens)
        hiddens = hiddens.transpose(1, 2)
        return hiddens

    def _active(self, hiddens: torch.Tensor):
        s1 = F.relu(hiddens[:, :, 0]) 
        mask = (s1 > 0).unsqueeze(-1) 
        return hiddens * mask
    
    def forward(self, hiddens: torch.Tensor):
        """
        Args: (batch_size, num_channels, state_dim)
        Returns: (batch_size, num_channels, state_dim)
        """
        hiddens = self._linear(hiddens, self.l1)
        hiddens = self._active(hiddens)
        hiddens = self._linear(hiddens, self.l2)
        hiddens = self._active(hiddens)
        hiddens = self._linear(hiddens, self.l3)
        return hiddens
