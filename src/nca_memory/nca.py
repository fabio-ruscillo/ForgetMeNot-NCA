"""Local perception for the cellular memory."""

import torch
from torch import nn
from torch.nn import functional as F


class LocalPerception(nn.Module):
    """Read each cell and its fixed depthwise filter responses."""

    def __init__(self, channels=16):
        """Create the fixed filters for the given number of channels.

        Args:
            channels (int): Number of state channels per cell.
        """
        super().__init__()
        self.channels = channels

        average = torch.tensor([[1.0, 2.0, 1.0]]) #(1, 3)
        difference = torch.tensor([[-1.0, 0.0, 1.0]]) #(1, 3)
        # Preserve constant values along the two perpendicular axes.
        smoothing = average / average.sum()

        # Make the centered difference return +1 on a unit ramp.
        derivative = difference / difference.abs().sum()
        
        # Implemented Sobel filters to give each cell directional cues about how
        # neighbouring states vary along the three spatial axes.
        sobel_x = smoothing.T.unsqueeze(-1) * (smoothing.T @ derivative)
        sobel_y = sobel_x.transpose(1, 2)
        sobel_z = sobel_x.transpose(0, 2)

        # Implemented Laplacian filter to detect how much each cell
        # differs from the average of its neighbors.
        laplacian = average.T.unsqueeze(-1) * (average.T @ average)
        
        # Normalize the neighbors alone so the filter cancels constant states.
        laplacian[1, 1, 1] = 0.0
        laplacian = laplacian / laplacian.sum()
        laplacian[1, 1, 1] = -1.0

        filters = torch.stack((sobel_x, sobel_y, sobel_z, laplacian))
        
        # Give each channel its own copy of the four filters for depthwise convolution.     
        self.register_buffer(
            "perception_filters", 
            filters.unsqueeze(1).repeat(channels, 1, 1, 1, 1)
            )

    def forward(self, state):
        """Return each cell's state and local filter responses as input to the update MLP.

        Args:
            state (torch.Tensor): Batched state of shape (B, C, D, H, W).

        Returns:
            torch.Tensor: Local features of shape (B, 5 * C, D, H, W).
        """
        
        # Give edge cells a full neighborhood by wrapping across opposite faces,
        # to ensure periodic boundary conditions.
        padded_state = F.pad(state, (1, 1, 1, 1, 1, 1), mode="circular")
        
        # Using depthwise convolution to apply the fixed filters to
        # each channel of the state 
        responses = F.conv3d(
            padded_state, self.perception_filters,
            groups=self.channels
        )
        return torch.cat((state, responses), dim=1)


class CellularMemory(nn.Module):
    """Apply one synchronous update to the cellular memory.

    A shared MLP, maps each cell's state and four filter responses per
    channel to C increments (5 * C → 5 * C → C, ReLU).

    Args:
        channels (int): Number of state channels per cell (C).
        update_probability (float): Probability of updating each cell [0.0, 1.1].
    """

    def __init__(self, channels=16, update_probability=1.0):
        super().__init__()
        self.update_probability = update_probability
        self.perception = LocalPerception(channels=channels)
        self.update_rule = nn.Sequential(
            nn.Linear(5 * channels, 5 * channels, bias=False),
            nn.ReLU(),
            nn.Linear(5 * channels, channels, bias=False),
        )

    def forward(self, state, generator):
        """
        Args:
            state (torch.Tensor): Current memory state.
            generator (torch.Generator): Seeded generator on the same
            device as state. For reproducible masks, it has to be reused for each step.

        Returns:
            torch.Tensor: Updated memory.
        """
        # Each cell's features last, so Linear acts on each cell separately.
        features = self.perception(state).movedim(1, -1)
        delta = self.update_rule(features).movedim(-1, 1)

        # Use a stochastic mask to choose which cells update at this step.
        if self.update_probability < 1.0:
            # Pick once per cell so all its channels update together.
            mask_shape = (state.shape[0], 1, *state.shape[2:])
            update_mask = torch.rand(
                mask_shape, device=state.device, generator=generator
            ) < self.update_probability
            delta = delta * update_mask

        return state + delta
