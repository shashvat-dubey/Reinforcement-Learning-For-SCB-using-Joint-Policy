import torch
import torch.nn as nn


class SCBValueHead(nn.Module):
    """
    Shared critic/value network for the SCB RL system.

    Input:
        global_embedding from SCBEncoder

    Output:
        V(s), a scalar estimate of the value of the current state.

    The critic is shared across all three policies.
    """

    def __init__(
        self,
        hidden_dim: int = 128,
        value_hidden_dim: int = 128,
    ):
        super().__init__()

        self.hidden_dim = hidden_dim

        self.network = nn.Sequential(
            nn.Linear(hidden_dim, value_hidden_dim),
            nn.ReLU(),
            nn.Linear(value_hidden_dim, 1),
        )

    def forward(
        self,
        global_embedding: torch.Tensor,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        global_embedding:
            Shape [hidden_dim] or [batch_size, hidden_dim].

        Returns
        -------
        torch.Tensor
            Shape [1] for a single state or [batch_size] for a batch.
        """

        if global_embedding.ndim == 1:
            if global_embedding.shape[0] != self.hidden_dim:
                raise ValueError(
                    f"Expected embedding dimension {self.hidden_dim}, "
                    f"got {global_embedding.shape[0]}."
                )

            value = self.network(
                global_embedding.unsqueeze(0)
            )

            return value.squeeze(0)

        if global_embedding.ndim == 2:
            if global_embedding.shape[1] != self.hidden_dim:
                raise ValueError(
                    f"Expected embedding dimension {self.hidden_dim}, "
                    f"got {global_embedding.shape[1]}."
                )

            value = self.network(
                global_embedding
            )

            return value.squeeze(-1)

        raise ValueError(
            "global_embedding must have shape "
            "[hidden_dim] or [batch_size, hidden_dim]."
        )