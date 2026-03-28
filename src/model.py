"""
model.py — Neural network definition and weight serialization for FL-IDS.

Architecture:
    Input(17) -> Linear(128) -> ReLU -> Dropout(0.3)
              -> Linear(64)  -> ReLU -> Dropout(0.3)
              -> Linear(8)   [logits -- no softmax, handled by CrossEntropyLoss]

Design rationale:
    - Simple feedforward network: the research contribution is the FL evaluation,
      not the model architecture. Keeping it simple ensures differences between
      IID/non-IID results are due to data heterogeneity, not model complexity.
    - ~11K parameters = ~44 KB per update: small enough that communication cost
      is realistic for IIoT edge devices.
    - Dropout(0.3) for regularization, especially important when per-client
      datasets are small in the FL setting.

Usage:
    from model import IDSNet, get_model_params, set_model_params
    from model import count_parameters, compute_model_size_bytes, create_model
"""

import torch
import torch.nn as nn
import numpy as np


# ======================================================================
#  NETWORK DEFINITION
# ======================================================================


class IDSNet(nn.Module):
    """
    Feedforward neural network for 8-class IIoT intrusion detection.

    Architecture:
        Input(num_features) -> 128 -> ReLU -> Dropout -> 64 -> ReLU -> Dropout -> 8

    Parameters
    ----------
    num_features : int
        Number of input features (default: 17 from Table 7 selection).
    num_classes : int
        Number of output classes (default: 8 attack categories).
    hidden1 : int
        Units in first hidden layer (default: 128).
    hidden2 : int
        Units in second hidden layer (default: 64).
    dropout_rate : float
        Dropout probability (default: 0.3).

    Notes
    -----
    - forward() returns raw logits (no softmax). PyTorch CrossEntropyLoss
      applies LogSoftmax internally, so adding Softmax here would be wrong.
    - For inference probabilities, apply torch.softmax(logits, dim=1).
    """

    def __init__(
        self,
        num_features=17,
        num_classes=8,
        hidden1=128,
        hidden2=64,
        dropout_rate=0.3,
    ):
        super().__init__()

        self.network = nn.Sequential(
            # Hidden layer 1: input -> 128
            nn.Linear(num_features, hidden1),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Hidden layer 2: 128 -> 64
            nn.Linear(hidden1, hidden2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            # Output layer: 64 -> num_classes (raw logits)
            nn.Linear(hidden2, num_classes),
        )

        # Kaiming (He) initialization -- good default for ReLU networks
        self._init_weights()

    def _init_weights(self):
        """Apply Kaiming initialization to all linear layers."""
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
                nn.init.zeros_(module.bias)

    def forward(self, x):
        """
        Forward pass -- returns raw logits (NOT probabilities).

        Parameters
        ----------
        x : torch.Tensor, shape (batch_size, num_features)

        Returns
        -------
        logits : torch.Tensor, shape (batch_size, num_classes)
        """
        return self.network(x)


# ======================================================================
#  WEIGHT SERIALIZATION FOR FEDAVG
# ======================================================================
#
#  These functions convert between a live PyTorch model and a list of
#  NumPy arrays. This is the "serialization" step that simulates sending
#  model weights over a network in federated learning.
#
#  In a real FL deployment, the client would serialize weights to bytes,
#  send them over the network, and the server would deserialize. Here we
#  skip the byte encoding and work directly with NumPy arrays.
# ======================================================================


def get_model_params(model):
    """
    Extract model parameters as a list of NumPy arrays.

    Simulates a client packaging its trained weights to send to the server.

    Parameters
    ----------
    model : IDSNet (or any nn.Module)

    Returns
    -------
    params : list[np.ndarray]
        One array per parameter tensor. For IDSNet: 6 arrays
        [fc1.weight, fc1.bias, fc2.weight, fc2.bias, fc3.weight, fc3.bias].
    """
    return [val.cpu().detach().numpy().copy() for val in model.state_dict().values()]


def set_model_params(model, params):
    """
    Load a list of NumPy arrays into a model's parameters.

    Simulates a client receiving the global model weights from the server.

    Parameters
    ----------
    model : IDSNet (or any nn.Module)
    params : list[np.ndarray]
        Must match the shape and order of model.state_dict().
    """
    state_dict = model.state_dict()
    for (key, _), new_param in zip(state_dict.items(), params):
        state_dict[key] = torch.tensor(new_param)
    model.load_state_dict(state_dict)


# ======================================================================
#  UTILITY FUNCTIONS
# ======================================================================


def count_parameters(model):
    """
    Count total and trainable parameters.

    Returns
    -------
    total : int
        Total number of parameters.
    trainable : int
        Number of parameters with requires_grad=True.
    """
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def compute_model_size_bytes(model, dtype_bytes=4):
    """
    Calculate model size in bytes (for communication cost tracking).

    In FedAvg, each participating client sends one full model update
    per round (upload), and receives one global model per round (download).
    Total bytes per round = model_size x participating_clients x 2.

    Parameters
    ----------
    model : IDSNet (or any nn.Module)
    dtype_bytes : int
        Bytes per parameter (default: 4 for float32).

    Returns
    -------
    size_bytes : int
    """
    total, _ = count_parameters(model)
    return total * dtype_bytes


def create_model(num_features=17, num_classes=8, device="cpu"):
    """
    Factory function to create a fresh IDSNet on the given device.

    Ensures all clients and the server use the same constructor call.

    Parameters
    ----------
    num_features : int
    num_classes : int
    device : str or torch.device
        "cpu", "cuda", or "mps"

    Returns
    -------
    model : IDSNet
    """
    model = IDSNet(num_features=num_features, num_classes=num_classes)
    return model.to(device)
