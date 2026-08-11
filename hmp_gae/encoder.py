# hmp_gae/encoder.py
# Two-stage (node -> hyperedge -> node) HMP encoder, implementing the paper's
# equations (15) and (16):
#
#   E^{(l)}   = sigma( D_E^{-1}  H^T  Z^{(l)}  W_E^{(l)} )       # node -> edge
#   Z^{(l+1)} = sigma( D_V^{-1}  H    E^{(l)}  W_V^{(l)} )       # edge -> node
#
# The encoder is intentionally thin: two linear projections per layer plus a
# ReLU. Keeping the formulation explicit makes it easy to map every tensor
# operation back to the paper equations.

from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hypergraph import apply_diag_inv


class HMPLayer(nn.Module):
    """
    Single HMP layer with a node-side and hyperedge-side linear projection.
    """

    def __init__(
        self,
        in_dim: int,
        edge_dim: int,
        out_dim: int,
        dropout: float = 0.0,
        residual: bool = False,
        activate_output: bool = True,
    ):
        super().__init__()
        self.W_E = nn.Linear(in_dim, edge_dim, bias=False)
        self.W_V = nn.Linear(edge_dim, out_dim, bias=False)
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.residual = bool(residual)
        self.activate_output = bool(activate_output)
        self.W_skip = (
            nn.Linear(in_dim, out_dim, bias=False) if self.residual else None
        )
        self.norm = nn.LayerNorm(out_dim) if self.residual else None
        # Kaiming init (linear w/ ReLU). nn.Linear already uses
        # kaiming_uniform_ by default; keep the standard init.

    def forward(
        self,
        Z: torch.Tensor,
        H: torch.Tensor,
        D_V_inv: torch.Tensor,
        D_E_inv: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # node -> hyperedge: E = sigma(D_E^{-1} H^T (Z W_E))
        msg_e = H.t() @ self.W_E(Z)                  # (M, in_dim) @ (in_dim, edge_dim) was done by W_E
        E = apply_diag_inv(D_E_inv, msg_e)            # (M, edge_dim)
        E = F.relu(E)
        E = self.dropout(E)

        # hyperedge -> node: Z' = sigma(D_V^{-1} H (E W_V))
        msg_v = H @ self.W_V(E)                       # (N, out_dim)
        Z_new = apply_diag_inv(D_V_inv, msg_v)
        if self.residual:
            # V8: preserve client-specific content on the tiny N=7 graph while
            # still admitting a bounded-depth hypergraph message. (The
            # pre-V8 pure-message path — residual=False — is retained as an
            # option but no live mode uses it since 2026-08-11.)
            Z_new = self.norm(Z_new + self.W_skip(Z))
        if self.activate_output:
            Z_new = F.relu(Z_new)
        return Z_new, E


class HMPEncoder(nn.Module):
    """
    Stack of L HMP layers mapping eta (N, eta_dim) to Z (N, latent_dim).

    Layer sizes: eta_dim -> hidden_dim -> ... -> latent_dim. When L == 1 the
    encoder has a single (eta_dim -> latent_dim) layer.
    """

    def __init__(
        self,
        eta_dim: int,
        hidden_dim: int,
        latent_dim: int,
        num_layers: int = 2,
        dropout: float = 0.0,
        residual: bool = False,
        signed_output: bool = False,
    ):
        super().__init__()
        self.num_layers = int(num_layers)
        dims = [int(eta_dim)]
        for _ in range(self.num_layers - 1):
            dims.append(int(hidden_dim))
        dims.append(int(latent_dim))
        # dims has length L + 1
        self.layers = nn.ModuleList([
            HMPLayer(
                in_dim=dims[i],
                edge_dim=max(dims[i], dims[i + 1]),
                out_dim=dims[i + 1],
                dropout=dropout,
                residual=residual,
                # A signed final latent lets V8's cosine decoder represent
                # both attraction and repulsion.  Hidden layers stay ReLU.
                activate_output=not (signed_output and i == self.num_layers - 1),
            )
            for i in range(self.num_layers)
        ])

    def forward(
        self,
        eta: torch.Tensor,
        H: torch.Tensor,
        D_V_inv: torch.Tensor,
        D_E_inv: torch.Tensor,
    ) -> torch.Tensor:
        Z = eta
        for layer in self.layers:
            Z, _ = layer(Z, H, D_V_inv, D_E_inv)
        return Z
