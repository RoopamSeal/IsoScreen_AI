import torch
import torch.nn as nn


class HybridADMETModel(nn.Module):

    def __init__(
        self,
        gnn_dim=128,
        fp_dim=2048,
        transformer_dim=768,
        outputs=12
    ):

        super().__init__()

        total_dim = gnn_dim + fp_dim + transformer_dim

        self.layers = nn.Sequential(

            nn.Linear(total_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(512, 256),
            nn.ReLU(),

            nn.Linear(256, outputs)
        )

    def forward(self, gnn, fp, transformer):

        x = torch.cat([gnn, fp, transformer], dim=1)

        return self.layers(x)
