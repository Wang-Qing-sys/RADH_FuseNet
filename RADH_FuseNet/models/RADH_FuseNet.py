
import torch
import torch.nn as nn

class RADHFuseNet(nn.Module):
    def __init__(self, modalities=5, hidden=96, classes=5):
        super().__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(1, hidden, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1)
            ) for _ in range(modalities)
        ])
        self.head = nn.Linear(hidden, classes)

    def forward(self, x):
        z=[]
        for i, layer in enumerate(self.features):
            z.append(layer(x[:,i]).squeeze(-1))
        fused=torch.stack(z,1).mean(1)
        return self.head(fused)
