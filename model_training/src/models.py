import torch
from torch import nn


class SimpleMLPFusion(nn.Module):
    """PF-MLP regression from four normalized receiver features."""

    def __init__(self, input_dim=4, hidden_dims=None, output_dim=2, dropout=0.0):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [128, 64]
        layers = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class SignalFusionNetLite(nn.Module):
    """SF-CNN regression from two normalized amplitude channels."""

    def __init__(self, Nc=3276, Ns=1, conv_channels=None, fc_dims=None,
                 pool_size=4, dropout=0.0):
        super().__init__()
        if Ns != 1:
            raise ValueError("SF-CNN requires Ns=1.")
        self.Nc = Nc
        self.Ns = Ns
        self.signal_len = Nc * Ns
        if conv_channels is None:
            conv_channels = [64, 128, 256, 256, 256]
        if fc_dims is None:
            fc_dims = [256, 128, 84]
        self.conv_channels = conv_channels
        self.fc_dims = fc_dims
        self.dropout = dropout

        conv_layers = []
        in_channels = 2
        for i, out_channels in enumerate(conv_channels):
            kernel_size, stride, padding = (7, 2, 3) if i < 3 else (3, 1, 1)
            conv_layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(0.1),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*conv_layers)
        self.pool = nn.AvgPool1d(pool_size)

        # Flattened size from a dummy pass (in train mode, so it also touches the BatchNorm stats)
        with torch.no_grad():
            dummy_input = torch.zeros(1, 2, self.signal_len)
            pooled_features = self.pool(self.features(dummy_input))
            flat_dim = pooled_features.view(1, -1).size(1)

        fc_layers = []
        prev_dim = flat_dim
        for fc_dim in fc_dims:
            fc_layers.extend([
                nn.Linear(prev_dim, fc_dim),
                nn.BatchNorm1d(fc_dim),
                nn.LeakyReLU(0.1)
            ])
            prev_dim = fc_dim
        fc_layers.extend([nn.Linear(prev_dim, 2), nn.Tanh()])
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        batch_size, channels, ns, nc = x.size()
        x = x.view(batch_size, channels, ns * nc)
        features = self.pool(self.features(x))
        return self.classifier(features.view(features.size(0), -1))


def create_model(config):
    """Build either paper model from its YAML configuration."""
    name = config['model']['name']
    params = config['model']['params']
    if name == 'SimpleMLPFusion':
        return SimpleMLPFusion(**params)
    if name == 'SignalFusionNetLite':
        return SignalFusionNetLite(**params)
    raise ValueError(f"Unsupported model: {name}")
