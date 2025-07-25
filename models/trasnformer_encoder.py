import torch
import torch.nn as nn
from einops import rearrange

class TransformerEncoder(nn.Module):
    def __init__(self, input_dims, output_dims, num_heads=8, depth=6, hidden_dim=256, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.input_projection = nn.Linear(input_dims, hidden_dim)
        self.positional_encoding = PositionalEncoding(hidden_dim)
        self.transformer_layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_dim,
                nhead=num_heads,
                dim_feedforward=hidden_dim * 4,
                dropout=dropout,
                activation='gelu'
            ) for _ in range(depth)
        ])
        self.output_projection = nn.Linear(hidden_dim, output_dims)

        # Define component_dims as the output dimension
        self.component_dims = output_dims

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, x):
        # x shape: (batch_size, seq_len, input_dims)
        x = self.input_projection(x)  # Project input to hidden_dim
        x = self.positional_encoding(x)  # Add positional encoding
        x = rearrange(x, 'b t d -> t b d')  # Transformer expects (seq_len, batch_size, hidden_dim)
        for layer in self.transformer_layers:
            x = layer(x)
        x = rearrange(x, 't b d -> b t d')  # Back to (batch_size, seq_len, hidden_dim)
        x = self.output_projection(x)  # Project to output_dims
        return x

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-torch.log(torch.tensor(10000.0)) / d_model))
        self.encoding[:, 0::2] = torch.sin(position * div_term)
        self.encoding[:, 1::2] = torch.cos(position * div_term)
        self.encoding = self.encoding.unsqueeze(0)  # Add batch dimension

    def forward(self, x):
        seq_len = x.size(1)
        return x + self.encoding[:, :seq_len, :].to(x.device)