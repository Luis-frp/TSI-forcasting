import torch
import torch.nn as nn
import math


class TemporalTransformer(nn.Module):
    """
    Transformer Temporal para captura de dependências de longo alcance em séries temporais.
    Substitui as convoluções dilatadas por mecanismos de atenção.
    """
    def __init__(self, input_dims, output_dims, num_heads=4, num_layers=2, max_len=3000, dropout=0.1):
        """
        Args:
            input_dims (int): Dimensão de entrada (número de canais).
            output_dims (int): Dimensão de saída (número de canais).
            num_heads (int): Número de cabeças de atenção.
            num_layers (int): Número de camadas do Transformer.
            max_len (int): Comprimento máximo da série temporal.
            dropout (float): Taxa de dropout para regularização.
        """
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.num_layers = num_layers
        
        # CORREÇÃO: Ajustar num_heads para ser divisível por output_dims
        self.num_heads = self._adjust_num_heads(output_dims, num_heads)
        
        print(f"🔧 TemporalTransformer: output_dims={output_dims}, num_heads ajustado para {self.num_heads}")
        
        # Projeção inicial para ajustar dimensões
        self.input_proj = nn.Linear(input_dims, output_dims)
        
        # Codificação posicional aprendível
        self.positional_encoding = nn.Parameter(torch.randn(1, max_len, output_dims) * 0.1)
        
        # Camadas do Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_dims,
            nhead=self.num_heads,  # Usar o valor ajustado
            dim_feedforward=output_dims * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, 
            num_layers=num_layers
        )
        
        # Projeção final
        self.output_proj = nn.Linear(output_dims, output_dims)
        
        # Layer norm final
        self.final_norm = nn.LayerNorm(output_dims)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
        # Inicialização Xavier
        self._init_weights()
    
    def _adjust_num_heads(self, embed_dim, desired_heads):
        """
        Ajusta o número de cabeças para ser divisível pela dimensão de embedding.
        Encontra o divisor mais próximo do valor desejado.
        """
        # Encontrar todos os divisores de embed_dim
        divisors = []
        for i in range(1, embed_dim + 1):
            if embed_dim % i == 0:
                divisors.append(i)
        
        # Encontrar o divisor mais próximo do valor desejado
        closest_divisor = min(divisors, key=lambda x: abs(x - desired_heads))
        
        # Garantir que seja pelo menos 1 e no máximo embed_dim
        return max(1, min(closest_divisor, embed_dim))
    
    def _init_weights(self):
        """Inicialização dos pesos"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Entrada no formato (batch, time, input_dims).
        
        Returns:
            torch.Tensor: Saída no formato (batch, time, output_dims).
        """
        batch_size, seq_len, _ = x.shape
        
        # Projeção inicial
        x = self.input_proj(x)  # (batch, time, output_dims)
        
        # Adicionar codificação posicional
        x = x + self.positional_encoding[:, :seq_len, :]
        
        # Aplicar dropout
        x = self.dropout(x)
        
        # Passar pelo Transformer Encoder
        x = self.transformer_encoder(x)  # (batch, time, output_dims)
        
        # Projeção final
        x = self.output_proj(x)
        
        # Normalização final
        x = self.final_norm(x)
        
        return x
    
    def get_attention_weights(self, x):
        """
        Obtém os pesos de atenção das camadas do Transformer.
        Útil para análise e visualização.
        """
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(module, 'self_attn') and hasattr(module.self_attn, 'last_attn_weights'):
                attention_weights.append(module.self_attn.last_attn_weights.detach())
        
        # Registrar hooks
        hooks = []
        for layer in self.transformer_encoder.layers:
            hook = layer.register_forward_hook(hook_fn)
            hooks.append(hook)
        
        # Forward pass
        _ = self.forward(x)
        
        # Remover hooks
        for hook in hooks:
            hook.remove()
        
        return attention_weights


def create_temporal_transformer(input_dims, output_dims, num_heads=4, num_layers=2, max_len=3000, dropout=0.1):
    """
    Função factory para criar o TemporalTransformer.
    Mantém compatibilidade com a interface existente.
    """
    return TemporalTransformer(
        input_dims=input_dims,
        output_dims=output_dims, 
        num_heads=num_heads,
        num_layers=num_layers,
        max_len=max_len,
        dropout=dropout
    )