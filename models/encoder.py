import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat

import numpy as np

from .dilated_conv import DilatedConvEncoder
from .temporal_transformer import create_temporal_transformer
from .informer_inspired_transformer import create_informer_inspired_transformer


def generate_continuous_mask(B, T, n=5, l=0.1):
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)
    
    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)
    
    for i in range(B):
        for _ in range(n):
            t = np.random.randint(T-l+1)
            res[i, t:t+l] = False
    return res


def generate_binomial_mask(B, T, p=0.5):
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)


class BandedFourierLayer(nn.Module):
    def __init__(self, in_channels, out_channels, band, num_bands, length=201):
        super().__init__()
        
        self.length = length
        self.total_freqs = (self.length // 2) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.band = band
        self.num_bands = num_bands
        
        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0
        )
        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs
        
        self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))
        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat))
        self.reset_parameters()
    
    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight.real, a=math.sqrt(5))
        nn.init.kaiming_uniform_(self.weight.imag, a=math.sqrt(5))
        fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight.real)
        bound = 1 / math.sqrt(fan_in)
        nn.init.uniform_(self.bias.real, -bound, bound)
        nn.init.uniform_(self.bias.imag, -bound, bound)
    
    def forward(self, input):
        b, t, _ = input.shape
        input_fft = fft.rfft(input, dim=1)
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, 
                                device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = (
            torch.einsum('bti,tio->bto', input_fft[:, self.start:self.end], self.weight) + self.bias
        )
        return fft.irfft(output_fft, n=input.size(1), dim=1)


class NonlinearICA(nn.Module):
    def __init__(self, input_dim, hidden_dim, source_dim):
        super(NonlinearICA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, source_dim)
        )
        
    def encode(self, x):
        return self.encoder(x)
    
    def forward(self, x):
        return self.encode(x)


class TSIEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial',
                 # Parâmetros para arquitetura híbrida
                 use_transformer_refiner=True,
                 transformer_type='temporal',  # 'temporal' ou 'informer'
                 transformer_heads=4,
                 transformer_depth=2,
                 transformer_dropout=0.1,
                 informer_factor=5,
                 informer_distil=True):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.use_transformer_refiner = use_transformer_refiner
        self.transformer_type = transformer_type
        self.length = length

        # Projeção inicial
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        # 1. EXTRAÇÃO BASE: DilatedConvEncoder (sempre presente)
        self.conv_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        # 2. REFINAMENTO: Transformer (opcional)
        if use_transformer_refiner:
            if transformer_type == 'informer':
                self.transformer_refiner = create_informer_inspired_transformer(
                    input_dims=output_dims,
                    output_dims=output_dims,
                    num_heads=transformer_heads,
                    num_layers=transformer_depth,
                    max_len=length,
                    dropout=transformer_dropout,
                    factor=informer_factor,
                    distil=informer_distil
                )
            else:  # 'temporal' (original)
                self.transformer_refiner = create_temporal_transformer(
                    input_dims=output_dims,
                    output_dims=output_dims,
                    num_heads=transformer_heads,
                    num_layers=transformer_depth,
                    max_len=length,
                    dropout=transformer_dropout
                )

        # Dropout para representações
        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        # 3. DECOMPOSIÇÃO: Tendência com Wavelet
        from .wavelet_trend_extractor import create_wavelet_trend_extractor
        self.tfd = create_wavelet_trend_extractor(
            input_dims=output_dims,
            output_dims=component_dims,
            kernels=kernels
        )

        # 4. DECOMPOSIÇÃO: Sazonalidade com Fourier
        self.sfd = nn.ModuleList([
            BandedFourierLayer(
                output_dims, 
                component_dims, 
                b, 
                1, 
                length=length
            ) for b in range(1)
        ])

    def forward(self, x, tcn_output=False, mask='all_true'):
        """
        Forward pass com pipeline híbrido: Conv → Transformer → Decomposição
        
        Args:
            x: (batch, time, input_dims)
            tcn_output: Se True, retorna apenas a saída após refinamento
            mask: Tipo de máscara a aplicar
        
        Returns:
            Se tcn_output=True: saída refinada
            Senão: (trend, season) - componentes de tendência e sazonalidade
        """
        # Tratar valores NaN
        nan_mask = ~x.isnan().any(axis=-1)
        x = torch.where(nan_mask.unsqueeze(-1), x, torch.zeros_like(x))
        
        # Projeção inicial
        x = self.input_fc(x)  # (batch, time, hidden_dims)

        # Gerar e aplicar máscara
        if mask is None:
            mask = self.mask_mode if self.training else 'all_true'

        if mask == 'binomial':
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False

        mask &= nan_mask
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

        # PIPELINE HÍBRIDO:
        
        # 1. EXTRAÇÃO BASE com DilatedConvEncoder
        x_conv = x.transpose(1, 2)  # (batch, hidden_dims, time) para conv1d
        x_conv = self.conv_extractor(x_conv)  # (batch, output_dims, time)
        x_conv = x_conv.transpose(1, 2)  # (batch, time, output_dims)

        # 2. REFINAMENTO com Transformer (se habilitado)
        if self.use_transformer_refiner:
            x_refined = self.transformer_refiner(x_conv)  # (batch, time_refined, output_dims)
            
            # Ajustar dimensões se a destilação mudou o tamanho temporal
            if x_refined.size(1) != x_conv.size(1):
                # Interpolar para manter o tamanho original
                x_refined = F.interpolate(
                    x_refined.transpose(1, 2), 
                    size=x_conv.size(1), 
                    mode='linear', 
                    align_corners=False
                ).transpose(1, 2)
            
            # Conexão residual: combinar conv + transformer
            x = x_conv + x_refined  # Soma residual
        else:
            x = x_conv

        if tcn_output:
            return x

        # 3. DECOMPOSIÇÃO em Tendência e Sazonalidade
        
        # Tendência com Wavelet
        x_for_trend = x.transpose(1, 2)  # (batch, output_dims, time)
        trend = self.tfd(x_for_trend)  # (batch, time, component_dims)

        # Sazonalidade com Fourier
        season = []
        for mod in self.sfd:
            out = mod(x)  # (batch, time, component_dims)
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)

    def get_attention_weights(self, x):
        """
        Obtém pesos de atenção do Transformer refinador (se habilitado).
        """
        if not self.use_transformer_refiner:
            return None
        
        # Processar entrada até o ponto do transformer
        nan_mask = ~x.isnan().any(axis=-1)
        x = torch.where(nan_mask.unsqueeze(-1), x, torch.zeros_like(x))
        x = self.input_fc(x)
        
        # Passar pela convolução primeiro
        x_conv = x.transpose(1, 2)
        x_conv = self.conv_extractor(x_conv)
        x_conv = x_conv.transpose(1, 2)
        
        # Obter pesos de atenção do refinador
        return self.transformer_refiner.get_attention_weights(x_conv)