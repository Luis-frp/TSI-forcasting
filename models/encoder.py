import math
from typing import List

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft
from einops import reduce, rearrange, repeat

import numpy as np

from .dilated_conv import DilatedConvEncoder
# from .lstm import LSTMEncoder

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

### AQUI É ONDE TEMOS O FFT APLICAÇÃO DA TRANSFORMADA DE FOURIER - VERSÃO MELHORADA
class BandedFourierLayer(nn.Module):
    """
    Versão melhorada do BandedFourierLayer com múltiplas melhorias:
    - Frequências aprendíveis
    - Ativação complexa
    - Regularização com dropout
    - Normalização de camada
    """
    def __init__(self, in_channels, out_channels, band, num_bands, length=201, 
                 use_learnable_freq=True, freq_dropout=0.1, complex_activation=True):
        super().__init__()
        
        self.length = length
        self.total_freqs = (self.length // 2) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.band = band
        self.num_bands = num_bands
        self.use_learnable_freq = use_learnable_freq
        self.complex_activation = complex_activation
        
        # Cálculo de frequências
        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0
        )
        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs
        
        # Pesos complexos melhorados
        if use_learnable_freq:
            # Frequências aprendíveis
            self.freq_weights = nn.Parameter(torch.ones(self.num_freqs))
            
        # Pesos principais
        self.weight_real = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels)))
        self.weight_imag = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels)))
        self.bias_real = nn.Parameter(torch.empty((self.num_freqs, out_channels)))
        self.bias_imag = nn.Parameter(torch.empty((self.num_freqs, out_channels)))
        
        # Dropout para regularização
        self.freq_dropout = nn.Dropout(freq_dropout)
        
        # Normalização
        self.layer_norm = nn.LayerNorm(out_channels)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Inicialização Xavier/Glorot para pesos reais
        for weight in [self.weight_real, self.weight_imag]:
            nn.init.xavier_uniform_(weight)
        
        # Inicialização pequena para bias
        nn.init.zeros_(self.bias_real)
        nn.init.zeros_(self.bias_imag)
    
    def complex_activation_fn(self, x):
        """Ativação complexa: CReLU ou similar com estabilidade numérica"""
        if self.complex_activation:
            real, imag = x.real, x.imag
            # Adicionar epsilon para estabilidade numérica
            eps = 1e-8
            magnitude = torch.sqrt(real**2 + imag**2 + eps)
            phase = torch.atan2(imag, real + eps)
            
            # Aplicar ativação na magnitude
            magnitude = F.relu(magnitude)
            
            # Reconstruir número complexo
            return magnitude * torch.complex(torch.cos(phase), torch.sin(phase))
        return x
    
    def forward(self, input):
        b, t, _ = input.shape
        
        # FFT
        input_fft = fft.rfft(input, dim=1)
        
        # Selecionar banda de frequências
        selected_fft = input_fft[:, self.start:self.end]
        
        # Aplicar pesos aprendíveis às frequências
        if self.use_learnable_freq:
            # Usar clamp para evitar valores extremos
            freq_weights = self.freq_dropout(F.softmax(torch.clamp(self.freq_weights, -10, 10), dim=0))
            selected_fft = selected_fft * freq_weights.unsqueeze(0).unsqueeze(-1)
        
        # Construir pesos complexos
        weight_complex = torch.complex(self.weight_real, self.weight_imag)
        bias_complex = torch.complex(self.bias_real, self.bias_imag)
        
        # Transformação linear complexa
        output_fft_band = torch.einsum('bti,tio->bto', selected_fft, weight_complex) + bias_complex
        
        # Ativação complexa
        output_fft_band = self.complex_activation_fn(output_fft_band)
        
        # Reconstruir FFT completa
        output_fft = torch.zeros(b, t // 2 + 1, self.out_channels, 
                                device=input.device, dtype=torch.cfloat)
        output_fft[:, self.start:self.end] = output_fft_band
        
        # IFFT
        output = fft.irfft(output_fft, n=input.size(1), dim=1)
        
        # Normalização
        output = self.layer_norm(output)
        
        return output


class NonlinearICA(nn.Module):
    def __init__(self, input_dim, hidden_dim, source_dim):
        super(NonlinearICA, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, source_dim),
            nn.Tanh()
        )
        self.decoder = nn.Sequential(
            nn.Linear(source_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim)
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

    def encode(self, x):
        return self.encoder(x)


# AQUI É ONDE ESTA ACONTECENDO A EXTRAÇÃO DA TENDENCIA E SAZONALIDADE
class TSIEncoder(nn.Module):
    def __init__(self, input_dims, output_dims,
                 kernels: List[int],
                 length: int,
                 hidden_dims=64, depth=10,
                 mask_mode='binomial'):
        super().__init__()

        component_dims = output_dims // 2

        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.input_fc = nn.Linear(input_dims, hidden_dims)

        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )

        self.repr_dropout = nn.Dropout(p=0.1)

        self.kernels = kernels

        # EXTRAÇÃO DE TENDÊNCIA MELHORADA - Método Wavelet
        self.use_wavelet_trend = True  # True para usar wavelet, False para original
        
        if self.use_wavelet_trend:
            # Método wavelet inspirado - melhor para séries temporais
            from wavelet_trend_extractor import create_wavelet_trend_extractor
            self.tfd = create_wavelet_trend_extractor(
                input_dims=output_dims,
                output_dims=component_dims,
                kernels=kernels
            )
        else:
            # Método original
            self.tfd = nn.ModuleList(
                [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
            )

        self.sfd = nn.ModuleList(
            [BandedFourierLayer(output_dims, component_dims, b, 1, length=length) for b in range(1)]
        )


    def forward(self, x, tcn_output=False, mask='all_true'):  # x: B x T x input_dims
        nan_mask = ~x.isnan().any(axis=-1)
        # Evitar operação in-place quando requires_grad=True
        x = torch.where(nan_mask.unsqueeze(-1), x, torch.zeros_like(x))
        x = self.input_fc(x)  # B x T x Ch

        # generate & apply mask
        if mask is None:
            if self.training:
                mask = self.mask_mode
            else:
                mask = 'all_true'

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
        # Evitar operação in-place quando requires_grad=True
        x = torch.where(mask.unsqueeze(-1), x, torch.zeros_like(x))

        # conv encoder
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T

        if tcn_output:
            return x.transpose(1, 2)

        # EXTRAÇÃO DE TENDÊNCIA MELHORADA - Wavelet
        if self.use_wavelet_trend:
            # Método wavelet - já retorna no formato correto (b, t, d)
            trend = self.tfd(x)
        else:
            # Método original
            trend = []
            for idx, mod in enumerate(self.tfd):
                out = mod(x)  # b d t
                if self.kernels[idx] != 1:
                    out = out[..., :-(self.kernels[idx] - 1)]
                trend.append(out.transpose(1, 2))  # b t d
            trend = reduce(
                rearrange(trend, 'list b t d -> list b t d'),
                'list b t d -> b t d', 'mean'
            )

        x = x.transpose(1, 2)  # B x T x Co

        season = []

        for mod in self.sfd:
            out = mod(x)  # b t d
            season.append(out)
        season = season[0]

        return trend, self.repr_dropout(season)
        #return trend
