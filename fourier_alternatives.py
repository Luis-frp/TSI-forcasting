import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math

class ImprovedBandedFourierLayer(nn.Module):
    """
    Versão melhorada do BandedFourierLayer com múltiplas melhorias
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
        """Ativação complexa: CReLU ou similar"""
        if self.complex_activation:
            real, imag = x.real, x.imag
            magnitude = torch.sqrt(real**2 + imag**2)
            phase = torch.atan2(imag, real)
            
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
            freq_weights = self.freq_dropout(F.softmax(self.freq_weights, dim=0))
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
