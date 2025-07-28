import torch
import torch.nn as nn
import torch.fft as fft
import torch.nn.functional as F
import math


class FNetLayer(nn.Module):
    """
    FNet: Substituição do self-attention por FFT
    Mais eficiente que Transformer para sequências longas
    """
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model)
        )
    
    def fourier_transform(self, x):
        """Aplica FFT real + imaginária"""
        # x: (batch, seq_len, d_model)
        x_fft = torch.fft.fft(x.float(), dim=1)  # FFT na dimensão temporal
        return torch.real(x_fft)  # Pega apenas a parte real
    
    def forward(self, x):
        # Fourier mixing (substitui self-attention)
        residual = x
        x = self.norm1(x)
        x = self.fourier_transform(x)
        x = residual + self.dropout(x)
        
        # Feed forward
        residual = x
        x = self.norm2(x)
        x = self.feed_forward(x)
        x = residual + x
        
        return x


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


class SpectralConv1d(nn.Module):
    """
    Convolução Espectral - Inspirada em Fourier Neural Operator (FNO)
    Muito eficiente para sequências longas
    """
    def __init__(self, in_channels, out_channels, modes, length):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes  # Número de modos de Fourier a manter
        self.length = length
        
        # Pesos espectrais complexos
        self.weights = nn.Parameter(torch.empty(in_channels, out_channels, modes, dtype=torch.cfloat))
        
        # Convolução residual no espaço físico
        self.conv_residual = nn.Conv1d(in_channels, out_channels, 1)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        # Inicialização para pesos complexos
        nn.init.xavier_uniform_(self.weights.real)
        nn.init.xavier_uniform_(self.weights.imag)
    
    def forward(self, x):
        # x: (batch, channels, length)
        batch_size = x.shape[0]
        
        # FFT
        x_ft = torch.fft.rfft(x, dim=-1)
        
        # Produto no espaço de Fourier
        out_ft = torch.zeros(batch_size, self.out_channels, x_ft.size(-1), 
                           dtype=torch.cfloat, device=x.device)
        
        # Multiplicação apenas nos modos baixos
        out_ft[:, :, :self.modes] = torch.einsum('bix,iox->box', x_ft[:, :, :self.modes], self.weights)
        
        # IFFT
        x_spectral = torch.fft.irfft(out_ft, n=x.size(-1), dim=-1)
        
        # Componente residual
        x_residual = self.conv_residual(x)
        
        return x_spectral + x_residual


class WaveletLayer(nn.Module):
    """
    Alternativa: Transformada Wavelet em vez de Fourier
    Melhor para sinais não-estacionários
    """
    def __init__(self, in_channels, out_channels, wavelet_type='db4', levels=3):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.levels = levels
        
        # Filtros wavelet aprendíveis
        self.wavelet_weights = nn.ModuleList([
            nn.Linear(in_channels, out_channels) for _ in range(levels + 1)
        ])
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        # Implementação simplificada - na prática usaria PyWavelets
        
        batch_size, seq_len, channels = x.shape
        
        # Simulação de decomposição wavelet multi-resolução
        outputs = []
        current = x
        
        for level in range(self.levels):
            # Downsampling simulado (na prática seria DWT)
            pooled = F.avg_pool1d(current.transpose(1, 2), kernel_size=2, stride=2)
            pooled = pooled.transpose(1, 2)
            
            # Transformação linear
            transformed = self.wavelet_weights[level](pooled)
            
            # Upsampling para tamanho original
            upsampled = F.interpolate(transformed.transpose(1, 2), 
                                    size=seq_len, mode='linear', align_corners=False)
            outputs.append(upsampled.transpose(1, 2))
            
            current = pooled
        
        # Combinação final
        return sum(outputs) / len(outputs)


class MultiScaleFourierLayer(nn.Module):
    """
    FFT Multi-escala: Processa diferentes escalas temporais
    """
    def __init__(self, in_channels, out_channels, scales=[1, 2, 4, 8], length=201):
        super().__init__()
        
        self.scales = scales
        self.length = length
        
        # Uma BandedFourierLayer para cada escala
        self.fourier_layers = nn.ModuleList([
            BandedFourierLayer(in_channels, out_channels // len(scales), 
                             0, 1, length // scale)
            for scale in scales
        ])
        
        # Projeção final
        self.output_proj = nn.Linear(out_channels, out_channels)
        
    def forward(self, x):
        # x: (batch, seq_len, channels)
        batch_size, seq_len, channels = x.shape
        
        outputs = []
        
        for scale, layer in zip(self.scales, self.fourier_layers):
            # Downsample
            if scale > 1:
                x_scaled = F.avg_pool1d(x.transpose(1, 2), kernel_size=scale, stride=scale)
                x_scaled = x_scaled.transpose(1, 2)
            else:
                x_scaled = x
            
            # Aplicar Fourier
            out_scaled = layer(x_scaled)
            
            # Upsample back
            if scale > 1:
                out_scaled = F.interpolate(out_scaled.transpose(1, 2), 
                                         size=seq_len, mode='linear', align_corners=False)
                out_scaled = out_scaled.transpose(1, 2)
            
            outputs.append(out_scaled)
        
        # Concatenar e projetar
        combined = torch.cat(outputs, dim=-1)
        return self.output_proj(combined)
