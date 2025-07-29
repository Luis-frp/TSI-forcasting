import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class WaveletTrendExtractor(nn.Module):
    """
    Extrator de tendência inspirado em wavelets
    
    Funciona como decomposição wavelet:
    1. Filtro passa-baixa: extrai tendências de longo prazo
    2. Filtro passa-alta: identifica detalhes/ruído
    3. Combina inteligentemente para tendência limpa
    """
    
    def __init__(self, input_dims, output_dims, kernels):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernels = kernels
        
        # Módulos para cada kernel
        self.wavelet_modules = nn.ModuleList()
        
        for k in kernels:
            # Filtro passa-baixa (tendência) - metade dos canais
            low_pass = nn.Conv1d(input_dims, output_dims // 2, k, padding=k-1)
            
            # Filtro passa-alta (detalhes/ruído) - metade dos canais  
            high_pass = nn.Conv1d(input_dims, output_dims // 2, k, padding=k-1)
            
            # Combinador final - como recombinar passa-baixa e passa-alta
            combiner = nn.Conv1d(output_dims, output_dims, 1)
            
            # Normalização para estabilidade
            norm = nn.LayerNorm(output_dims)
            
            # Criar módulo para este kernel
            module = nn.ModuleDict({
                'low_pass': low_pass,      # Extrai tendência
                'high_pass': high_pass,    # Extrai detalhes
                'combiner': combiner,      # Combina resultado
                'norm': norm               # Normaliza saída
            })
            
            # Adicionar kernel_size como atributo (não como módulo)
            module.kernel_size = k
            
            self.wavelet_modules.append(module)
        
        # Dropout para regularização
        self.dropout = nn.Dropout(0.1)
    
    def forward(self, x):
        """
        x: (batch, channels, time) - formato conv1d
        Retorna: (batch, time, output_dims) - formato correto
        """
        trends = []
        
        for i, mod in enumerate(self.wavelet_modules):
            # 1. Aplicar filtros passa-baixa e passa-alta
            low_freq = mod['low_pass'](x)   # Tendência (suave)
            high_freq = mod['high_pass'](x) # Detalhes (rápidos)
            
            # 2. Cortar padding se necessário
            k = mod.kernel_size  # Acessar o atributo
            if k != 1:
                low_freq = low_freq[..., :-(k - 1)]
                high_freq = high_freq[..., :-(k - 1)]
            
            # 3. Combinar: concatenar e processar
            # A ideia é: tendência = passa_baixa - α * passa_alta
            combined = torch.cat([low_freq, high_freq], dim=1)  # Concatenar
            
            # 4. Aprender como combinar (equivale a aprender α)
            trend_filtered = mod['combiner'](combined)
            
            # 5. Transpor para formato (batch, time, dims)
            trend_filtered = trend_filtered.transpose(1, 2)
            
            # 6. Normalizar
            trend_filtered = mod['norm'](trend_filtered)
            
            trends.append(trend_filtered)
        
        # 7. Combinar todas as escalas (kernels diferentes)
        if len(trends) == 1:
            final_trend = trends[0]
        else:
            # Média ponderada das diferentes escalas
            final_trend = torch.stack(trends, dim=0).mean(dim=0)
        
        # 8. Aplicar dropout para regularização
        final_trend = self.dropout(final_trend)
        
        return final_trend


# Função para substituir facilmente no código existente
def create_wavelet_trend_extractor(input_dims, output_dims, kernels):
    """
    Criar extrator de tendência wavelet - substituição direta
    """
    return WaveletTrendExtractor(input_dims, output_dims, kernels)

