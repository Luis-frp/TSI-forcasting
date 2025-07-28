#!/usr/bin/env python3
"""
Exemplo de integração das melhorias FFT no TSIEncoder
"""

import torch
import torch.nn as nn
from fourier_alternatives import (
    ImprovedBandedFourierLayer, 
    SpectralConv1d, 
    FNetLayer,
    MultiScaleFourierLayer
)

# Importar o encoder original
import sys
sys.path.append('models')
from encoder import TSIEncoder
from dilated_conv import DilatedConvEncoder


class TSIEncoderImproved(TSIEncoder):
    """
    TSI Encoder com melhorias na parte FFT/Fourier
    """
    
    def __init__(self, input_dims, output_dims, kernels, length, hidden_dims=64, 
                 depth=10, mask_mode='binomial', fourier_type='improved'):
        
        # Não chamar super().__init__ para customizar completamente
        nn.Module.__init__(self)
        
        component_dims = output_dims // 2
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.component_dims = component_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode
        self.fourier_type = fourier_type
        
        # Componentes originais
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.feature_extractor = DilatedConvEncoder(
            hidden_dims,
            [hidden_dims] * depth + [output_dims],
            kernel_size=3
        )
        self.repr_dropout = nn.Dropout(p=0.1)
        self.kernels = kernels
        
        # Extração de tendência (mantida igual)
        self.tfd = nn.ModuleList(
            [nn.Conv1d(output_dims, component_dims, k, padding=k-1) for k in kernels]
        )
        
        # NOVO: Extração de sazonalidade melhorada
        if fourier_type == 'improved':
            self.sfd = nn.ModuleList([
                ImprovedBandedFourierLayer(
                    output_dims, component_dims, b, 1, length=length,
                    use_learnable_freq=True, freq_dropout=0.1, complex_activation=True
                ) for b in range(1)
            ])
        
        elif fourier_type == 'spectral':
            # Usar Spectral Convolution (FNO-style)
            self.sfd_spectral = SpectralConv1d(
                output_dims, component_dims, modes=min(16, length//4), length=length
            )
            
        elif fourier_type == 'fnet':
            # Usar FNet para capturar padrões sazonais
            self.sfd_fnet = nn.Sequential(
                nn.Linear(output_dims, component_dims),
                FNetLayer(component_dims, dropout=0.1),
                FNetLayer(component_dims, dropout=0.1)
            )
            
        elif fourier_type == 'multiscale':
            # FFT Multi-escala
            self.sfd = nn.ModuleList([
                MultiScaleFourierLayer(
                    output_dims, component_dims, 
                    scales=[1, 2, 4], length=length
                )
            ])
        
        else:  # 'original'
            # Manter BandedFourierLayer original
            from encoder import BandedFourierLayer
            self.sfd = nn.ModuleList([
                BandedFourierLayer(output_dims, component_dims, b, 1, length=length) 
                for b in range(1)
            ])
    
    def forward(self, x, tcn_output=False, mask='all_true'):
        # Parte inicial igual ao original
        nan_mask = ~x.isnan().any(axis=-1)
        x[~nan_mask] = 0
        x = self.input_fc(x)
        
        # Aplicar máscara (código original)
        if mask is None:
            mask = self.mask_mode if self.training else 'all_true'
        
        if mask == 'binomial':
            from encoder import generate_binomial_mask
            mask = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'continuous':
            from encoder import generate_continuous_mask
            mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == 'all_true':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == 'all_false':
            mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == 'mask_last':
            mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            mask[:, -1] = False
        
        mask &= nan_mask
        x[~mask] = 0
        
        # Feature extraction
        x = x.transpose(1, 2)  # B x Ch x T
        x = self.feature_extractor(x)  # B x Co x T
        
        if tcn_output:
            return x.transpose(1, 2)
        
        # Extração de tendência (mantida igual)
        trend = []
        for idx, mod in enumerate(self.tfd):
            out = mod(x)
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trend.append(out.transpose(1, 2))
        
        from einops import reduce, rearrange
        trend = reduce(
            rearrange(trend, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )
        
        # NOVO: Extração de sazonalidade melhorada
        x = x.transpose(1, 2)  # B x T x Co
        
        if self.fourier_type == 'spectral':
            # Spectral Convolution
            x_conv = x.transpose(1, 2)  # B x Co x T para conv1d
            season = self.sfd_spectral(x_conv).transpose(1, 2)  # Volta para B x T x D
            
        elif self.fourier_type == 'fnet':
            # FNet
            season = self.sfd_fnet(x)
            
        else:  # 'improved', 'multiscale', 'original'
            season = []
            for mod in self.sfd:
                out = mod(x)
                season.append(out)
            season = season[0]
        
        return trend, self.repr_dropout(season)


def demo_comparacao():
    """Demonstra as diferentes abordagens"""
    
    print("🔬 Demonstração das Melhorias FFT")
    print("=" * 50)
    
    # Parâmetros de teste
    batch_size = 4
    seq_len = 256
    input_dims = 7
    output_dims = 128
    kernels = [1, 2, 4, 8]
    
    # Dados sintéticos
    x = torch.randn(batch_size, seq_len, input_dims)
    
    print(f"📊 Dados de entrada: {x.shape}")
    print(f"🎛️  Parâmetros: output_dims={output_dims}, kernels={kernels}")
    
    # Testar diferentes abordagens
    approaches = ['original', 'improved', 'spectral', 'fnet', 'multiscale']
    
    results = {}
    
    for approach in approaches:
        print(f"\\n🧪 Testando abordagem: {approach}")
        
        try:
            model = TSIEncoderImproved(
                input_dims=input_dims,
                output_dims=output_dims,
                kernels=kernels,
                length=seq_len,
                fourier_type=approach
            )
            
            # Forward pass
            with torch.no_grad():
                trend, season = model(x)
            
            print(f"   ✅ Trend shape: {trend.shape}")
            print(f"   ✅ Season shape: {season.shape}")
            
            # Contar parâmetros
            total_params = sum(p.numel() for p in model.parameters())
            print(f"   📈 Parâmetros totais: {total_params:,}")
            
            results[approach] = {
                'trend_shape': trend.shape,
                'season_shape': season.shape,
                'params': total_params,
                'success': True
            }
            
        except Exception as e:
            print(f"   ❌ Erro: {e}")
            results[approach] = {'success': False, 'error': str(e)}
    
    # Resumo comparativo
    print(f"\\n📋 Resumo Comparativo:")
    print("-" * 50)
    
    for approach, result in results.items():
        if result['success']:
            print(f"{approach:12} | Params: {result['params']:8,} | ✅")
        else:
            print(f"{approach:12} | Erro: {result['error'][:30]}... | ❌")
    
    return results


def recomendacoes():
    """Recomendações para diferentes cenários"""
    
    print("\\n💡 Recomendações de Uso:")
    print("=" * 50)
    
    recomendacoes = {
        "🚀 Melhor Performance": {
            "tipo": "spectral",
            "vantagens": ["Muito eficiente O(N log N)", "Boa para seq. longas", "Inspirado em FNO"],
            "desvantagens": ["Mais complexo", "Requer tuning de 'modes'"],
            "quando_usar": "Sequências muito longas (>1000), recursos computacionais limitados"
        },
        
        "🎯 Melhor Precisão": {
            "tipo": "improved",
            "vantagens": ["Frequências aprendíveis", "Ativação complexa", "Regularização"],
            "desvantagens": ["Mais parâmetros", "Computação ligeiramente maior"],
            "quando_usar": "Quando precisão é prioritária, dados suficientes para treino"
        },
        
        "⚖️ Balanceado": {
            "tipo": "multiscale",
            "vantagens": ["Multi-resolução", "Captura padrões variados", "Robusto"],
            "desvantagens": ["Complexidade média", "Mais memória"],
            "quando_usar": "Dados com padrões em múltiplas escalas temporais"
        },
        
        "🔄 Substituto Attention": {
            "tipo": "fnet",
            "vantagens": ["Substitui attention", "Linear complexity", "Simples"],
            "desvantagens": ["Pode perder dependências complexas"],
            "quando_usar": "Alternativa ao Transformer, sequências muito longas"
        },
        
        "🛡️ Conservador": {
            "tipo": "original",
            "vantagens": ["Testado", "Simples", "Estável"],
            "desvantagens": ["Limitações conhecidas"],
            "quando_usar": "Produção, quando estabilidade é crucial"
        }
    }
    
    for categoria, info in recomendacoes.items():
        print(f"\\n{categoria}")
        print(f"Tipo: {info['tipo']}")
        print(f"✅ Vantagens: {', '.join(info['vantagens'])}")
        print(f"⚠️  Desvantagens: {', '.join(info['desvantagens'])}")
        print(f"🎯 Quando usar: {info['quando_usar']}")


if __name__ == "__main__":
    demo_comparacao()
    recomendacoes()
