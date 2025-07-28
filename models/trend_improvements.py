#!/usr/bin/env python3
"""
Melhorias para Extração de Tendência no TSI
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import List


class ImprovedTrendExtractor(nn.Module):
    """
    Extrator de tendência melhorado com múltiplas técnicas avançadas
    """
    def __init__(self, input_dims, output_dims, kernels, 
                 trend_method='adaptive_conv',
                 trend_regularization=True,
                 trend_attention=True,
                 trend_residual=True):
        super().__init__()
        
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.kernels = kernels
        self.trend_method = trend_method
        self.trend_regularization = trend_regularization
        self.trend_attention = trend_attention
        self.trend_residual = trend_residual
        
        if trend_method == 'adaptive_conv':
            self._init_adaptive_conv()
        elif trend_method == 'hierarchical':
            self._init_hierarchical()
        elif trend_method == 'wavelet_inspired':
            self._init_wavelet_inspired()
        elif trend_method == 'multi_scale':
            self._init_multi_scale()
        else:  # 'original'
            self._init_original()
            
        # Componentes adicionais
        if trend_attention:
            self.trend_attention_layer = TrendAttentionLayer(output_dims)
            
        if trend_regularization:
            self.trend_norm = nn.LayerNorm(output_dims)
            self.trend_dropout = nn.Dropout(0.1)
            
    def _init_adaptive_conv(self):
        """Convoluções adaptativas com pesos aprendíveis"""
        self.tfd = nn.ModuleList()
        
        for k in self.kernels:
            # Convolução principal
            conv = nn.Conv1d(self.input_dims, self.output_dims, k, padding=k-1)
            
            # Pesos adaptativos por kernel
            adaptive_weights = nn.Parameter(torch.ones(1))
            
            # Normalização por kernel
            kernel_norm = nn.BatchNorm1d(self.output_dims)
            
            self.tfd.append(nn.ModuleDict({
                'conv': conv,
                'weight': adaptive_weights,
                'norm': kernel_norm
            }))
    
    def _init_hierarchical(self):
        """Extração hierárquica: pequeno -> grande"""
        self.tfd = nn.ModuleList()
        
        # Ordenar kernels do menor para o maior
        sorted_kernels = sorted(self.kernels)
        
        for i, k in enumerate(sorted_kernels):
            # Dimensões progressivas
            in_dim = self.input_dims if i == 0 else self.output_dims
            out_dim = self.output_dims
            
            # Convolução com conexão residual da anterior
            conv = nn.Conv1d(in_dim, out_dim, k, padding=k-1)
            
            # Gating mechanism
            gate = nn.Sequential(
                nn.Conv1d(in_dim, out_dim, 1),
                nn.Sigmoid()
            )
            
            self.tfd.append(nn.ModuleDict({
                'conv': conv,
                'gate': gate,
                'kernel_size': k
            }))
    
    def _init_wavelet_inspired(self):
        """Inspirado em wavelets: decomposição multi-resolução"""
        self.tfd = nn.ModuleList()
        
        for k in self.kernels:
            # Filtro passa-baixa (tendência)
            low_pass = nn.Conv1d(self.input_dims, self.output_dims // 2, k, padding=k-1)
            
            # Filtro passa-alta (detalhes, para subtrair)
            high_pass = nn.Conv1d(self.input_dims, self.output_dims // 2, k, padding=k-1)
            
            # Combinação adaptativa
            combiner = nn.Conv1d(self.output_dims, self.output_dims, 1)
            
            self.tfd.append(nn.ModuleDict({
                'low_pass': low_pass,
                'high_pass': high_pass,
                'combiner': combiner
            }))
    
    def _init_multi_scale(self):
        """Multi-escala com pooling adaptativo"""
        self.tfd = nn.ModuleList()
        
        scales = [1, 2, 4, 8]  # Diferentes escalas temporais
        
        for k in self.kernels:
            scale_convs = nn.ModuleList()
            
            for scale in scales:
                # Convolução na escala específica
                conv = nn.Conv1d(self.input_dims, self.output_dims // len(scales), 
                               k, padding=k-1, dilation=scale)
                scale_convs.append(conv)
            
            # Fusão das escalas
            fusion = nn.Conv1d(self.output_dims, self.output_dims, 1)
            
            self.tfd.append(nn.ModuleDict({
                'scale_convs': scale_convs,
                'fusion': fusion
            }))
    
    def _init_original(self):
        """Método original para comparação"""
        self.tfd = nn.ModuleList([
            nn.Conv1d(self.input_dims, self.output_dims, k, padding=k-1) 
            for k in self.kernels
        ])
    
    def forward(self, x):
        """
        x: (batch, channels, time) - já transposto do TSIEncoder
        """
        if self.trend_method == 'adaptive_conv':
            return self._forward_adaptive_conv(x)
        elif self.trend_method == 'hierarchical':
            return self._forward_hierarchical(x)
        elif self.trend_method == 'wavelet_inspired':
            return self._forward_wavelet_inspired(x)
        elif self.trend_method == 'multi_scale':
            return self._forward_multi_scale(x)
        else:  # 'original'
            return self._forward_original(x)
    
    def _forward_adaptive_conv(self, x):
        """Forward pass para convoluções adaptativas"""
        trends = []
        
        for idx, mod in enumerate(self.tfd):
            # Convolução
            out = mod['conv'](x)  # (b, d, t)
            
            # Cortar padding se necessário
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            
            # Aplicar peso adaptativo
            out = out * mod['weight']
            
            # Normalização
            out = mod['norm'](out)
            
            trends.append(out.transpose(1, 2))  # (b, t, d)
        
        # Combinação ponderada em vez de média simples
        trend_weights = F.softmax(torch.stack([
            mod['weight'] for mod in self.tfd
        ]), dim=0)
        
        weighted_trends = []
        for i, trend in enumerate(trends):
            weighted_trends.append(trend * trend_weights[i])
        
        combined_trend = torch.stack(weighted_trends).sum(dim=0)
        
        return self._apply_post_processing(combined_trend)
    
    def _forward_hierarchical(self, x):
        """Forward pass hierárquico"""
        current = x
        trends = []
        
        for mod in self.tfd:
            # Convolução
            conv_out = mod['conv'](current)
            
            # Gating
            gate_out = mod['gate'](current)
            
            # Aplicar gate
            out = conv_out * gate_out
            
            # Cortar padding
            k = mod['kernel_size']
            if k != 1:
                out = out[..., :-(k - 1)]
            
            trends.append(out.transpose(1, 2))
            
            # Usar saída para próxima camada (hierárquico)
            current = out
        
        # Somar contribuições hierárquicas
        combined_trend = torch.stack(trends).sum(dim=0)
        
        return self._apply_post_processing(combined_trend)
    
    def _forward_wavelet_inspired(self, x):
        """Forward pass inspirado em wavelets"""
        trends = []
        
        for idx, mod in enumerate(self.tfd):
            # Filtros passa-baixa e passa-alta
            low = mod['low_pass'](x)
            high = mod['high_pass'](x)
            
            # Cortar padding
            if self.kernels[idx] != 1:
                low = low[..., :-(self.kernels[idx] - 1)]
                high = high[..., :-(self.kernels[idx] - 1)]
            
            # Combinar (tendência = baixa frequência - detalhes de alta)
            combined = torch.cat([low, high], dim=1)  # Concatenar
            trend = mod['combiner'](combined)
            
            trends.append(trend.transpose(1, 2))
        
        # Média das contribuições de diferentes escalas
        combined_trend = torch.stack(trends).mean(dim=0)
        
        return self._apply_post_processing(combined_trend)
    
    def _forward_multi_scale(self, x):
        """Forward pass multi-escala"""
        trends = []
        
        for mod in self.tfd:
            # Convoluções em diferentes escalas
            scale_outputs = []
            for scale_conv in mod['scale_convs']:
                scale_out = scale_conv(x)
                scale_outputs.append(scale_out)
            
            # Concatenar escalas
            multi_scale = torch.cat(scale_outputs, dim=1)
            
            # Fusão
            fused = mod['fusion'](multi_scale)
            
            trends.append(fused.transpose(1, 2))
        
        # Combinar diferentes kernels
        combined_trend = torch.stack(trends).mean(dim=0)
        
        return self._apply_post_processing(combined_trend)
    
    def _forward_original(self, x):
        """Forward pass original"""
        trends = []
        
        for idx, mod in enumerate(self.tfd):
            out = mod(x)
            if self.kernels[idx] != 1:
                out = out[..., :-(self.kernels[idx] - 1)]
            trends.append(out.transpose(1, 2))
        
        # Média simples (original)
        from einops import reduce, rearrange
        combined_trend = reduce(
            rearrange(trends, 'list b t d -> list b t d'),
            'list b t d -> b t d', 'mean'
        )
        
        return self._apply_post_processing(combined_trend)
    
    def _apply_post_processing(self, trend):
        """Aplicar processamento adicional"""
        
        # Atenção para tendência
        if self.trend_attention:
            trend = self.trend_attention_layer(trend)
        
        # Regularização
        if self.trend_regularization:
            trend = self.trend_norm(trend)
            trend = self.trend_dropout(trend)
        
        return trend


class TrendAttentionLayer(nn.Module):
    """
    Atenção específica para tendência - foca em padrões de longo prazo
    """
    def __init__(self, d_model, num_heads=4):
        super().__init__()
        
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        # Projeções para Q, K, V
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        
        # Projeção de saída
        self.out_proj = nn.Linear(d_model, d_model)
        
        # Bias posicional para tendência (favorece dependências longas)
        self.pos_bias = nn.Parameter(torch.zeros(1, 1, d_model))
        
    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        batch_size, seq_len, d_model = x.shape
        
        # Adicionar bias posicional
        x = x + self.pos_bias
        
        # Projeções
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Transpor para (batch, num_heads, seq_len, head_dim)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Atenção
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # Bias para favorecer dependências de longo prazo
        # Criar matriz que favorece posições distantes
        pos_bias = self._create_trend_bias(seq_len, x.device)
        scores = scores + pos_bias
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Aplicar atenção
        out = torch.matmul(attn_weights, v)
        
        # Concatenar cabeças
        out = out.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        # Projeção final
        out = self.out_proj(out)
        
        # Conexão residual
        return x + out
    
    def _create_trend_bias(self, seq_len, device):
        """Criar bias que favorece dependências de longo prazo"""
        # Matriz que dá mais peso para posições distantes
        bias = torch.zeros(1, 1, seq_len, seq_len, device=device)
        
        for i in range(seq_len):
            for j in range(seq_len):
                # Maior peso para maior distância (tendência = longo prazo)
                distance = abs(i - j)
                bias[0, 0, i, j] = math.log(1 + distance) * 0.1
        
        return bias


def create_improved_trend_extractor(input_dims, output_dims, kernels, method='adaptive_conv'):
    """
    Factory function para criar extrator de tendência melhorado
    
    Args:
        method: 'adaptive_conv', 'hierarchical', 'wavelet_inspired', 'multi_scale', 'original'
    """
    return ImprovedTrendExtractor(
        input_dims=input_dims,
        output_dims=output_dims, 
        kernels=kernels,
        trend_method=method,
        trend_regularization=True,
        trend_attention=True,
        trend_residual=True
    )


# Exemplo de uso
if __name__ == "__main__":
    print("🔧 Melhorias para Extração de Tendência")
    print("=" * 50)
    
    # Parâmetros de teste
    batch_size = 4
    seq_len = 201
    input_dims = 64
    output_dims = 32
    kernels = [1, 3, 5, 7]
    
    methods = ['adaptive_conv', 'hierarchical', 'wavelet_inspired', 'multi_scale', 'original']
    
    x = torch.randn(batch_size, input_dims, seq_len)  # Formato conv1d
    
    for method in methods:
        print(f"\n🧪 Testando: {method}")
        
        try:
            extractor = create_improved_trend_extractor(
                input_dims, output_dims, kernels, method
            )
            
            with torch.no_grad():
                trend = extractor(x)
            
            print(f"✅ Sucesso! Output shape: {trend.shape}")
            print(f"   Parâmetros: {sum(p.numel() for p in extractor.parameters()):,}")
            
        except Exception as e:
            print(f"❌ Erro: {e}")
    
    print(f"\n🎯 Recomendação: 'adaptive_conv' para melhor qualidade")
    print(f"   'hierarchical' para padrões complexos")
    print(f"   'multi_scale' para diferentes escalas temporais")
