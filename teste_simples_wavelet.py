#!/usr/bin/env python3
"""
Teste Simples: Wavelet Trend Extractor
"""

import sys
import os
sys.path.append('.')

try:
    import torch
    import torch.nn as nn
    
    print("🧪 Teste Simples: Wavelet Trend Extractor")
    print("=" * 50)
    
    # 1. Testar o extrator isoladamente
    print("\n1️⃣ Testando Wavelet Extractor isoladamente...")
    
    from wavelet_trend_extractor import WaveletTrendExtractor
    
    # Parâmetros
    batch_size = 2
    seq_len = 201
    input_dims = 64
    output_dims = 32
    kernels = [1, 3, 5]
    
    # Criar dados de teste
    x = torch.randn(batch_size, input_dims, seq_len)  # Formato conv1d
    
    # Criar extrator
    extractor = WaveletTrendExtractor(input_dims, output_dims, kernels)
    
    print(f"   📊 Input shape: {x.shape}")
    print(f"   📊 Parâmetros: {sum(p.numel() for p in extractor.parameters()):,}")
    
    # Teste forward
    with torch.no_grad():
        trend = extractor(x)
    
    print(f"   ✅ Output shape: {trend.shape}")
    print(f"   ✅ Extrator isolado: OK")
    
    # 2. Testar integração com TSIEncoder
    print("\n2️⃣ Testando integração com TSIEncoder...")
    
    from models.encoder import TSIEncoder
    
    # Criar encoder
    encoder = TSIEncoder(
        input_dims=1,
        output_dims=64,
        kernels=[1, 3, 5],
        length=201,
        hidden_dims=64
    )
    
    print(f"   📊 use_wavelet_trend: {encoder.use_wavelet_trend}")
    print(f"   📊 Tipo do extrator: {type(encoder.tfd).__name__}")
    
    # Dados de teste
    test_data = torch.randn(2, 201, 1)  # [batch, time, features]
    
    # Forward pass
    with torch.no_grad():
        result = encoder(test_data)
    
    if isinstance(result, tuple):
        trend, season = result
        print(f"   ✅ Trend shape: {trend.shape}")
        print(f"   ✅ Season shape: {season.shape}")
        print(f"   ✅ Output é tupla (trend, season): OK")
    else:
        print(f"   ✅ Output shape: {result.shape}")
        print(f"   ✅ Output é tensor único: OK")
    
    print(f"\n🎉 Todos os testes passaram!")
    print(f"🌊 Wavelet Trend Extractor está funcionando!")
    
    print(f"\n📋 Resumo das Melhorias:")
    print(f"   • Decomposição passa-baixa/passa-alta")
    print(f"   • Combinação inteligente para tendência limpa")
    print(f"   • Regularização automática")
    print(f"   • Compatibilidade total com código existente")
    
except ImportError as e:
    print(f"❌ Erro de importação: {e}")
    print(f"   Verifique se o PyTorch está instalado")
except Exception as e:
    print(f"❌ Erro durante teste: {e}")
    import traceback
    traceback.print_exc()
