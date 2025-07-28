#!/usr/bin/env python3
"""
Teste da implementação melhorada do BandedFourierLayer
"""

import torch
import sys
import os

# Adicionar o diretório principal ao path
sys.path.append('/home/felipe/Área de trabalho/Trabalho de Séries /TSI-forcasting')

def test_improved_fourier_layer():
    print("🧪 Testando a versão melhorada do BandedFourierLayer...")
    
    try:
        from models.encoder import BandedFourierLayer
        
        # Parâmetros de teste
        batch_size = 2
        seq_len = 201
        in_channels = 64
        out_channels = 32
        
        # Criar dados de teste
        x = torch.randn(batch_size, seq_len, in_channels)
        
        # Instanciar a camada melhorada
        layer = BandedFourierLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            band=0,
            num_bands=1,
            length=seq_len,
            use_learnable_freq=True,
            freq_dropout=0.1,
            complex_activation=True
        )
        
        print(f"✅ Camada criada com sucesso!")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Parâmetros: in_channels={in_channels}, out_channels={out_channels}")
        print(f"   - Frequências aprendíveis: {layer.use_learnable_freq}")
        print(f"   - Ativação complexa: {layer.complex_activation}")
        
        # Teste forward pass
        with torch.no_grad():
            output = layer(x)
            
        print(f"✅ Forward pass executado com sucesso!")
        print(f"   - Output shape: {output.shape}")
        print(f"   - Output dtype: {output.dtype}")
        
        # Verificar se as dimensões estão corretas
        assert output.shape == (batch_size, seq_len, out_channels), f"Shape incorreta: {output.shape}"
        assert output.dtype == torch.float32, f"Dtype incorreto: {output.dtype}"
        
        # Teste com gradientes
        x.requires_grad_(True)
        output = layer(x)
        loss = output.sum()
        loss.backward()
        
        print(f"✅ Backpropagation executada com sucesso!")
        print(f"   - Gradientes calculados para {len(list(layer.parameters()))} parâmetros")
        
        # Verificar se há gradientes
        for name, param in layer.named_parameters():
            if param.grad is not None:
                print(f"   - {name}: grad norm = {param.grad.norm().item():.6f}")
        
        print(f"\n🎉 Todos os testes passaram! A versão melhorada está funcionando corretamente.\n")
        
        # Comparar número de parâmetros
        total_params = sum(p.numel() for p in layer.parameters())
        print(f"📊 Estatísticas da camada:")
        print(f"   - Total de parâmetros: {total_params:,}")
        print(f"   - Frequências processadas: {layer.num_freqs}")
        print(f"   - Range de frequências: [{layer.start}:{layer.end}]")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_tsi_encoder():
    print("🧪 Testando TSIEncoder com a camada melhorada...")
    
    try:
        from models.encoder import TSIEncoder
        
        # Parâmetros de teste
        batch_size = 2
        seq_len = 201
        input_dims = 7  # 7 features como nos datasets
        output_dims = 64
        
        # Criar dados de teste
        x = torch.randn(batch_size, seq_len, input_dims)
        
        # Instanciar o encoder
        encoder = TSIEncoder(
            input_dims=input_dims,
            output_dims=output_dims,
            kernels=[1, 3, 5],
            length=seq_len,
            hidden_dims=64,
            depth=3
        )
        
        print(f"✅ TSIEncoder criado com sucesso!")
        
        # Teste forward pass
        with torch.no_grad():
            trend, season = encoder(x)
            
        print(f"✅ TSIEncoder forward pass executado com sucesso!")
        print(f"   - Input shape: {x.shape}")
        print(f"   - Trend shape: {trend.shape}")
        print(f"   - Season shape: {season.shape}")
        
        expected_component_dims = output_dims // 2
        assert trend.shape == (batch_size, seq_len, expected_component_dims)
        assert season.shape == (batch_size, seq_len, expected_component_dims)
        
        print(f"\n🎉 TSIEncoder está funcionando corretamente com a camada melhorada!\n")
        
        return True
        
    except Exception as e:
        print(f"❌ Erro durante o teste do TSIEncoder: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    print("🚀 Iniciando testes da implementação melhorada...\n")
    
    success1 = test_improved_fourier_layer()
    print("-" * 60)
    success2 = test_tsi_encoder()
    
    if success1 and success2:
        print("✅ Todos os testes passaram! A implementação está pronta para uso.")
    else:
        print("❌ Alguns testes falharam. Verifique os erros acima.")
