#!/usr/bin/env python3
"""
Exemplo de uso do TSI com refinamento de tendência via Transformer.

Este script demonstra como usar a nova funcionalidade mantendo 
compatibilidade total com o código original.
"""

import numpy as np
import torch
from tsi import TSI, TSIWithTransformer
from datautils import load_forecast_csv

def exemplo_uso_basico():
    """Exemplo básico de uso do TSI com Transformer"""
    
    print("=== Exemplo TSI com Transformer ===")
    
    # Carregar dados de exemplo
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv('ETTh1')
    
    # Dados de treino
    train_data = data[:, train_slice]
    
    print(f"Shape dos dados: {train_data.shape}")
    print(f"Comprimento máximo de treino: {train_data.shape[1]}")
    
    # Criar modelo TSI com Transformer
    tsi_transformer = TSIWithTransformer(
        input_dims=train_data.shape[-1],
        kernels=[1, 2, 4, 8, 16, 32, 64, 128],
        alpha=0.05,
        max_train_length=train_data.shape[1],
        output_dims=320,
        hidden_dims=64,
        depth=10,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        lr=0.001,
        batch_size=8,
        # Parâmetros do Transformer
        use_transformer=True,
        transformer_heads=4,
        transformer_depth=2,
        transformer_dropout=0.1
    )
    
    print("Modelo TSI com Transformer criado!")
    print(f"Usando Transformer: {tsi_transformer.use_transformer}")
    print(f"Device: {tsi_transformer.device}")
    
    # Treinar por algumas iterações para teste
    print("\\nIniciando treinamento...")
    loss_log = tsi_transformer.fit(train_data, n_iters=10, verbose=True)
    print(f"Treinamento concluído. Loss final: {loss_log[-1]:.4f}")
    
    # Testar encoding
    print("\\nTestando encoding...")
    encoded = tsi_transformer.encode(train_data[:, :100], mode='forecasting')
    print(f"Shape do encoding: {encoded.shape}")
    
    # Analisar pesos de atenção
    print("\\nAnalisando pesos de atenção...")
    sample_data = train_data[:1, :100]  # Uma amostra pequena
    attention_weights = tsi_transformer.get_attention_weights(sample_data)
    
    if attention_weights:
        print(f"Obtidos pesos de atenção de {len(attention_weights)} camadas")
        for i, weights in enumerate(attention_weights):
            print(f"  Camada {i+1}: {weights.shape}")
    
    return tsi_transformer


def comparacao_com_original():
    """Compara TSI original vs TSI com Transformer"""
    
    print("\\n=== Comparação TSI Original vs Transformer ===")
    
    # Carregar dados
    data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv('ETTh1')
    train_data = data[:, train_slice]
    
    # Parâmetros comuns
    common_params = {
        'input_dims': train_data.shape[-1],
        'kernels': [1, 2, 4, 8],
        'alpha': 0.05,
        'max_train_length': min(train_data.shape[1], 512),  # Limitar para teste
        'output_dims': 128,
        'hidden_dims': 32,
        'depth': 6,
        'device': 'cuda' if torch.cuda.is_available() else 'cpu',
        'lr': 0.001,
        'batch_size': 4
    }
    
    # TSI Original
    print("Criando TSI original...")
    tsi_original = TSI(**common_params)
    
    # TSI com Transformer
    print("Criando TSI com Transformer...")
    tsi_transformer = TSIWithTransformer(
        **common_params,
        use_transformer=True,
        transformer_heads=2,
        transformer_depth=1,
        transformer_dropout=0.1
    )
    
    # TSI sem Transformer (para comparação)
    print("Criando TSI sem Transformer...")
    tsi_no_transformer = TSIWithTransformer(
        **common_params,
        use_transformer=False
    )
    
    # Treinar ambos com os mesmos dados
    sample_data = train_data[:, :common_params['max_train_length']]
    
    print("\\nTreinando modelos...")
    
    print("  - TSI Original...")
    loss_original = tsi_original.fit(sample_data, n_iters=5, verbose=False)
    
    print("  - TSI com Transformer...")
    loss_transformer = tsi_transformer.fit(sample_data, n_iters=5, verbose=False)
    
    print("  - TSI sem Transformer...")
    loss_no_transformer = tsi_no_transformer.fit(sample_data, n_iters=5, verbose=False)
    
    # Comparar resultados
    print("\\nResultados:")
    print(f"  TSI Original - Loss final: {loss_original[-1]:.4f}")
    print(f"  TSI com Transformer - Loss final: {loss_transformer[-1]:.4f}")
    print(f"  TSI sem Transformer - Loss final: {loss_no_transformer[-1]:.4f}")
    
    # Testar encoding
    test_data = sample_data[:, :50]
    
    enc_original = tsi_original.encode(test_data, mode='forecasting')
    enc_transformer = tsi_transformer.encode(test_data, mode='forecasting')
    enc_no_transformer = tsi_no_transformer.encode(test_data, mode='forecasting')
    
    print(f"\\nShapes dos encodings:")
    print(f"  Original: {enc_original.shape}")
    print(f"  Com Transformer: {enc_transformer.shape}")
    print(f"  Sem Transformer: {enc_no_transformer.shape}")
    
    return tsi_original, tsi_transformer, tsi_no_transformer


def teste_compatibilidade():
    """Testa se a interface é completamente compatível"""
    
    print("\\n=== Teste de Compatibilidade ===")
    
    # Dados simples para teste
    batch_size, seq_len, features = 2, 100, 7
    test_data = np.random.randn(batch_size, seq_len, features).astype(np.float32)
    
    # Parâmetros mínimos
    params = {
        'input_dims': features,
        'kernels': [1, 2, 4],
        'alpha': 0.05,
        'max_train_length': seq_len,
        'output_dims': 64,
        'device': 'cpu',
        'batch_size': 2
    }
    
    # Criar ambos os modelos
    tsi_original = TSI(**params)
    tsi_transformer = TSIWithTransformer(**params, use_transformer=True)
    
    print("Modelos criados com sucesso!")
    
    # Testar métodos principais
    try:
        # Fit
        tsi_original.fit(test_data, n_iters=2, verbose=False)
        tsi_transformer.fit(test_data, n_iters=2, verbose=False)
        print("✓ Método fit() funciona em ambos")
        
        # Encode
        enc1 = tsi_original.encode(test_data, mode='forecasting')
        enc2 = tsi_transformer.encode(test_data, mode='forecasting')
        print("✓ Método encode() funciona em ambos")
        
        # Save/Load
        tsi_original.save('test_original.pth')
        tsi_transformer.save('test_transformer.pth')
        print("✓ Método save() funciona em ambos")
        
        tsi_original.load('test_original.pth')
        tsi_transformer.load('test_transformer.pth')
        print("✓ Método load() funciona em ambos")
        
        # Método específico do Transformer
        attn = tsi_transformer.get_attention_weights(test_data[:1])
        print("✓ Método get_attention_weights() funciona")
        
        print("\\n🎉 Todos os testes de compatibilidade passaram!")
        
    except Exception as e:
        print(f"❌ Erro no teste: {e}")
        raise
    
    finally:
        # Limpeza
        import os
        for file in ['test_original.pth', 'test_transformer.pth']:
            if os.path.exists(file):
                os.remove(file)


if __name__ == "__main__":
    # Executar exemplos
    try:
        modelo = exemplo_uso_basico()
        comparacao_com_original()
        teste_compatibilidade()
        
        print("\\n✅ Todos os exemplos executados com sucesso!")
        print("\\n📝 Resumo da implementação:")
        print("- ✅ Transformer adicionado após convoluções dilatadas")
        print("- ✅ Compatibilidade total mantida")
        print("- ✅ Estrutura original preservada")
        print("- ✅ Novos recursos adicionais disponíveis")
        
    except Exception as e:
        print(f"❌ Erro durante execução: {e}")
        import traceback
        traceback.print_exc()
