#!/usr/bin/env python3
"""
Teste rápido para verificar se o problema de dimensões foi corrigido
"""

import subprocess
import sys

def test_training():
    """Teste rápido com poucas épocas"""
    
    print("🧪 Teste Rápido: ETTh1 com melhorias FFT")
    print("=" * 50)
    
    # Comando de teste com poucas épocas
    cmd = [
        "python", "train.py",
        "ETTh1",                         # dataset
        "teste_correcao_dimensoes",      # run_name
        "--archive", "forecast_csv",
        "--gpu", "0",
        "--batch-size", "8",             # batch pequeno para teste
        "--lr", "0.001",
        "--repr-dims", "64",             # dimensões menores para teste
        "--max-train-length", "201",
        "--epochs", "3",                 # poucas épocas para teste
        "--kernels", "1", "3", "5",      # kernels menores
        "--alpha", "0.05",
        "--eval",                        # importante: testar a avaliação
        "--seed", "42"
    ]
    
    print(f"🔧 Executando: {' '.join(cmd)}")
    print("=" * 50)
    
    try:
        # Executar o teste
        result = subprocess.run(cmd, check=True, text=True)
        print(f"\n✅ Teste concluído com sucesso!")
        print(f"📁 Modelo salvo em: training/teste_correcao_dimensoes/")
        return True
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Erro durante o teste: {e}")
        print(f"Código de saída: {e.returncode}")
        return False
    
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        return False

if __name__ == "__main__":
    print("🎯 Testando correção do problema de dimensões")
    print("=" * 60)
    
    success = test_training()
    
    if success:
        print("\n🎉 TESTE PASSOU! O problema de dimensões foi corrigido.")
        print("Agora você pode treinar normalmente com:")
        print("python train.py ETTh1 meu_experimento --archive forecast_csv --epochs 100 --eval")
    else:
        print("\n❌ O teste falhou. Verifique os erros acima.")
    
    sys.exit(0 if success else 1)
