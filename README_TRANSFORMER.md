# TSI com Refinamento de Tendência via Transformer

## 📋 Resumo da Modificação

Esta implementação adiciona um **Transformer Encoder** para refinar a extração de tendência no modelo TSI (Time Series Representation Learning), mantendo **100% de compatibilidade** com o código original.

## 🏗️ Arquitetura

### Fluxo Original:
```
Entrada → Projeção Linear → Conv Dilatadas → [Tendência + Sazonalidade]
                                        ↓
                                Conv 1D (tendência)
                                FFT (sazonalidade)
```

### Fluxo com Transformer:
```
Entrada → Projeção Linear → Conv Dilatadas → [Tendência + Sazonalidade]
                                        ↓
                                Conv 1D → Transformer → Tendência Refinada
                                FFT → Sazonalidade
```

## 🚀 Novos Componentes

### 1. `TransformerEncoder`
- **Multi-Head Attention** para capturar dependências de longo prazo
- **Positional Encoding** para informação temporal
- **Feed-Forward Networks** com GELU
- **Layer Normalization** e **Dropout**

### 2. `TSIEncoderWithTransformer`
- Extensão do `TSIEncoder` original
- Parâmetro `use_transformer` para ativar/desativar
- Mantém interface idêntica
- Adiciona refinamento na tendência

### 3. `TSIWithTransformer`
- Versão estendida da classe `TSI`
- Novos métodos para análise de atenção
- Compatibilidade total com scripts existentes

## 🔧 Como Usar

### Uso Básico (Compatível)
```python
# Código original continua funcionando
from tsi import TSI

model = TSI(
    input_dims=7,
    kernels=[1, 2, 4, 8],
    alpha=0.05,
    max_train_length=512
)
```

### Novo Uso com Transformer
```python
from tsi import TSIWithTransformer

model = TSIWithTransformer(
    input_dims=7,
    kernels=[1, 2, 4, 8],
    alpha=0.05,
    max_train_length=512,
    # Novos parâmetros
    use_transformer=True,
    transformer_heads=4,
    transformer_depth=2,
    transformer_dropout=0.1
)
```

### Análise de Atenção
```python
# Obter pesos de atenção
attention_weights = model.get_attention_weights(data)

# Análise completa
model.analyze_trend_attention(data)
```

## 📁 Arquivos Modificados

### Novos Arquivos:
- `models/transformer_encoder.py` - Implementação do Transformer
- `exemplo_transformer.py` - Exemplos de uso
- `README_TRANSFORMER.md` - Esta documentação

### Arquivos Modificados:
- `tsi.py` - Adicionada classe `TSIWithTransformer`
- `models/encoder.py` - Adicionada classe `TSIEncoderWithTransformer`

## ⚙️ Parâmetros do Transformer

| Parâmetro | Descrição | Padrão |
|-----------|-----------|---------|
| `use_transformer` | Ativar/desativar Transformer | `True` |
| `transformer_heads` | Número de cabeças de atenção | `4` |
| `transformer_depth` | Número de camadas Transformer | `2` |
| `transformer_dropout` | Taxa de dropout | `0.1` |

## 🧪 Testes e Validação

Execute o arquivo de exemplo para validar:
```bash
python exemplo_transformer.py
```

### Testes Incluídos:
1. ✅ **Uso Básico** - Criação e treinamento
2. ✅ **Comparação** - TSI original vs Transformer
3. ✅ **Compatibilidade** - Interface idêntica
4. ✅ **Funcionalidades** - Novos métodos

## 🔍 Vantagens da Implementação

### 1. **Compatibilidade Total**
- Scripts existentes funcionam sem modificação
- Mesma interface de métodos (`fit`, `encode`, `save`, `load`)
- Parâmetros opcionais não quebram código legado

### 2. **Flexibilidade**
- Transformer pode ser ativado/desativado
- Parâmetros configuráveis
- Degradação graceful se desabilitado

### 3. **Melhoria Teórica**
- **Dependências de longo prazo**: Transformer captura relações temporais complexas
- **Atenção adaptativa**: Foca em partes relevantes da série temporal
- **Refinamento**: Aprimora tendência extraída pelas convoluções

### 4. **Análise Adicional**
- Visualização de pesos de atenção
- Interpretabilidade melhorada
- Debug e análise de comportamento

## 📊 Impacto na Performance

### Computational:
- **Mínimo** quando `use_transformer=False`
- **Moderado** quando habilitado (O(n²) na atenção)
- **Configurável** via `transformer_depth` e `transformer_heads`

### Memória:
- Overhead pequeno quando desabilitado
- Escalável com tamanho da sequência

## 🎯 Casos de Uso Recomendados

### Use Transformer quando:
- Séries longas com dependências complexas
- Padrões de tendência não-lineares
- Necessidade de interpretabilidade
- Dados com sazonalidade irregular

### Use versão original quando:
- Séries curtas
- Recursos computacionais limitados
- Padrões de tendência simples
- Prioridade na velocidade

## 🔮 Extensões Futuras

1. **Transformer na Sazonalidade**: Aplicar também na componente sazonal
2. **Atenção Cruzada**: Entre tendência e sazonalidade
3. **Atenção Esparsa**: Para séries muito longas
4. **Transformer Hierárquico**: Multi-escala temporal

## 📝 Exemplo Completo

```python
import numpy as np
from tsi import TSIWithTransformer
from datautils import load_forecast_csv

# Carregar dados
data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = load_forecast_csv('ETTh1')
train_data = data[:, train_slice]

# Criar modelo
model = TSIWithTransformer(
    input_dims=train_data.shape[-1],
    kernels=[1, 2, 4, 8, 16, 32, 64, 128],
    alpha=0.05,
    max_train_length=train_data.shape[1],
    output_dims=320,
    hidden_dims=64,
    depth=10,
    device='cuda',
    use_transformer=True,
    transformer_heads=4,
    transformer_depth=2
)

# Treinar
loss_log = model.fit(train_data, n_epochs=10, verbose=True)

# Encoding
representations = model.encode(train_data, mode='forecasting')

# Analisar atenção
attention_weights = model.get_attention_weights(train_data[:1, :100])
model.analyze_trend_attention(train_data[:1, :100])

# Salvar modelo
model.save('tsi_transformer_model.pth')
```

## ✅ Conclusão

A implementação adiciona capacidades avançadas de refinamento de tendência via Transformer mantendo **total compatibilidade** com o código existente. O design permite migração gradual e experimentação sem riscos ao sistema atual.
