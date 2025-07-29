#!/bin/bash
# Script para testar TSI com Informer-Inspired Transformer em diferentes configurações

echo "🚀 Testando TSI com Informer-Inspired Transformer..."

# Experimento 1: Informer com destilação (recomendado)
echo "🔄 Experimento 1: Informer com destilação"
for seed in $(seq 5 7); do
  echo "   Seed: ${seed}"
  python -u ../train.py ETTm2 forecast_multivar_informer_distil \
    --archive forecast_csv \
    --use-transformer-refiner \
    --transformer-type informer \
    --transformer-heads 8 \
    --transformer-depth 3 \
    --transformer-dropout 0.1 \
    --informer-factor 5 \
    --informer-distil \
    --alpha 0.0005 \
    --epochs 15 \
    --kernels 1 2 4 8 16 32 64 128 \
    --max-train-length 201 \
    --batch-size 128 \
    --repr-dims 320 \
    --max-threads 8 \
    --seed ${seed} \
    --eval
done

echo ""

# Experimento 2: Informer sem destilação (comparação)
echo "🔄 Experimento 2: Informer sem destilação"
for seed in $(seq 8 9); do
  echo "   Seed: ${seed}"
  python -u ../train.py ETTm2 forecast_multivar_informer_no_distil \
    --archive forecast_csv \
    --use-transformer-refiner \
    --transformer-type informer \
    --transformer-heads 8 \
    --transformer-depth 3 \
    --transformer-dropout 0.1 \
    --informer-factor 5 \
    --no-informer-distil \
    --alpha 0.0005 \
    --epochs 15 \
    --kernels 1 2 4 8 16 32 64 128 \
    --max-train-length 201 \
    --batch-size 128 \
    --repr-dims 320 \
    --max-threads 8 \
    --seed ${seed} \
    --eval
done

echo ""

# Experimento 3: Comparação com Temporal Transformer original
echo "🔄 Experimento 3: Temporal Transformer (baseline)"
python -u ../train.py ETTm2 forecast_multivar_temporal_comparison \
  --archive forecast_csv \
  --use-transformer-refiner \
  --transformer-type temporal \
  --transformer-heads 8 \
  --transformer-depth 3 \
  --transformer-dropout 0.1 \
  --alpha 0.0005 \
  --epochs 15 \
  --kernels 1 2 4 8 16 32 64 128 \
  --max-train-length 201 \
  --batch-size 128 \
  --repr-dims 320 \
  --max-threads 8 \
  --seed 5 \
  --eval

echo ""
echo "✅ Experimentos Informer concluídos!"
echo "📊 Resultados salvos em training/ETTm2/"
echo ""
echo "🔍 Para analisar resultados, execute: python evaluation.py"