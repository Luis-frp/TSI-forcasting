#!/bin/bash

echo "🚀 Iniciando experimentos TSI com Transformer Refiner..."

# Experimento 1: Com Transformer Refiner (Temporal - Arquitetura Híbrida)
# echo "📊 Testando com Temporal Transformer Refiner..."
# for seed in $(seq 5 9); do
#   echo "🔄 Seed: ${seed} - Temporal Transformer Refiner"
#   python -u ../train.py ETTh1 forecast_multivar_temporal_refiner \
#     --archive forecast_csv \
#     --use-transformer-refiner \
#     --transformer-type temporal \
#     --transformer-heads 8 \
#     --transformer-depth 2 \
#     --transformer-dropout 0.1 \
#     --alpha 0.0005 \
#     --epochs 20 \
#     --kernels 1 2 4 8 16 32 64 128 \
#     --max-train-length 201 \
#     --batch-size 128 \
#     --repr-dims 320 \
#     --max-threads 8 \
#     --seed ${seed} \
#     --eval
# done

echo "✅ Experimentos com Temporal Transformer Refiner concluídos!"

# Experimento 2: Com Informer Transformer Refiner (Nova arquitetura)
echo "📊 Testando com Informer Transformer Refiner..."
for seed in $(seq 5 9); do
  echo "🔄 Seed: ${seed} - Informer Transformer Refiner"
  python -u ../train.py ETTm2 forecast_multivar_informer_refiner \
    --archive forecast_csv \
    --use-transformer-refiner \
    --transformer-type informer \
    --transformer-heads 4 \
    --transformer-depth 3 \
    --transformer-dropout 0.1 \
    --informer-factor 5 \
    --informer-distil \
    --alpha 0.0005 \
    --epochs 20 \
    --kernels 1 2 4 8 16 32 64 128 \
    --max-train-length 201 \
    --batch-size 128 \
    --repr-dims 320 \
    --max-threads 8 \
    --seed ${seed} \
    --eval
done
