for seed in $(seq 0 4); do
  #python -u train.py exchange_rate forecast_univar --alpha 0.0005 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --archive forecast_csv_univar --repr-dims 320 --max-threads 8 --seed ${seed} --eval
  # multivar --- Weather and 
  python -u ../train.py exchange_rate forecast_multivar --archive forecast_csv --alpha 0.0005 --epochs 20 --kernels 1 2 4 8 16 32 64 128 --max-train-length 201 --batch-size 128 --repr-dims 320 --max-threads 8 --seed ${seed} --eval

done