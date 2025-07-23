import pickle
root_dir = '/home/luis/Desktop/Trabalho final de Séries Temporias/TSI-forcasting/scripts/training/ETTh1/forecast_multivar_20250723_094329/eval_res.pkl'  # Adjust this path as needed

with open(root_dir, 'rb') as f:
    eval_res = pickle.load(f)

print("Resultados da avaliação:")
for k, v in eval_res.items():
    print(f"{k}: {v}")
