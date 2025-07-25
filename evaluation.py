import os
import pickle

root_dir = r'/home/luis/Desktop/Trabalho final de Séries Temporias/TSI-forcasting/'
# test = r"/home/luis/Desktop/Trabalho final de Séries Temporias/TSI-forcasting/scripts/training/ETTm2/forecast_multivar_20250723_123840/eval_res.pkl"

    # with open(root_dir, 'rb') as f:
    #     eval_data = pickle.load(f)

    # for horizon, values in eval_data['ours'].items():
    #     if 'norm' in values:
    #         mse = values['norm'].get('MSE', None)
    #         mae = values['norm'].get('MAE', None)
    #         if mse is not None and mae is not None:
    #             print(f"Horizon {horizon}: MSE = {mse}, MAE = {mae}")
    #         else:
    #             print(f"Horizon {horizon} does not have MSE or MAE values.")
    #     else:
    #         print(f"Horizon {horizon} does not have normalized values.")

training_dir = os.path.join(root_dir, "scripts/training/ETTh1")

resultados = {}
melhor_pasta = None
melhor_media = float("inf")
melhor_resultado = None

# 1. Carrega todos os resultados
for subdir in os.listdir(training_dir):
    subpath = os.path.join(training_dir, subdir)
    eval_path = os.path.join(subpath, "eval_res.pkl")

    if os.path.isdir(subpath) and os.path.isfile(eval_path):
        try:
            with open(eval_path, 'rb') as f:
                eval_data = pickle.load(f)
                resultados[subdir] = eval_data
        except Exception as e:
            print(f"[ERRO] Falha ao carregar {eval_path}: {e}")

# 2. Encontra a melhor pasta com base na média de norm_MSE e norm_MAE
for nome_pasta, res in resultados.items():
    if 'ours' not in res:
        continue

    norm_mse, norm_mae = [], []

    for horizonte, valores in res['ours'].items():
        if 'norm' in valores:
            norm_mse.append(valores['norm'].get('MSE', 0))
            norm_mae.append(valores['norm'].get('MAE', 0))

    if not norm_mse or not norm_mae:
        continue

    avg_mse = sum(norm_mse) / len(norm_mse)
    avg_mae = sum(norm_mae) / len(norm_mae)
    media = (avg_mse + avg_mae) / 2

    if media < melhor_media:
        melhor_media = media
        melhor_pasta = nome_pasta
        melhor_resultado = res

# 3. Mostra os resultados detalhados da melhor pasta
if melhor_resultado:
    print(f"\n🏆 MELHOR PASTA: {melhor_pasta}")
    print(f"📈 Média Geral (norm_MSE + norm_MAE) / 2: {melhor_media:.4f}")

    print("\n🔍 RESULTADOS POR HORIZONTE (NORMALIZADOS):")
    for horizonte in sorted(melhor_resultado['ours'].keys()):
        valores = melhor_resultado['ours'][horizonte].get('norm', {})
        mse = valores.get('MSE', None)
        mae = valores.get('MAE', None)

        if mse is not None and mae is not None:
            print(f"  ⏱️ Horizonte {horizonte}:")
            print(f"     🔸 MSE: {mse:.4f}")
            print(f"     🔹 MAE: {mae:.4f}")
else:
    print("❗Nenhum resultado válido encontrado.")
