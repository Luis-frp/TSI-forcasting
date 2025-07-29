import argparse
import os
import time
import datetime
import math
import numpy as np
import tasks
import datautils
from utils import init_dl_program, name_with_datetime, pkl_save, data_dropout

# import methods
from tsi import TSI


def save_checkpoint_callback(
    save_every=1,
    unit='epoch'
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', help='The dataset name')
    parser.add_argument('run_name', help='The folder name used to save model, output and evaluation metrics. This can be set to any word')
    parser.add_argument('--archive', type=str, required=True, help='The archive name that the dataset belongs to. This can be set to forecast_csv, or forecast_csv_univar')
    parser.add_argument('--gpu', type=int, default=0, help='The gpu no. used for training and inference (defaults to 0)')
    parser.add_argument('--batch-size', type=int, default=8, help='The batch size (defaults to 8)')
    parser.add_argument('--lr', type=float, default=0.001, help='The learning rate (defaults to 0.001)')
    parser.add_argument('--repr-dims', type=int, default=320, help='The representation dimension (defaults to 320)')
    parser.add_argument('--max-train-length', type=int, default=3000, help='For sequence with a length greater than <max_train_length>, it would be cropped into some sequences, each of which has a length less than <max_train_length> (defaults to 3000)')
    parser.add_argument('--iters', type=int, default=None, help='The number of iterations')
    parser.add_argument('--epochs', type=int, default=None, help='The number of epochs')
    parser.add_argument('--save-every', type=int, default=None, help='Save the checkpoint every <save_every> iterations/epochs')
    parser.add_argument('--seed', type=int, default=None, help='The random seed')
    parser.add_argument('--max-threads', type=int, default=None, help='The maximum allowed number of threads used by this process')
    parser.add_argument('--eval', action="store_true", help='Whether to perform evaluation after training')

    parser.add_argument('--kernels', type=int, nargs='+', default=[1, 2, 4, 8, 16, 32, 64, 128], help='The kernel sizes used in the mixture of AR expert layers')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Weighting hyperparameter for loss function')
    
    # Argumentos para refinador Transformer
    parser.add_argument('--use-transformer-refiner', action='store_true', default=True, help='Use Transformer as refiner after dilated convolutions')
    parser.add_argument('--no-use-transformer-refiner', dest='use_transformer_refiner', action='store_false', help='Disable Transformer refiner')
    parser.add_argument('--transformer-type', type=str, default='temporal', choices=['temporal', 'informer'], help='Type of transformer: temporal or informer')
    parser.add_argument('--transformer-heads', type=int, default=4, help='Number of attention heads in Transformer refiner (defaults to 4)')
    parser.add_argument('--transformer-depth', type=int, default=2, help='Number of Transformer encoder layers in refiner (defaults to 2)')
    parser.add_argument('--transformer-dropout', type=float, default=0.1, help='Dropout rate for Transformer refiner (defaults to 0.1)')
    parser.add_argument('--informer-factor', type=int, default=5, help='Factor for ProbSparse attention in Informer (defaults to 5)')
    parser.add_argument('--informer-distil', action='store_true', default=True, help='Enable distilling operation in Informer')
    parser.add_argument('--no-informer-distil', dest='informer_distil', action='store_false', help='Disable distilling operation in Informer')

    

    args = parser.parse_args()

    print("Dataset:", args.dataset)
    print("Arguments:", str(args))
    
    device = init_dl_program(args.gpu, seed=args.seed, max_threads=args.max_threads)

    if args.archive == 'forecast_csv':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_csv_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_csv(args.dataset, univar=True)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset)
        train_data = data[:, train_slice]
    elif args.archive == 'forecast_npy_univar':
        task_type = 'forecasting'
        data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols = datautils.load_forecast_npy(args.dataset, univar=True)
        train_data = data[:, train_slice]
    else:
        raise ValueError(f"Archive type {args.archive} is not supported.")

    config = dict(
        batch_size=args.batch_size,
        lr=args.lr,
        output_dims=args.repr_dims,
        # Parâmetros do Transformer refinador
        use_transformer_refiner=args.use_transformer_refiner,
        transformer_type=args.transformer_type,
        transformer_heads=args.transformer_heads,
        transformer_depth=args.transformer_depth,
        transformer_dropout=args.transformer_dropout,
        informer_factor=args.informer_factor,
        informer_distil=args.informer_distil,
    )
    
    if args.save_every is not None:
        unit = 'epoch' if args.epochs is not None else 'iter'
        config[f'after_{unit}_callback'] = save_checkpoint_callback(args.save_every, unit)

    run_dir = f"training/{args.dataset}/{name_with_datetime(args.run_name)}"

    os.makedirs(run_dir, exist_ok=True)
    
    t = time.time()

    model = TSI(
        input_dims=train_data.shape[-1],
        kernels=args.kernels,
        alpha=args.alpha,
        max_train_length=args.max_train_length,
        device=device,
        **config
    )

    loss_log = model.fit(
        train_data,
        n_epochs=args.epochs,
        n_iters=args.iters,
        verbose=True
    )
    
    model.save(f'{run_dir}/model.pkl')

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    if args.eval:
        out, eval_res = tasks.eval_forecasting(model, data, train_slice, valid_slice, test_slice, scaler, pred_lens, n_covariate_cols, args.max_train_length-1, args.dataset)
        print('Evaluation result:', eval_res)
        pkl_save(f'{run_dir}/eval_res.pkl', eval_res)
        pkl_save(f'{run_dir}/out.pkl', out)

    print("Finished.")