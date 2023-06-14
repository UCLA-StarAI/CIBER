import argparse
import torch
from data_process import get_regression_data
import losses
from model import RegressionRunner, RegNet, HiddenRegNet
import logging
import time
import numpy as np
import os
import pickle
from utils import get_log


class ModelCfg():
    base = None
    args = list()
    kwargs = {}


def run(ARGS, data, model, log_path=None, uci_small=False, w_freq=1, hidden=False):
    res = {}
    
    logging.info(f"data standard deviation is: {data.Y_std[0][0]}")
    res['Y_std'] = data.Y_std[0][0]
    logging.info("training with early stopping ...")
    start = time.time()
    _, sample_weights = model.fit(data.X_train, data.Y_train)
    fit_time = time.time() - start
    logging.info(f"training finishes in {fit_time} seconds")
    res['fit_time'] = fit_time

    np_weights = [w.numpy() for w in sample_weights]

    
    start = time.time()
    if hidden:
        likelihood_path = model.wmi_hidden(data, np_weights, log_path=log_path, w_freq=w_freq, n_vars=2)
    else:
        likelihood_path, m = model.wmi(data, np_weights, log_path=log_path, w_freq=w_freq, n_vars=2)
    log_time = time.time() - start
    res['write_log_time'] = log_time
    res['likelihood_path'] = likelihood_path

    if not hidden:
        """
        rmse
        """
        d = data.Y_test - m
        du = d * data.Y_std

        res['test_rmse'] = np.average(d**2)**0.5
        res['test_rmse_unnormalized'] = np.average(du**2)**0.5

    res.update(ARGS.__dict__)
    logging.info(res)
    return res


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default='RegNet', nargs='?', type=str)
    parser.add_argument("--dataset", default='energy', nargs='?', type=str)
    parser.add_argument("--split", default=0, nargs='?', type=int)
    parser.add_argument("--seed", default=666, nargs='?', type=int)
    parser.add_argument('--epochs', type=int, default=50, metavar='N', help='number of epochs to train (default: 200)')

    parser.add_argument('--lr_init', type=float, default=0.01, metavar='LR', help='initial learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=400, metavar='N', help='input batch size (default: 128)')    
    parser.add_argument('--log_dir', type=str, default=None)
    parser.add_argument('--weight_init', type=str, default='xavier_uniform_')
    parser.add_argument('--no_schedule', action='store_true', help='store schedule')
    parser.add_argument('--uci-small', action='store_true')
    parser.add_argument('--double-bias-lr', action='store_true')
    
    parser.add_argument('--factor', type=float, default=0.8, help='decay factor of scheduler')
    parser.add_argument('--patience', type=int, default=40, help='patience of scheduler')
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--hidden', action='store_true')

    args = parser.parse_args()
    
    # random seed
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False    
    torch.manual_seed(args.seed + args.split)
    torch.cuda.manual_seed(args.seed + args.split)
    np.random.seed(args.seed + args.split)

    if torch.cuda.is_available():
        print(f"use gpu {args.gpu}")
        torch.cuda.set_device(args.gpu)

    # prepare logging
    log_name = "-".join([args.dataset, str(args.seed), str(args.split)])
    log_path = get_log(args.log_dir, log_name)
    print(f"log path = {log_path}")

    # prepare dataset
    dataset = get_regression_data(args.dataset, split=args.split)
    print(f"dataset {args.dataset} is prepared")

    """
    prepare model
    """
    model_cfg = ModelCfg()
    model_cfg.base = HiddenRegNet if args.hidden else RegNet 
    if args.uci_small:
        model_cfg.kwargs['dimensions'] = [50] if dataset.N < 40000 else [100]
    else:
        if dataset.N > 6000:
            model_cfg.kwargs['dimensions'] = [1000, 1000, 500, 50, 2]
        else:
            model_cfg.kwargs['dimensions'] = [1000, 500, 50, 2]

    # heteroscedastic
    output_dim = 2
    regclass = RegressionRunner
    regression_model = regclass(
        base = model_cfg.base,
        epochs = args.epochs,
        criterion = losses.GaussianLikelihood,
        batch_size=args.batch_size,
        lr_init=args.lr_init,
        momentum = args.momentum, 
        wd=args.wd,
        use_cuda = torch.cuda.is_available(),
        double_bias_lr=args.double_bias_lr,
        const_lr=args.no_schedule,
        factor=args.factor,
        patience=args.patience,
        data=dataset,
        # RegNet parameters
        weight_init=args.weight_init if args.weight_init != "default" else None,
        input_dim=dataset.D, 
        output_dim=output_dim, 
        apply_var=True,
        **model_cfg.kwargs
    )

    """
    run
    """
    mname = args.model
    bb_args = argparse.Namespace(
        model=mname, 
        dataset=args.dataset, 
        split=args.split, 
        seed=args.seed,
        factor=args.factor, 
        patience=args.patience, 
        wd=args.wd, 
        lr_init=args.lr_init, 
        epochs=args.epochs, 
        batch=args.batch_size
    )

    bb_result = run(
        bb_args, 
        data=dataset, 
        model=regression_model,
        log_path=log_path,
        uci_small=args.uci_small,
        hidden=args.hidden
    )
    print([(k, bb_result[k]) for k in sorted(bb_result)])

    f = open(log_path + "/res.pkl","wb")
    pickle.dump(bb_result,f)
    f.close()

