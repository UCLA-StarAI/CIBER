import numpy as np
import torch
from torch.autograd import Variable
from numpy.random import default_rng
from collections import defaultdict
import os
import logging
import time
from fractions import Fraction


TRI_RADIUS = 2.45  # 2.297
# TRI_RADIUS = 2.297 # 2.33779
SAVE_WEIGHT_PATH = 'weights/'
SAVE_MODEL_PATH = 'models/'


def get_log(log_dir, log_name):
    if not log_dir:
        log_path = os.path.join('log/', log_name)
        os.mkdir(log_path)
        logging.basicConfig(
            filename=log_path + f'/{log_name}.log', level=logging.INFO
        )
    else:
        log_dir_path = os.path.join('log/', log_dir)
        log_path = os.path.join('log/', log_dir, log_name)
        if not os.path.exists(log_dir_path):        
            os.mkdir(log_dir_path)
        os.mkdir(log_path)
        logging.basicConfig(
            filename=log_path + f'/{log_name}.log', level=logging.INFO
        )
    return log_path


def set_weights(model, vector, device='cuda'):
    offset = 0
    for param in model.parameters():
        param.data.copy_(vector[offset:offset + param.numel()].view(param.size()).to(device))
        offset += param.numel()

def get_state_dict(w, model_params, weight_indices):
    for k in model_params.keys():
        s, e = weight_indices[k]
        layer_w = w[s:e].reshape(model_params[k].shape)
        model_params[k] = torch.from_numpy(layer_w)
    return model_params


def get_weight_indices(model_params):
    weight_indices = dict()    
    st_idx = 0
    for k in model_params.keys():
        shape = np.array(model_params[k].shape).prod()
        weight_indices[k] = (st_idx, st_idx + shape)
        st_idx += shape
    return weight_indices


def adjust_learning_rate(optimizer, factor):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= factor


def train_epoch(loader, model, criterion, optimizer, cuda=True, regression=True):
    loss_sum = 0.0
    stats_sum = defaultdict(float)
    correct = 0.0
    verb_stage = 0

    num_objects_current = 0
    num_batches = len(loader)

    model.train()    

    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, output, stats = criterion(model, input, target)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
            
        loss_sum += loss.data.item() * input.size(0)
        for key, value in stats.items():
            stats_sum[key] += value * input.size(0)

        num_objects_current += input.size(0)        
    
    return {
        'loss': loss_sum / num_objects_current,
        'accuracy': None if regression else correct / num_objects_current * 100.0,
        'stats': {key: value / num_objects_current for key, value in stats_sum.items()}
    }


def val_epoch(loader, model, criterion, cuda=True):
    loss_sum = 0.
    num_objects_current = 0
    
    for i, (input, target) in enumerate(loader):
        if cuda:
            input = input.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)

        loss, _, _ = criterion(model, input, target)

        loss_sum += loss.data.item() * input.size(0)
        num_objects_current += input.size(0)

    return loss_sum / num_objects_current


def log_gaussian_loss(output, target, sigma, no_dim, sum_reduce=True):
    exponent = -0.5 * (target - output) ** 2 / sigma ** 2
    log_coeff = -no_dim * torch.log(sigma) - 0.5 * no_dim * np.log(2 * np.pi)

    if sum_reduce:
        return -(log_coeff + exponent).sum()
    else:
        return -(log_coeff + exponent)


def to_variable(var=(), cuda=False, volatile=False):
    out = []
    for v in var:

        if isinstance(v, np.ndarray):
            v = torch.from_numpy(v).type(torch.FloatTensor)

        if not v.is_cuda and cuda:
            v = v.cuda()

        if not isinstance(v, Variable):
            v = Variable(v, volatile=volatile)

        out.append(v)
    return out


def likelihood_output(param_dict):

    last_w = param_dict['last_w']
    last_b = param_dict['last_b']

    example_x = param_dict['example_x']
    example_y = param_dict['example_y']
    
    n_vars = param_dict['n_vars']
    
    min_ws = param_dict['min_ws']
    max_ws = param_dict['max_ws']
    range_ws = param_dict['range_ws']
    min_bs = param_dict['min_bs']
    max_bs = param_dict['max_bs']
    range_bs = param_dict['range_bs']
    
    output = param_dict['output']
    np_z = param_dict['np_z']
    
    min_y = param_dict['min_y']
    max_y = param_dict['max_y']
    # epsilon = param_dict['epsilon']
    i = param_dict['test_index']
    j = param_dict['w_index']

    """
    formulate WMI problem
    """
    hlines = []
    # define weights
    # choose stochastic weights by ranges
    range_weights_z = range_ws * np.abs(np_z)
    logging.debug(f"n_vars = {n_vars}")
    logging.debug(f"np.count_nonzero(range_weights_z) = {np.count_nonzero(range_weights_z)}")
    # print(f"range_weights_z = {range_weights_z}")
    n_vars_x = min(n_vars, np.count_nonzero(range_weights_z))
    w_ind = np.argpartition(range_weights_z, -n_vars_x)[-n_vars_x:]
    # TODO: deal with all w having zero vol
    # w_ind = [] if n_vars_x == 0 else np.argpartition(range_weights_z, -n_vars_x)[-n_vars_x:]

    logging.debug(f"w_ind = {w_ind}")
    logging.debug(f"n_vars_x = {n_vars_x}")
    magic = 1e-4
    if n_vars_x == 0:
        hlines += [f"weight w_{wi} uniform {min_ws[wi] - 0.5 * magic} {max_ws[wi] + 0.5 * magic}" for wi in w_ind[:n_vars]]
    else:
        hlines += [f"weight w_{wi} uniform {min_ws[wi]} {max_ws[wi]}" for wi in w_ind]
    if max_bs[0] - min_bs[0] > 0:
        hlines.append(f"weight w_b uniform {min_bs[0]} {max_bs[0]}")
    else:
        hlines.append(f"weight w_b uniform {min_bs[0] - 0.5 * magic} {max_bs[0] + 0.5 * magic}")
    # if len(w_ind) == 0:
    #     assert max_bs[0] != min_bs[0], f"for test sample {i}, weight {j} it's fully deterministic!"

    # define triangular distributions
    v = output[1]
    radius = TRI_RADIUS * v**0.5
    hline = f"output y triangular {radius} "
    for wi in w_ind:
        hline += f"w_{wi} {float(np_z[wi])} "
    hline += f"w_b 1 "
    z_w_prod = np.multiply(np_z, last_w)
    del_z_w_prod = np.delete(z_w_prod, w_ind)
    hline += f"constant {np.sum(del_z_w_prod)}"
    hlines.append(hline)

    # output range
    hlines.append(f"output y range {min_y} {max_y}")
    hlines.append(f"output y query {example_y}")

    return hlines
