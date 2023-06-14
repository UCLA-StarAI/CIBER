import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

from utils import train_epoch, val_epoch
from utils import get_weight_indices, get_state_dict, set_weights, likelihood_output
from utils import TRI_RADIUS

import logging
import time
import os
from collections import deque, defaultdict
import csv


def flatten(lst):
    tmp = [i.contiguous().view(-1,1) for i in lst]
    return torch.cat(tmp).view(-1)


class SplitDim(nn.Module):
    def __init__(self, nonlin_col=1, nonlin_type=torch.nn.functional.softplus, correction = True):
        super(SplitDim, self).__init__()
        self.nonlinearity = nonlin_type
        self.col = nonlin_col

        if correction:
            self.var = torch.nn.Parameter(torch.zeros(1))
        else:
            self.register_buffer('var', torch.ones(1, requires_grad = False)*-15.)

        self.correction = correction

    def forward(self, input):
        transformed_output = self.nonlinearity(input[:,self.col])
        
        transformed_output = (transformed_output + self.nonlinearity(self.var))
        stack_list = [input[:,:self.col], transformed_output.view(-1,1)]
        if self.col+1 < input.size(1):
            stack_list.append(input[:,(self.col+1):])
                
        output = torch.cat(stack_list,1)
        return output


class RegNet(nn.Sequential):
    def __init__(self, dimensions, input_dim=1, output_dim=2, apply_var=True):
        super(RegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())
            
        self.add_module('var_split', SplitDim(correction=apply_var))

    def forward(self, x, output_hidden=False):
        if not output_hidden:
            return super().forward(x)
        else:
            for module in list(self._modules.values())[:-2]:
                x = module(x)                
            return x


class HiddenRegNet(nn.Sequential):
    def __init__(self, dimensions, input_dim=1, output_dim=2, apply_var=True):
        super(HiddenRegNet, self).__init__()
        self.dimensions = [input_dim, *dimensions, output_dim]
        for i in range(len(self.dimensions) - 1):
            self.add_module('linear%d' % i, torch.nn.Linear(self.dimensions[i], self.dimensions[i + 1]))
            if i < len(self.dimensions) - 2:
                self.add_module('relu%d' % i, torch.nn.ReLU())
            
        self.add_module('var_split', SplitDim(correction=apply_var))

    def forward(self, x, output_hidden=0):
        if output_hidden >= 0:
            return super().forward(x)
        else:
            for module in list(self._modules.values())[:output_hidden]:
                x = module(x)
            return x


class RegressionRunner():
    def __init__(self, base, epochs, criterion, batch_size=50, lr_init=1e-2, momentum=0.9, wd=1e-4, sample_start=50, use_cuda=True, double_bias_lr=True, model_variance=True, const_lr=False, factor=0.8, patience=40, data=None, weight_init='xavier_uniform_', *args, **kwargs):

        self.model = base(*args, **kwargs)
        num_pars = 0
        for p in self.model.parameters():
            num_pars += p.numel()
        
        if use_cuda:
            self.model.cuda()

        # initialize the model (for uci large)
        def init_weights(m, weight_init):
            if type(m) == nn.Linear:
                if weight_init == "xavier_uniform_":
                    torch.nn.init.xavier_uniform_(m.weight)
                elif weight_init == "kaiming_normal_":
                    torch.nn.init.kaiming_normal_(m.weight)
                elif weight_init == "xavier_normal_":
                    torch.nn.init.xavier_normal_(m.weight)
                else:
                    raise NotImplementedError
        if weight_init:
            self.model.apply(lambda m: init_weights(m, weight_init))

        self.use_cuda = use_cuda
        self.device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        if not double_bias_lr:
            pars = self.model.parameters()
        else:
            pars = []
            for name, module in self.model.named_parameters():
                if 'bias' in str(name):
                    print('Doubling lr of ', name)
                    pars.append({'params':module, 'lr':2.0 * lr_init})
                else:
                    pars.append({'params':module, 'lr':lr_init})
       
        self.optimizer = torch.optim.SGD(pars, lr=lr_init, momentum=momentum, weight_decay=wd)

        self.const_lr = const_lr
        self.patience = patience
        self.factor = factor
        self.epochs = epochs
        self.lr_init = lr_init

        if data:
            self.data = data

        self.batch_size = batch_size
        self.criterion = criterion(noise_var = None) if model_variance else criterion(noise_var = 1.0)

        if self.criterion.noise_var is not None:
            self.var = self.criterion.noise_var


    def train(self, model, loader, optimizer, criterion, lr_init=1e-2, epochs=3000, print_freq=100, use_cuda=True, const_lr=False, factor=0.8, 
    patience=40, n_weights=40):
        if const_lr:
            lr = lr_init
        else:
            lr = lr_init
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, factor=factor, patience=patience, verbose=True
            )

        all_weights = deque([], maxlen=n_weights)

        min_val_loss = np.inf
        n_bad_epochs = 0
        train_res_list = []
        for epoch in range(epochs):
            train_res = train_epoch(loader, model, criterion, optimizer, cuda=use_cuda, regression=True)
            train_res_list.append(train_res)
            
            val_loss = val_epoch(self.val_data_loader, model, criterion, cuda=use_cuda)
            
            w = flatten([param.detach().cpu() for param in model.parameters()])
            all_weights.append(w)

            if not const_lr:
                scheduler.step(val_loss)

            # early stopping
            if val_loss < min_val_loss:
                n_bad_epochs = 0
                min_val_loss = val_loss
            else:
                n_bad_epochs += 1

            if epoch > max(epochs * 2 / 3, n_weights) and n_bad_epochs == patience:
                logging.info(f"early stopping at epoch = {epoch}")
                break
            
            if (epoch % print_freq == 0 or epoch == epochs - 1):
                print('Epoch %d. LR: %g. Loss: %.4f' % (epoch, lr, train_res['loss']))

        return train_res_list, all_weights
 
    def fit(self, features, labels):
        print(f"len of labels = {len(labels)}")
        indices = np.arange(0, len(labels))
        np.random.shuffle(indices)
        n_train = int(len(labels) * 8 / 9)  # 8:1:1 train:val:test split

        self.features, self.labels = torch.FloatTensor(features), torch.FloatTensor(labels)

        # construct data loader
        start = time.time()
        self.data_loader = DataLoader(TensorDataset(self.features, self.labels), batch_size = self.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices[:n_train]))
        
        self.val_data_loader = DataLoader(TensorDataset(self.features, self.labels), batch_size = self.batch_size, sampler=torch.utils.data.SubsetRandomSampler(indices[n_train:]))
        prepare_time = time.time() - start
        print(f"preparing DataLoader takes time {prepare_time}")

        # training
        result = self.train(
            model=self.model, loader=self.data_loader, optimizer=self.optimizer, criterion=self.criterion, 
            lr_init=self.lr_init, use_cuda=self.use_cuda, 
            epochs=self.epochs, const_lr=self.const_lr, factor=self.factor,
            patience=self.patience
        )

        if self.criterion.noise_var is not None:
            preds, targets = utils.predictions(model=self.model, test_loader=self.data_loader, regression=True,cuda=self.use_cuda)
            self.var = np.power(np.linalg.norm(preds - targets), 2.0) / targets.shape[0]
            print(self.var)

        return result


    def wmi(self, dataset, sample_weights, log_path='', w_freq=1, n_vars=5):
        device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        model_params = self.model.state_dict()
        weight_indices = get_weight_indices(model_params)

        last_layer_idx = len(self.model.dimensions) - 2
        # get last layer weight range
        w_name = 'linear%d' % last_layer_idx + '.weight'
        b_name = 'linear%d' % last_layer_idx + '.bias'

        ws, bs = [], []
        for w in sample_weights:
            s, e = weight_indices[w_name]
            ws.append(w[s:e].reshape(model_params[w_name].shape)[0])
            s, e = weight_indices[b_name]
            bs.append(w[s:e].reshape(model_params[b_name].shape)[0])
        ws = np.array(ws)
        bs = np.array(bs)

        max_ws, min_ws = np.max(ws, axis=0), np.min(ws, axis=0)
        max_bs, min_bs = np.max(bs, axis=0).reshape(-1), np.min(bs, axis=0).reshape(-1)

        pad = 1e-3
        max_y = np.max(dataset.Y_test) + pad
        min_y = np.min(dataset.Y_test) - pad

        # create wmi problems
        likelihood_path = f"{log_path}/wmi"
        if not os.path.isdir(likelihood_path):
            os.mkdir(likelihood_path)
        print(f"likelihood problems will be saved at {likelihood_path}")

        problem_collections = []
        prediction_collections = []
        cnt = 1
        test_indices = defaultdict(list)
        for i in range(len(dataset.X_test)):  # for each x
            wmi_prediction = []
            for j, w in enumerate(sample_weights):  # for each collapsed sample
                if not j % w_freq == 0:
                    continue

                param_dict = dict()
                param_dict['last_w'] = ws[j]
                param_dict['last_b'] = bs[j]
                param_dict['example_x'] = dataset.X_test[i].reshape(1,-1)
                param_dict['example_y'] = dataset.Y_test[i][0]
                param_dict['n_vars'] = n_vars
                param_dict['min_ws'] = min_ws
                param_dict['max_ws'] = max_ws
                param_dict['range_ws'] = max_ws - min_ws
                param_dict['min_bs'] = min_bs
                param_dict['max_bs'] = max_bs
                param_dict['range_bs'] = max_bs - min_bs
                param_dict['min_y'] = min_y
                param_dict['max_y'] = max_y
                # param_dict['epsilon'] = epsilon
                param_dict['test_index'] = i
                param_dict['w_index'] = j
                # get output
                example_x = dataset.X_test[i].reshape(1,-1)
                set_weights(self.model, torch.from_numpy(w))
                output = self.model((torch.FloatTensor(example_x).to(device)))
                param_dict['output'] = output.detach().cpu().numpy()[0]
                # get latent
                z = self.model((torch.FloatTensor(example_x).to(device)), output_hidden=True)
                param_dict['np_z'] = z.detach().cpu().numpy()[0]

                """
                likelihood
                """
                hlines = likelihood_output(param_dict)
                with open(f"{likelihood_path}/input_{cnt}.txt", "w") as f:
                    f.writelines([hline + '\n' for hline in hlines])

                test_indices[i].append(cnt)
                cnt += 1

                """
                prediction
                """
                # choose stochastic weights by ranges
                np_z = param_dict['np_z']
                last_w = param_dict['last_w']
                range_ws = param_dict['range_ws']
                range_weights_z = range_ws * np.abs(np_z)
                n_vars_x = min(n_vars, np.count_nonzero(range_weights_z))
                w_ind = np.argpartition(range_weights_z, -n_vars_x)[-n_vars_x:]
                if n_vars_x == 0:
                    w_ind = w_ind[:n_vars]
                # compute prediction
                z_w_prod = np.multiply(np_z, last_w)
                prediction = np.sum(np.delete(z_w_prod, w_ind))
                mean_ws = (max_ws + min_ws) / 2
                for wi in w_ind:
                    prediction += float(np_z[wi]) * mean_ws[wi]
                prediction += (min_bs[0] + max_bs[0]) / 2
                wmi_prediction.append(prediction)
            prediction_collections.append(np.mean(wmi_prediction))
        prediction_collections = np.array(prediction_collections).reshape(-1, 1)

        field_names = ["idx", "cnt"]
        rows = [{"idx": k, "cnt": v} for k, v in test_indices.items()]
        with open(f"{likelihood_path}/test_indices.csv", 'w') as f:
            writer = csv.DictWriter(f, fieldnames = field_names)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"total number of likelihood problems: {cnt - 1}")

        return likelihood_path, prediction_collections


    def wmi_hidden(self, dataset, sample_weights, log_path='', w_freq=1, n_vars=5):
        device = torch.device('cuda') if self.use_cuda else torch.device('cpu')

        model_params = self.model.state_dict()
        weight_indices = get_weight_indices(model_params)

        # collect weights
        last_second_layer_idx = len(self.model.dimensions) - 3
        w2_name = 'linear%d' % last_second_layer_idx + '.weight'
        b2_name = 'linear%d' % last_second_layer_idx + '.bias'

        last_layer_idx = len(self.model.dimensions) - 2
        w_name = 'linear%d' % last_layer_idx + '.weight'
        b_name = 'linear%d' % last_layer_idx + '.bias'

        w2s, b2s = [], []
        z_idx = 0
        for w in sample_weights:
            s, e = weight_indices[w2_name]
            w2s.append(w[s:e].reshape(model_params[w2_name].shape)[z_idx])
            s, e = weight_indices[b2_name]
            b2s.append(w[s:e].reshape(model_params[b2_name].shape)[z_idx])

        w2s = np.array(w2s)
        b2s = np.array(b2s)

        max_w2s, min_w2s = np.max(w2s, axis=0), np.min(w2s, axis=0)
        max_b2s, min_b2s = np.max(b2s, axis=0).reshape(-1), np.min(b2s, axis=0).reshape(-1)
        range_w2s = max_w2s - min_w2s

        pad = 1e-3  # TODO: choose a proper pad
        max_y = np.max(dataset.Y_test) + pad
        min_y = np.min(dataset.Y_test) - pad

        # create wmi problems
        likelihood_path = f"{log_path}/wmi"
        if not os.path.isdir(likelihood_path):
            os.mkdir(likelihood_path)
        print(f"WMI problems will be saved at {likelihood_path}")

        problem_collections = []
        cnt = 1
        test_indices = defaultdict(list)
        for i in range(len(dataset.X_test)):  # for each x
            for j, w in enumerate(sample_weights):  # for each collapsed sample
                if not j % w_freq == 0:
                    continue

                example_x = dataset.X_test[i].reshape(1,-1)
                example_x = (torch.FloatTensor(example_x).to(device))
                example_y = dataset.Y_test[i][0]

                set_weights(self.model, torch.from_numpy(w))
                output = self.model(example_x).cpu().detach().numpy()[0]
                z = self.model(example_x, output_hidden=-2).cpu().detach().numpy()[0]
                h = self.model(example_x, output_hidden=-4).cpu().detach().numpy()[0]


                """
                formulate WMI problem
                """
                hlines = []
                # choose stochastic weights by ranges
                range_weights_z = range_w2s * np.abs(h)
                n_vars_x = min(n_vars, np.count_nonzero(range_weights_z))
                w_ind = np.argpartition(range_weights_z, -n_vars_x)[-n_vars_x:]
                
                magic = 1e-4
                if n_vars_x == 0:
                    hlines += [f"weight w_{wi} uniform {min_w2s[wi] - 0.5 * magic} {max_w2s[wi] + 0.5 * magic}" for wi in w_ind[:n_vars]]
                else:
                    hlines += [f"weight w_{wi} uniform {min_w2s[wi]} {max_w2s[wi]}" for wi in w_ind]
                if max_b2s[0] - min_b2s[0] > 0:
                    hlines.append(f"weight w_b uniform {min_b2s[0]} {max_b2s[0]}")
                else:
                    hlines.append(f"weight w_b uniform {min_b2s[0] - 0.5 * magic} {max_b2s[0] + 0.5 * magic}")

                last_w, last_b = w2s[j], b2s[j]
                # ReLu
                h_w_prod = np.multiply(h, last_w)
                del_h_w_prod = np.delete(h_w_prod, w_ind)
                relu = "relu "
                for wi in w_ind:
                    relu += f"w_{wi} {h[wi]} "
                relu += "w_b 1 "
                relu += f"constant {np.sum(del_h_w_prod)}"
                hlines.append(relu)

                # last layer
                s, e = weight_indices[w_name]
                w_z = w[s:e].reshape(model_params[w_name].shape)[0]
                s, e = weight_indices[b_name]
                b_z = w[s:e].reshape(model_params[b_name].shape)[0]

                # define triangular distributions
                radius = TRI_RADIUS * output[1]**0.5
                z_w_prod = np.multiply(z, w_z)
                del_z_w_prod = np.delete(z_w_prod, [z_idx])
                hlines.append(f"output y triangular {radius} relu {w_z[z_idx]} constant {np.sum(del_z_w_prod) + b_z}")

                # output range
                hlines.append(f"output y range {min_y} {max_y}")
                hlines.append(f"output y query {example_y}")

                with open(f"{likelihood_path}/input_{cnt}.txt", "w") as f:
                    f.writelines([hline + '\n' for hline in hlines])
                
                test_indices[i].append(cnt)
                cnt += 1
        
        field_names = ["idx", "cnt"]
        rows = [{"idx": k, "cnt": v} for k, v in test_indices.items()]
        with open(f"{likelihood_path}/test_indices.csv", 'w') as f:
            writer = csv.DictWriter(f, fieldnames = field_names)
            writer.writeheader()
            writer.writerows(rows)
        logging.info(f"total number of likelihood problems: {cnt - 1}")

        return likelihood_path