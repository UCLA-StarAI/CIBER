import argparse
import os, sys
import time
import tabulate

import torch
import torch.nn.functional as F
import torchvision
import numpy as np
import pickle

from subspace_inference import data, models, utils, losses
from subspace_inference.posteriors import SWAG


def ece_score(py, y_test, n_bins=10):
    py = np.array(py)
    y_test = np.array(y_test)
    if y_test.ndim > 1:
        y_test = np.argmax(y_test, axis=1)
    py_index = np.argmax(py, axis=1)
    py_value = []
    for i in range(py.shape[0]):
        py_value.append(py[i, py_index[i]])
    py_value = np.array(py_value)
    acc, conf = np.zeros(n_bins), np.zeros(n_bins)
    Bm = np.zeros(n_bins)
    for m in range(n_bins):
        a, b = m / n_bins, (m + 1) / n_bins
        for i in range(py.shape[0]):
            if py_value[i] > a and py_value[i] <= b:
                Bm[m] += 1
                if py_index[i] == y_test[i]:
                    acc[m] += 1
                conf[m] += py_value[i]
        if Bm[m] != 0:
            acc[m] = acc[m] / Bm[m]
            conf[m] = conf[m] / Bm[m]
    ece = 0
    for m in range(n_bins):
        ece += Bm[m] * np.abs((acc[m] - conf[m]))
    return ece / sum(Bm)


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH', help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of workers (default: 0)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL', help='model name (default: None)')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--all_collapsed', nargs='+', type=int)
args = parser.parse_args()

# args.device = torch.device('cpu')
args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    print(f"use gpu {args.gpu}")
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

model_cfg = getattr(models, args.model)


"""
load test data
"""
loaders, num_classes = data.loaders(
    args.dataset,
    args.data_path,
    args.batch_size,
    args.num_workers,
    model_cfg.transform_train,
    model_cfg.transform_test,
    use_validation=not args.use_test,
    split_classes=args.split_classes
)

all_labels = []
for batch_idx, data in enumerate(loaders['test']):
    _, labels = data
    all_labels.append(labels)
labels = torch.cat(all_labels)
np_labels = labels.cpu().detach().numpy()

print(f"number of test data: {np_labels.shape[0]}")


"""
load results from wmi
"""
collapsed_i = 0
all_prob = np.load(f"{args.dir}/all_prob_{args.all_collapsed[collapsed_i]}.npz")
for collapsed_i in range(1, len(args.all_collapsed)):
    all_prob += np.load(f"{args.dir}/all_prob_{args.all_collapsed[collapsed_i]}.npz")
all_prob /= len(args.all_collapsed)

# negative likelihood
nll = [- np.log(all_prob[i, label]) for i, label in enumerate(np_labels)]
print(f"negative log likelihood: {np.mean(nll)}")

# accuracy
predictions = np.argmax(all_prob, axis=1)
acc = np.sum([predictions[i] == np_labels[i] for i in range(np_labels.shape[0])]) / np_labels.shape[0]
print(f"acc: {acc}")

# ece
ece = ece_score(all_prob, np_labels, n_bins=20)
print(f"ece: {ece}")

with open(os.path.join(args.dir, f'res-{len(args.all_collapsed)}.pkl'), 'wb') as f:
    pickle.dump(
        {"nll": np.mean(nll), "acc": acc, "ece": ece}, f
    )