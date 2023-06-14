import argparse
import os
from tqdm import tqdm
import torch
import torch.nn.functional as F
# import torchvision
import numpy as np
import gc

from training import get_weights
from subspace_inference import models

gc.collect()
torch.cuda.empty_cache()

# arguments
parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--collapsed', type=str)
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--model', type=str, default=None, required=True, metavar='MODEL', help='model name (default: None)')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--gpu', type=int, default=0)
args = parser.parse_args()

args.device = None
if torch.cuda.is_available():
    args.device = torch.device('cuda')
    print(f"use gpu {args.gpu}")
    torch.cuda.set_device(args.gpu)
else:
    args.device = torch.device('cpu')

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False    
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
np.random.seed(args.seed)

print('Using model %s' % args.model)
model_cfg = getattr(models, args.model)
num_classes = 100 if args.dataset == "CIFAR100" else 10

all_collapsed = args.collapsed.split()
collect_weights = []

print(f"RUN {args.seed} and COLLAPSED {args.collapsed}")

for collapsed in tqdm(all_collapsed):

    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.to(args.device)

    PATH = os.path.join(args.dir, f"checkpoint-{collapsed}.pt")
    checkpoint = torch.load(PATH)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    np_w = get_weights(model, args.model)
    collect_weights.append(np_w[0])

    # clean cache
    del model
    del checkpoint
    torch.cuda.empty_cache()

with open(f"{args.dir}/collapsed_weights_{all_collapsed[0]}_{all_collapsed[-1]}.npy", 'wb') as f:
    np.save(f, collect_weights)
