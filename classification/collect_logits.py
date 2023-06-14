import argparse
import os
import torch
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm

# from subspace_inference import data, models, utils, losses
from subspace_inference import data, models

import gc
from training import get_weights
from collections import defaultdict

def get_hidden(model, x, model_name="WideResNet28x10"):
    x = x.cuda(non_blocking=True)
    if model_name == "WideResNet28x10":
        out = model.conv1(x)
        out = model.layer1(out)
        out = model.layer2(out)
        out = model.layer3(out)
        out = F.relu(model.bn1(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
    elif model_name == "VGG16":
        out = model.features(x)
        out = out.view(x.size(0), -1)
        for layer in list(model.classifier)[:-1]:
            out = layer(out)
    elif model_name == "PreResNet164":
        out = model.conv1(x)
        out = model.layer1(out)  # 32x32
        out = model.layer2(out)  # 16x16
        out = model.layer3(out)  # 8x8
        out = model.bn(out)
        out = model.relu(out)
        out = model.avgpool(out)
        out = out.view(out.size(0), -1)
    else:
        raise NotImplementedError
    return out


def get_stats(z, w, w_ranges):
    z_prod_w_range = z * w_ranges
    zw_indices = np.argmax(z_prod_w_range, 1)
    z_prod_w = z * w
    zw_k = np.array([z_prod_w[i][zw_indices[i]] for i in range(zw_indices.shape[0])])
    chosen_w_max = np.array([w_max[i][zw_indices[i]] for i in range(zw_indices.shape[0])])
    chosen_w_min = np.array([w_min[i][zw_indices[i]] for i in range(zw_indices.shape[0])])
    return z[zw_indices], zw_k, chosen_w_max, chosen_w_min


parser = argparse.ArgumentParser()
parser.add_argument('--dir', type=str, default=None, required=True, help='training directory (default: None)')
parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset name (default: CIFAR10)')
parser.add_argument('--data_path', type=str, default=None, required=True, metavar='PATH', help='path to datasets location (default: None)')
parser.add_argument('--use_test', dest='use_test', action='store_true', help='use test dataset instead of validation (default: False)')
parser.add_argument('--split_classes', type=int, default=None)
parser.add_argument('--batch_size', type=int, default=128, metavar='N', help='input batch size (default: 128)')
parser.add_argument('--num_workers', type=int, default=0, metavar='N', help='number of workers (default: 0)')
parser.add_argument('--collapsed', type=str)
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

print(f"RUN {args.seed} and COLLAPSED {args.collapsed}")

print('Loading dataset %s from %s' % (args.dataset, args.data_path))
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

test_ds = loaders['test']
del loaders
torch.cuda.empty_cache()

all_collapsed = args.collapsed.split()
all_weights = np.load(f"{args.dir}/collapsed_weights_{all_collapsed[0]}_{all_collapsed[-1]}.npy")

w_max, w_min = np.max(all_weights, 0), np.min(all_weights, 0)
w_ranges = w_max - w_min


# for each collapsed
for collapsed in tqdm(all_collapsed):

    # load model
    model = model_cfg.base(*model_cfg.args, num_classes=num_classes, **model_cfg.kwargs)
    model.to(args.device)

    PATH = os.path.join(args.dir, f"checkpoint-{collapsed}.pt")
    checkpoint = torch.load(PATH, map_location='cpu')
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()


    # free mem
    del checkpoint
    torch.cuda.empty_cache()

    """
    collect logits
    """
    all_output = []
    for batch_idx, data in enumerate(test_ds):
        features, labels = data
        x = features.cuda(non_blocking=True)
        all_output.append(model(x).cpu().detach().numpy())
        torch.cuda.empty_cache()
    output = np.concatenate(all_output, axis=0)

    if args.split_classes is None:
        with open(f"{args.dir}/logits_{collapsed}.npy", 'wb') as f:
            np.save(f, output)
    else:
        with open(f"{args.dir}/logits_{collapsed}_split_{args.split_classes}.npy", 'wb') as f:
            np.save(f, output)

    del output
    torch.cuda.empty_cache()


    """
    collect wmi stats
    """
    w = get_weights(model, args.model)[0]
    all_stats = defaultdict(list)
    for _, data in enumerate(test_ds):
        inputs, _ = data
        batch_z = get_hidden(model, inputs, args.model)
        results = map(
            get_stats, 
            batch_z.detach().cpu().numpy(),
            [w] * inputs.shape[0],
            [w_ranges] * inputs.shape[0]
        )
        for stats in list(results):
            for i in range(len(stats)):
                all_stats[i].append(stats[i])

    np_all_stats = dict()
    for k in all_stats.keys():
        np_all_stats[k] = np.array(all_stats[k])

    stats_file = f"wmi_stats_{collapsed}.npz" if args.split_classes is None else f"wmi_stats_{collapsed}_split_{args.split_classes}.npz"

    np.savez(
        os.path.join(args.dir, stats_file), 
        z=np_all_stats[0],
        zw=np_all_stats[1],
        w_max=np_all_stats[2],
        w_min=np_all_stats[3]
    )

    del model
    del all_stats
    del np_all_stats
    torch.cuda.empty_cache()
