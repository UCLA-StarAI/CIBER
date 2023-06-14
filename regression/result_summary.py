import numpy as np
import pickle
import json
import os
from collections import defaultdict
import numpy as np
import argparse

uci_small_datasests = ['boston', 'concrete', 'naval', 'yacht', 'energy']
uci_large_datasets = ['wilson_elevators', 'wilson_protein', 'wilson_pol', 'wilson_keggdirected', 'wilson_keggundirected', 'wilson_skillcraft']

def get_datasetname(dataset):
    if dataset in uci_small_datasests:
        return dataset
    for dataset_name in uci_large_datasets:
        if dataset in dataset_name:
            return dataset_name

def get_split(folder):
    return int(folder.split("-")[-1])

def get_epsilon_results(fp):
    with open(fp, 'r') as f:
        lines = f.read().splitlines()
        return (float(lines[0]), float(lines[1]))

def get_rmse_results(fp):
    with open(fp, 'r') as f:
        return float(f.read().splitlines()[0])

separate = "*" * 60


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--seed", type=int, default=666)
    parser.add_argument("--hidden", action='store_true')
    
    args = parser.parse_args()

    dataset = args.dataset
    seed = args.seed
    dataset_name = get_datasetname(dataset)
    is_small = dataset in uci_small_datasests

    folder = f"{dataset}_stop" if not args.hidden else f"{dataset}_relu"
    
    splits = []
    for d_dir in next(os.walk(os.path.join('log', folder)))[1]:
        if d_dir.startswith(f"{dataset_name}-{seed}-"):
            splits.append(get_split(d_dir))
    
    print(f"dataset {dataset} has {len(splits)} test runs")
    print(f"with splits: {splits}")
    exp_dirs = ["-".join([dataset_name, str(seed), str(split)]) for split in splits]
    res_dir = [os.path.join('log', folder, exp_dir) for exp_dir in exp_dirs]

    def get_test(res_dir, is_small, relu=False):
        """
        test results
        """
        # mean rmse
        rmse_k = "test_rmse_unnormalized"
        if not relu:
            all_rmse = dict()
            for d_dir in res_dir:
                f = open(os.path.join(d_dir, "res.pkl"), 'rb')
                res = pickle.load(f)
                all_rmse[d_dir] = res[rmse_k]
        else:
            all_rmse = dict()
            for d_dir in res_dir:
                rmse_path = os.path.join(d_dir, "unnormalized_rmse.txt")
                if not os.path.isfile(rmse_path):
                    continue
                rmse = get_rmse_results(rmse_path)
                f = open(os.path.join(d_dir, "res.pkl"), 'rb')
                res = pickle.load(f)
                all_rmse[d_dir] = rmse * res["Y_std"]
        print(separate)
        print(f"mean {rmse_k}: {np.mean([v for _, v in all_rmse.items()])}")
        print(f"std {rmse_k}: {np.std([v for _, v in all_rmse.items()])}")
        print(separate)
        for k, v in all_rmse.items():
            print(f"{k}: {v}")

        all_ll = dict()
        likelihood = get_epsilon_results(os.path.join(res_dir[-1], "epsilon.txt"))
        epsilons = likelihood[0]
        for d_dir in res_dir:
            res_path = os.path.join(d_dir, "res.pkl")
            epsilon_path = os.path.join(d_dir, "epsilon.txt")
            if not os.path.isfile(res_path) or not os.path.isfile(epsilon_path):
                continue
            f = open(res_path, 'rb')
            res = pickle.load(f)
            
            likelihood = get_epsilon_results(epsilon_path)
            all_ll[d_dir]= likelihood[1] if not is_small else likelihood[1] - np.log(res['Y_std'])  # normalized for large, unnormalized for small

        print(separate)
        print(f"mean log likelihood: {np.mean([v for _, v in all_ll.items()])} with epsilon: {epsilons}")
        print(f"std: {np.std([v for _, v in all_ll.items()])}")
        print(separate)
        # for d_dir in res_dir:
        for d_dir in all_ll:
            print(f"{d_dir}: {all_ll[d_dir]}")

    get_test(res_dir, is_small, relu=args.hidden)