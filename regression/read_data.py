import pandas as pd
import numpy as np
import pickle
from torch.utils.data import Dataset
from utils import to_variable

from pathlib import Path
import os


RANDOM_SEED = 666
np.random.seed(RANDOM_SEED)

DATA_PATH = "data/"
DATASET_DIMENSIONS = {
    "housing": 13,
    "concrete": 8,
    "energy": 8,
    "naval": 14,
    "yacht": 6,
    "power": 4,
    "wine": 11
}


class MyDataset(Dataset):
    def __init__(self, x, y):
        super(MyDataset, self).__init__()
        assert x.shape[0] == y.shape[0]
        self.x, self.y = to_variable(var=(x, y), cuda=False)
        self.x = self.x.float()
        self.y = self.y.float()

    def __len__(self):
        return self.y.shape[0]

    def __getitem__(self, index):
        return self.x[index], self.y[index]


def prepare_data(dataset_name, regenerate=False):
    save_data_path = DATA_PATH + f'{dataset_name}/'
    # if training set and test set are generated
    if not regenerate and os.path.isfile(save_data_path + 'train.pkl') and \
            os.path.isfile(save_data_path + 'test.pkl'):
        return

    Path(save_data_path).mkdir(parents=True, exist_ok=True)
    args = {'dataset': dataset_name}

    """
    load dataset:
    housing, concrete, energy, power, naval, wine, yacht
    """
    if args['dataset'] == 'housing':
        """# Housing dataset"""
        data = pd.read_csv(f'{DATA_PATH}/housing.data', header=0, delimiter="\s+").values
    elif args['dataset'] == 'concrete':
        """# Concrete compressive dataset"""
        data = pd.read_excel(f'{DATA_PATH}/Concrete_Data.xls', header=0).values
    elif args['dataset'] == 'energy':
        """# Energy efficiency dataset"""
    #     data = pd.read_excel(f'{data_path}/ENB2012_data.xlsx', header=0).values
        data = pd.read_excel(f'{DATA_PATH}/ENB2012_data.xlsx').iloc[:, :-3]
        data = data.values
    elif args['dataset'] == 'power':
        """# Power dataset"""
        data = pd.read_excel(f'{DATA_PATH}/CCPP/Folds5x2_pp.xlsx', header=0).values
    elif args['dataset'] == 'naval':
        """# Naval dataset"""
        data = pd.read_fwf(f'{DATA_PATH}/UCI CBM Dataset/data.txt', header=None).values
        data = np.delete(data, [8, 11, -1], axis=1)
    elif args['dataset'] == 'wine':
        """# Red wine dataset"""
        data = pd.read_csv(f'{DATA_PATH}/winequality-red.csv', header=1, delimiter=';').values
    elif args['dataset'] == 'yacht':
        """# Yacht dataset"""
        data = pd.read_csv(f'{DATA_PATH}/yacht_hydrodynamics.data', header=1, delimiter='\s+').values
    elif args['dataset'] == 'uniform':
        """# synthetic uniform dataset"""
        data = pd.read_csv(f'{DATA_PATH}/uniform.csv').values
    elif args['dataset'] == 'mixture':
        """# synthetic uniform dataset"""
        data = pd.read_csv(f'{DATA_PATH}/mixture.csv').values
    else:
        raise NotImplementedError(f"no dataset {args['dataset']}")

    print(f"import dataset: {args['dataset']}")

    data = data[np.random.permutation(np.arange(len(data)))]
    train_rate = 0.9
    print(f"training data percentage = {train_rate}")

    n_train = int(train_rate * data.shape[0])
    in_dim = data.shape[1] - 1

    train_index = np.array([i for i in range(n_train)])
    test_index = np.array([i for i in range(n_train, data.shape[0])])
    print(f"Train dataset size: {n_train}")
    print(f"Test dataset size: {data.shape[0] - n_train}")

    x_train, y_train = data[train_index, :in_dim], data[train_index, in_dim:]
    x_test, y_test = data[test_index, :in_dim], data[test_index, in_dim:]

    x_means, x_stds = x_train.mean(axis=0), x_train.var(axis=0) ** 0.5
    y_means, y_stds = y_train.mean(axis=0), y_train.var(axis=0) ** 0.5

    x_train = (x_train - x_means) / x_stds
    y_train = (y_train - y_means) / y_stds

    x_test = (x_test - x_means) / x_stds
    y_test = (y_test - y_means) / y_stds

    # Save
    test_dictionary = {
        'y_test': y_test,
        'x_test': x_test,
        'x_stds': x_stds, 'x_means': x_means,
        'y_stds': y_stds, 'y_means': y_means}
    with open(save_data_path + "test.pkl", "wb") as output_file:
        pickle.dump(test_dictionary, output_file)

    train_dictionary = {
        'y_train': y_train,
        'x_train': x_train,
        'x_stds': x_stds, 'x_means': x_means,
        'y_stds': y_stds, 'y_means': y_means}
    with open(save_data_path + "train.pkl", "wb") as output_file:
        pickle.dump(train_dictionary, output_file)

    print("Training and Test data saved.")
    return




